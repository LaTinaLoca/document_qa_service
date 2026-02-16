import os
import io
import sqlite3
import uuid
import numpy as np
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from transformers import AutoTokenizer, pipeline
from functools import lru_cache
from typing import List
from pypdf import PdfReader


# --- Configuration ---
# Get the variables from environment, fallback to local if not set
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DB_PATH = os.getenv("DATABASE_URL", "docs.db")
RETRIEVER_MODEL_PATH = os.getenv("RETRIEVER_MODEL_PATH", "retriever_onnx/model.onnx")
GENERATOR_DIRECTORY_PATH = os.getenv("GENERATOR_DIRECTORY_PATH", "generator_onnx/")
# --- Generator model loading at FastAPI app start-up ---
gen_model = ORTModelForSeq2SeqLM.from_pretrained(GENERATOR_DIRECTORY_PATH, use_cache=True, use_merged=False)
gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_DIRECTORY_PATH)
generator = pipeline("text2text-generation", model=gen_model, tokenizer=gen_tokenizer)
print("Generator is ready!")

# --- Embedding logic with LRU Cache ---
class EmbeddingEngine:
    def __init__(self, model_path: str = RETRIEVER_MODEL_PATH):
        # Initialize ONNX session and Tokenizer
        self.session = ort.InferenceSession(model_path)
        model_dir = os.path.dirname(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Last Recently Used Cache: to skip the onnxruntime matrix multiplication for text already seen before so 
    # reducing computational load and response latency. 
    # Value set to 1024 to use a reasonable amount of RAM for the this use case given the retriever model embedding dimension (384).
    @lru_cache(maxsize=1024)
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Computes the embedding. Results are cached in memory.
        It stores the 1024 most recent unique chunks.
        """
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        # Ensure inputs are int64 for ONNX compatibility
        ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        ort_outs = self.session.run(None, ort_inputs)
        # Mean Pooling: take average of token embeddings
        embeddings = np.mean(ort_outs[0], axis=1)
        return embeddings[0].astype(np.float32)

engine = EmbeddingEngine()

def get_engine():
    return engine

app = FastAPI(title="Document Q&A service", 
              description="Upload documents and ask questions about them. " \
              "Uses Python-based FastAPI, ONNX for embedding and SQLite for docs storage.",)

#--- Database Initialization ---
# SQLite database: it creates the necessary table if it doesn't exist
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                filename TEXT,
                content TEXT,
                embedding BLOB
            )
        """)
init_db()

# --- Utility function ---
# To overcome the token limit of the retriever model and make sure the whole document is stored
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Spits the uploaded text into chunks of a given size, 
    It applies an overlap to make sure context is preserved.
    """
    chunks = []
    if len(text) <= size:
        return [text]
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i : i + size])
    return chunks

# --- Endpoints ---
@app.post("/upload", summary="Upload a .pdf or .txt document and embed it for future Q&A")
async def upload_document(file: UploadFile = File(...),
                           eng: EmbeddingEngine = Depends(get_engine)):
    content = ""
    extension = file.filename.split('.')[-1].lower()

    if extension == "pdf":
        pdf_content = await file.read()
        reader = PdfReader(io.BytesIO(pdf_content))
        content = " ".join([page.extract_text() for page in reader.pages])
    elif extension == "txt":
        content = (await file.read()).decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt are supported")
    
    chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Store in SQLite with a unique ID and the filename for reference
    with sqlite3.connect(DB_PATH) as conn:
        for chunk in chunks:
            chunk_embedding = eng.get_embedding(chunk)
            conn.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", 
                         (str(uuid.uuid4().hex), file.filename, chunk, chunk_embedding.tobytes()))
    
    return {"status": "success", "message": f"Processed {file.filename} into {len(chunks)} chunks."}

@app.post("/ask", summary="Ask a question about the uploaded documents")
async def ask_question(question: str,
                       eng: EmbeddingEngine = Depends(get_engine)):

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT filename, content, embedding FROM chunks")
        results = cursor.fetchall()
    
    if not results:
        raise HTTPException(status_code=404, detail="No documents found.")

    q_embedding = eng.get_embedding(question)

    # vector search
    best_doc = None
    max_sim = -1

    for filename, content, emb_blob in results:
        doc_emb = np.frombuffer(emb_blob, dtype=np.float32)
        # Simple Cosine Similarity: dot(a, b) / (||a|| * ||b||)
        sim = np.dot(q_embedding, doc_emb) / (np.linalg.norm(q_embedding) * np.linalg.norm(doc_emb))
        if sim > max_sim:
            max_sim = sim
            best_doc = filename
            rag_answer = content

    # Feed the best chunk to the generator to get a more "friendly" answer for the user, 
    # instead of just returning the raw document text.
    prompt = f"Answer the question based on the context.\nContext: {rag_answer}\nQuestion: {question}"
    
    # Generate the friendly answer
    output = generator(prompt, max_length=100, clean_up_tokenization_spaces=True)
    friendly_answer = output[0]['generated_text']

    return {
        "question": question,
        "answer": rag_answer[:100] + "..." if len(rag_answer) > 100 else rag_answer, # Returns a snippet of the context
        "friendly_answer": friendly_answer,
        "source": best_doc,
        "confidence": float(max_sim)
    }