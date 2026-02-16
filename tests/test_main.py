import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app, init_db, chunk_text, EmbeddingEngine
from reportlab.pdfgen import canvas
import io
import sqlite3

@pytest.fixture(autouse=True)
def cleanup_database():
    """
    Ensures each test starts with a fresh in-memory database.
    """
    # Creates in-memory connection
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, filename TEXT, content TEXT, embedding BLOB)"
    )
    # Overrides the production DB dependency
    def override_init_db():
        yield connection
    app.dependency_overrides[init_db] = override_init_db

    yield  # --- The tests run here ---

    # Clears overrides and close connection
    app.dependency_overrides.clear()
    connection.close()

@pytest.fixture(autouse=True)
def clear_engine_cache():
    """
    Clears the embedding cache before each test to ensure test isolation.
    """
    yield
    if hasattr(EmbeddingEngine.get_embedding, "cache_clear"):
        EmbeddingEngine.get_embedding.cache_clear()

client = TestClient(app)

def create_test_pdf(content: str) -> bytes:
    """
    Helper to create a valid pdf in memory.
    This ensures that our upload endpoint can process it as a real pdf.
    """
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(100, 750, content)
    p.showPage()
    p.save()
    return buffer.getvalue()

def test_chunk_text_logic():
    text = "This is a long sentence that should be split into multiple chunks."
    # Size 20, Overlap 5
    chunks = chunk_text(text, size=20, overlap=5)
    assert len(chunks) > 1
    assert chunks[0] == text[:20]
    # Check if overlap exists in the second chunk
    assert text[15:20] in chunks[1]

def test_invalid_file_type():
    # Test that uploading an image fails as expected
    files = {"file": ("image.png", b"not-an-image", "image/png")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Only .pdf and .txt are supported" in response.json()["detail"] 

def test_ask_no_docs():
    response = client.post("/ask", params={"question": "What is the answer?"})
    assert response.status_code == 404
    assert response.json()["detail"] == "No documents found."

def test_upload():
    response = client.post("/upload", files={"file": ("test.txt", "hello", "text/plain")})
    assert response.status_code == 200
    assert "Processed test.txt" in response.json()["message"]

def test_full_flow():
    pdf_bytes = create_test_pdf("The answer to the ultimate question of life, the universe, and everything is 42.")
    files = {"file": ("secret.pdf", pdf_bytes, "application/pdf")}
    # Upload
    upload_response = client.post("/upload", files=files)
    assert upload_response.status_code == 200
    assert "Processed secret.pdf" in upload_response.json()["message"]
    # Ask
    ask_response = client.post("/ask", params={"question": "What is answer to the ultimate question of life?"})
    assert ask_response.status_code == 200
    assert ask_response.json()["confidence"] > 0.5
    assert "42" in ask_response.json()["answer"]
    assert "42" in ask_response.json()["friendly_answer"]
    assert ask_response.json()["source"] == "secret.pdf"
