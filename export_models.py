from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
import os
import torch

def export_to_onnx(model_id: str, save_dir: str, is_generator: bool = False):
    """
    Uses optimum to convert a model to ONNX format.
    """
    # This single line handles the conversion of the PyTorch/TF model to ONNX
    # export=True triggers the ONNX conversion automatically
    if not is_generator:
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True, dtype=torch.float32)
    else:
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Save the optimized model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"Success! Model {model_id} saved in ONNX format at {save_dir}")

if __name__ == "__main__":
    # As a retriver a small, efficient model for embedding, suitable for CPU inference and quick testing
    retriever = {"dir": "retriever_onnx", "model_id": "sentence-transformers/all-MiniLM-L6-v2", "is_generator": False}
    # As a generator a lightweight Text-to-Text choice 
    generator = {"dir": "generator_onnx", "model_id": "google/flan-t5-small", "is_generator": True}
    for model_dict in [retriever, generator]:
        dir = model_dict["dir"]
        model_id = model_dict["model_id"]
        is_generator = model_dict["is_generator"]
        if os.path.exists(os.path.join(dir, "model.onnx")):
            print(f"In directory {dir} model is already exported to ONNX format. Skipping export.")
        else:
            print(f"Model will be exported to directory {dir}")
            export_to_onnx(model_id, dir, is_generator)