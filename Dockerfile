# Multi-stage build
# Stage 1: Build the application, export the ONNX model and run tests
FROM python:3.12-slim AS builder

WORKDIR /app

# Install dependencies including the ones needed for testing
# optimum and torch are (required for model export), 
# onnxruntime is required for running the ONNX models, and pytest/httpx are needed for testing
# pypdf and reportlab are needed for PDF handling in tests
RUN pip install --no-cache-dir \
    fastapi uvicorn optimum[onnxruntime] torch onnxruntime transformers \
    numpy python-multipart pypdf reportlab pytest httpx

# Copy the application, the export_model script and the tests
COPY main.py .
COPY export_models.py .
COPY tests/ ./tests/

# Run the export script to create the models files
RUN python export_models.py

# Execute unit tests. If they fail, the build fails here.
RUN python -m pytest tests/test_main.py

# Stage 2: Create the final image with only runtime dependencies and the ONNX models
FROM python:3.12-slim

WORKDIR /app

# Copy only the necessary libraries from the builder stage to keep the final image small
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only the application code and models from the builder stage
COPY main.py .
COPY --from=builder /app/retriever_onnx ./retriever_onnx
COPY --from=builder /app/generator_onnx ./generator_onnx

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]