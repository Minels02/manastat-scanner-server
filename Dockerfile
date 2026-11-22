FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scanner_server.py build_index.py mobilenet_v3_small.onnx mobilenet_v3_small.onnx.data vectors.npy metadata.json ./

EXPOSE 7860

CMD ["uvicorn", "scanner_server:app", "--host", "0.0.0.0", "--port", "7860"]