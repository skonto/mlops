uv run uvicorn app.onnx:app --host 0.0.0.0 --port 8000 --workers 1

uv run uvicorn app.torch:app --host 0.0.0.0 --port 8000 --workers 1