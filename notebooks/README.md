# Running the notebooks

Notebooks use the uv venv. To start the jyputer lab do:

```uv run --with jupyter jupyter lab```

To start the mlflow server you can just do:

```mlflow server   --host 127.0.0.1   --port 8080   --backend-store-uri ./mlruns   --default-artifact-root ./mlruns```