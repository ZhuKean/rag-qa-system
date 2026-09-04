from fastapi import FastAPI

app = FastAPI(title="RAG QA system", version="0.1.0")

@app.get("/healthcheck")
def healthcheck() -> dict:
    return {"status": "ok"}
