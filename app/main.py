import logging
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from app.models import SummarizeRequest, SummarizeResponse
from app.summarizer import summarize

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(title="Summary Service", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(request: SummarizeRequest):
    if not request.text and not request.patient_data:
        raise HTTPException(status_code=422, detail="Text field cannot be empty")
    try:
        return summarize(request)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")