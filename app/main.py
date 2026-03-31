import logging
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from app.models import SummarizeRequest, SummarizeResponse, AgentResponse
from app.summarizer import summarize
from app.agent import run_agent
from app.summarizer import patient_data_to_text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(title="Summary Service", version="2.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(request: SummarizeRequest):
    if not request.text and not request.patient_data:
        raise HTTPException(status_code=422, detail="Either text or patient_data must be provided")
    try:
        return summarize(request)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/summarize/agent", response_model=AgentResponse)
def agent_endpoint(request: SummarizeRequest):
    if not request.text and not request.patient_data:
        raise HTTPException(status_code=422, detail="Either text or patient_data must be provided")
    try:
        if request.patient_data:
            original_text = patient_data_to_text(request.patient_data)
        else:
            original_text = request.text
        return run_agent(request, original_text)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")