import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)

MOCK_RESPONSE = {
    "short_summary": "This is a test summary.",
    "bullet_summary": ["Point one", "Point two"],
    "keywords": ["test", "summary"],
    "confidence_note": "High confidence."
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("app.summarizer.call_llm", return_value=MOCK_RESPONSE)
def test_summarize_success(mock_llm):
    response = client.post("/summarize", json={"text": "This is a long enough test text for summarization."})
    assert response.status_code == 200
    data = response.json()
    assert "short_summary" in data
    assert isinstance(data["bullet_summary"], list)
    assert isinstance(data["keywords"], list)
    assert "confidence_note" in data

def test_summarize_missing_text():
    response = client.post("/summarize", json={})
    assert response.status_code == 422

def test_summarize_short_text():
    response = client.post("/summarize", json={"text": "Hi"})
    assert response.status_code == 422