from pydantic import BaseModel, Field
from typing import Optional, List

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw text to summarize")
    language: Optional[str] = "english"
    max_length: Optional[int] = None
    summary_style: Optional[str] = "neutral"

class SummarizeResponse(BaseModel):
    short_summary: str
    bullet_summary: List[str]
    keywords: List[str]
    confidence_note: str