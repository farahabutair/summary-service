import logging
from app.prompts import build_summarize_prompt
from app.llm_service import call_llm
from app.models import SummarizeRequest, SummarizeResponse

logger = logging.getLogger(__name__)

def summarize(request: SummarizeRequest) -> SummarizeResponse:
    logger.info(f"Summarizing text of length {len(request.text)}")

    prompt = build_summarize_prompt(
        text=request.text,
        language=request.language or "english",
        style=request.summary_style or "neutral"
    )

    result = call_llm(prompt)

    return SummarizeResponse(
        short_summary=result["short_summary"],
        bullet_summary=result["bullet_summary"],
        keywords=result["keywords"],
        confidence_note=result["confidence_note"]
    )