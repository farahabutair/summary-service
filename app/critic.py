import json
import logging
from app.llm_service import call_llm
from app.models import SummarizeResponse, CritiqueResult

logger = logging.getLogger(__name__)

def build_critique_prompt(summary: SummarizeResponse, original_text: str) -> str:
    return f"""
You are a strict medical summary reviewer. Evaluate the following summary against the original patient data.

Scoring guide:
- 9-10: All critical fields present (diagnosis, medications, vitals, lab results, allergies, warnings, surgeries). Well written and accurate.
- 7-8: Most fields present but 1-2 minor details missing.
- 5-6: Several important fields missing.
- Below 5: Major information missing or inaccurate.

Original patient data:
\"\"\"{original_text}\"\"\"

Summary to evaluate:
- Short summary: {summary.short_summary}
- Bullet points: {summary.bullet_summary}
- Keywords: {summary.keywords}
- Confidence note: {summary.confidence_note}

Return ONLY a JSON object with exactly these fields:
{{
  "score": <number from 0 to 10>,
  "completeness": "feedback on whether all important medical details are covered",
  "hallucination_risk": "feedback on whether anything was added that wasn't in the original",
  "missing_details": "a single string describing any important details that were missed",
  "length_feedback": "feedback on whether the summary is too short, too long, or appropriate",
  "overall_feedback": "one sentence on what needs to improve most"
}}

Return ONLY valid JSON, no markdown, no explanation.
""".strip()

def critique_summary(summary: SummarizeResponse, original_text: str) -> CritiqueResult:
    logger.info("Critiquing summary...")
    prompt = build_critique_prompt(summary, original_text)
    result = call_llm(prompt)
    return CritiqueResult(
        score=result["score"],
        completeness=result["completeness"],
        hallucination_risk=result["hallucination_risk"],
        missing_details=result["missing_details"],
        length_feedback=result["length_feedback"],
        overall_feedback=result["overall_feedback"]
    )