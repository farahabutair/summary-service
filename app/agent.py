import logging
from app.models import SummarizeRequest, SummarizeResponse, AgentResponse, IterationLog
from app.summarizer import summarize
from app.critic import critique_summary
from app.prompts import build_summarize_prompt
from app.llm_service import call_llm

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 9.0
MAX_ITERATIONS = 3

def build_revision_prompt(original_text: str, current_summary: SummarizeResponse, critique) -> str:
    return f"""
You are a medical summarization assistant. Revise the following summary based on the critique provided.

Original patient data:
\"\"\"{original_text}\"\"\"

Current summary:
- Short summary: {current_summary.short_summary}
- Bullet points: {current_summary.bullet_summary}
- Keywords: {current_summary.keywords}
- Confidence note: {current_summary.confidence_note}

Critique feedback:
- Completeness: {critique.completeness}
- Hallucination risk: {critique.hallucination_risk}
- Missing details: {critique.missing_details}
- Length feedback: {critique.length_feedback}
- Overall: {critique.overall_feedback}

Return ONLY a valid JSON object with exactly these fields:
{{
  "short_summary": "revised concise summary",
  "bullet_summary": ["revised point 1", "revised point 2", "revised point 3"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "confidence_note": "updated confidence note"
}}

No markdown, no explanation, only JSON.
""".strip()

def run_agent(request: SummarizeRequest, original_text: str) -> AgentResponse:
    logs = []
    current_summary = summarize(request)
    final_score = 0.0
    stopped_early = False

    for i in range(1, MAX_ITERATIONS + 1):
        logger.info(f"Agent iteration {i}")

        critique = critique_summary(current_summary, original_text)
        logger.info(f"Iteration {i} score: {critique.score}")

        logs.append(IterationLog(
            iteration=i,
            draft=current_summary,
            critique=critique
        ))

        if critique.score >= QUALITY_THRESHOLD:
            final_score = critique.score
            stopped_early = True
            logger.info(f"Quality threshold reached at iteration {i}")
            break

        if i < MAX_ITERATIONS:
            revision_prompt = build_revision_prompt(original_text, current_summary, critique)
            revised = call_llm(revision_prompt)
            current_summary = SummarizeResponse(
                short_summary=revised["short_summary"],
                bullet_summary=revised["bullet_summary"],
                keywords=revised["keywords"],
                confidence_note=revised["confidence_note"]
            )
            final_score = critique.score

    return AgentResponse(
        final_summary=current_summary,
        iterations_run=len(logs),
        final_score=final_score,
        stopped_early=stopped_early,
        logs=logs
    )