import logging
from app.models import SummarizeRequest, SummarizeResponse, AgentResponse, IterationLog
from app.summarizer import summarize
from app.critic import critique_summary
from app.prompts import build_summarize_prompt
from app.llm_service import call_llm

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 9.0
MAX_ITERATIONS = 5

def build_revision_prompt(original_text: str, current_summary: SummarizeResponse, critique) -> str:
    return f"""
You are a medical summarization assistant. You MUST include ALL of the following fields in your revised summary:
- Diagnosis
- All symptoms
- All medications with dosages
- All surgeries and their status
- All allergies
- All medical warnings
- All problems
- All vitals (BP, HR, RR, Temp, SpO2)
- All lab results with values

Original patient data:
\"\"\"{original_text}\"\"\"

Current summary that needs improvement:
- Short summary: {current_summary.short_summary}
- Bullet points: {current_summary.bullet_summary}
- Keywords: {current_summary.keywords}
- Confidence note: {current_summary.confidence_note}

Critique feedback:
- Completeness: {critique.completeness}
- Missing details: {critique.missing_details}
- Overall: {critique.overall_feedback}

Return ONLY a valid JSON object:
{{
  "short_summary": "comprehensive summary including ALL medical fields listed above",
  "bullet_summary": ["point covering diagnosis and vitals", "point covering medications and allergies", "point covering surgeries and warnings and lab results"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "confidence_note": "confidence note"
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