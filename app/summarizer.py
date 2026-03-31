import json
import logging
from app.prompts import build_summarize_prompt
from app.llm_service import call_llm
from app.models import SummarizeRequest, SummarizeResponse

logger = logging.getLogger(__name__)

def patient_data_to_text(patient_data) -> str:
    lines = []
    if patient_data.Age:
        lines.append(f"Age: {patient_data.Age}")
    if patient_data.Gender:
        lines.append(f"Gender: {patient_data.Gender}")
    if patient_data.Diagnosis:
        lines.append(f"Diagnosis: {patient_data.Diagnosis}")
    if patient_data.Symptoms:
        lines.append(f"Symptoms: {', '.join(patient_data.Symptoms)}")
    if patient_data.Medications:
        lines.append(f"Medications: {', '.join(patient_data.Medications)}")
    if patient_data.Surgeries:
        surgeries = ', '.join([f"{s.name} ({s.status})" for s in patient_data.Surgeries])
        lines.append(f"Surgeries: {surgeries}")
    if patient_data.Allergies:
        lines.append(f"Allergies: {', '.join(patient_data.Allergies)}")
    if patient_data.Medical_Warnings:
        lines.append(f"Medical Warnings: {', '.join(patient_data.Medical_Warnings)}")
    if patient_data.Problems:
        lines.append(f"Problems: {', '.join(patient_data.Problems)}")
    if patient_data.Vitals:
        v = patient_data.Vitals
        lines.append(f"Vitals: BP {v.BP}, HR {v.HR}, RR {v.RR}, Temp {v.Temp}, SpO2 {v.SpO2}")
    if patient_data.Lab_Results:
        labs = ', '.join([f"{k}: {v}" for k, v in patient_data.Lab_Results.items()])
        lines.append(f"Lab Results: {labs}")
    return "\n".join(lines)

def summarize(request: SummarizeRequest) -> SummarizeResponse:
    if request.patient_data:
        text = patient_data_to_text(request.patient_data)
    elif request.text:
        text = request.text
    else:
        raise ValueError("Either text or patient_data must be provided")

    logger.info(f"Summarizing text of length {len(text)}")

    prompt = build_summarize_prompt(
        text=text,
        language="english",
        style="neutral"
    )

    result = call_llm(prompt)

    return SummarizeResponse(
        short_summary=result["short_summary"],
        bullet_summary=result["bullet_summary"],
        keywords=result["keywords"],
        confidence_note=result["confidence_note"]
    )