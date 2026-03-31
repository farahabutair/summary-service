def build_summarize_prompt(text: str, language: str = "english", style: str = "neutral") -> str:
    return f"""
You are a medical summarization assistant. You MUST include ALL of the following in your summary:
- Patient age and gender
- Primary diagnosis
- All symptoms
- All medications with exact dosages
- All surgeries with their status
- All allergies
- All medical warnings
- All chronic problems
- All vitals (BP, HR, RR, Temp, SpO2)
- All lab results with exact values

Return ONLY a JSON object with exactly these fields:
{{
  "short_summary": "A comprehensive 3-4 sentence summary covering diagnosis, medications, vitals, lab results, allergies and warnings",
  "bullet_summary": ["bullet covering diagnosis and symptoms", "bullet covering all medications and allergies", "bullet covering vitals, lab results and surgeries"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "confidence_note": "confidence note about the source data quality"
}}

Rules:
- Language: {language}
- Style: {style}
- Return ONLY valid JSON. No explanation, no markdown, no code fences.

Text to summarize:
\"\"\"
{text}
\"\"\"
""".strip()