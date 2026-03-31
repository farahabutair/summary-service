def build_summarize_prompt(text: str, language: str = "english", style: str = "neutral") -> str:
    return f"""
You are a summarization assistant. Analyze the following text and return a JSON object with exactly these fields:

{{
  "short_summary": "A concise 1-2 sentence summary",
  "bullet_summary": ["bullet point 1", "bullet point 2", "bullet point 3"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "confidence_note": "A short note about how clear or complete the source text was"
}}

Rules:
- Language: {language}
- Style: {style}
- Return ONLY valid JSON. No explanation, no markdown, no code fences.
- bullet_summary must be a list of strings
- keywords must be a list of strings

Text to summarize:
\"\"\"
{text}
\"\"\"
""".strip()