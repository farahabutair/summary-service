import os
import json
import logging
from dotenv import load_dotenv
from groq import Groq
import groq as groq_lib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((groq_lib.APIConnectionError, groq_lib.RateLimitError)),
    reraise=True
)
def call_llm(prompt: str, timeout: int = 30) -> dict:
    logger.info("Calling LLM...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()
    logger.info("LLM responded successfully")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON: {raw}")
        raise ValueError(f"LLM returned invalid JSON: {e}")