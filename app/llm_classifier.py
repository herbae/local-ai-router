"""LLM-based classifier that asks the local model whether a prompt needs the cloud.

Used as a second-stage check after heuristics. Only runs when the heuristic
classifier would have defaulted to local — it can escalate to cloud but never
the reverse. Falls back to local on any error so the router keeps working.
"""
import logging

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a routing classifier. Analyze the user's prompt and decide if it should be handled by a small local model (LOCAL) or a large cloud model (CLOUD).

LOCAL is appropriate for: simple factual questions, basic translations, short definitions, casual chat, one-sentence answers.

CLOUD is appropriate for: code generation, debugging, complex multi-step reasoning, specialized domain expertise, long-form writing, math beyond arithmetic.

Respond with exactly one word: LOCAL or CLOUD. No explanation, no punctuation."""


async def llm_classify(
    prompt: str,
    base_url: str,
    model: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    timeout: float = 10.0,
) -> str:
    """Ask the local model to classify a prompt as LOCAL or CLOUD.

    Returns "cloud" if the model says CLOUD, otherwise "local" (including on errors).
    """
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json=request,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip().upper()
            first_word = answer.split()[0] if answer.split() else ""
            logger.info("LLM classifier raw output: %r -> first_word=%r", answer, first_word)
            if first_word == "CLOUD":
                return "cloud"
            return "local"
    except Exception as e:
        logger.warning("LLM classifier failed, defaulting to local: %s", e)
        return "local"
