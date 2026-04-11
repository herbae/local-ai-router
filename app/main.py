import logging
import time

from fastapi import FastAPI, HTTPException

from app.classifier import classify_prompt
from app.clients.anthropic_client import call_claude
from app.clients.ollama_client import call_ollama
from app.config import settings
from app.models import ChatCompletionRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local AI Router")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    last_message = request.messages[-1].content or ""

    route, clean_content = classify_prompt(
        last_message, settings, strip_prefix=True
    )

    request_data = request.model_dump()
    request_data["messages"][-1]["content"] = clean_content

    logger.info(
        "Route: %s | Prompt: %s",
        route,
        clean_content[:80] + ("..." if len(clean_content) > 80 else ""),
    )

    start = time.time()

    try:
        if route == "local":
            result = await call_ollama(
                request_data,
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )
        else:
            result = await call_claude(
                request_data,
                api_key=settings.anthropic_api_key,
                model=settings.claude_model,
            )
    except Exception as e:
        logger.error("Error calling %s: %s", route, e)
        raise HTTPException(status_code=502, detail=f"Upstream error ({route}): {e}")

    elapsed = time.time() - start
    logger.info("Route: %s | Completed in %.2fs", route, elapsed)

    return result


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "local-router",
                "object": "model",
                "created": 1700000000,
                "owned_by": "localrouter",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
