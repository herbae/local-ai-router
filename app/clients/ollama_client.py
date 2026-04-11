import httpx


async def call_ollama(request_data: dict, base_url: str, model: str) -> dict:
    request_data = {**request_data, "model": model, "stream": False}
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json=request_data,
        )
        response.raise_for_status()
        return response.json()
