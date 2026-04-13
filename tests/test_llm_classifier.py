from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm_classifier import llm_classify


def fake_response(text: str):
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }
    response.raise_for_status = MagicMock()
    return response


@pytest.mark.asyncio
async def test_returns_cloud_when_model_says_cloud():
    fake_post = AsyncMock(return_value=fake_response("CLOUD"))
    with patch("httpx.AsyncClient.post", fake_post):
        result = await llm_classify("anything", "http://ollama:11434", "gemma4:e4b")
    assert result == "cloud"


@pytest.mark.asyncio
async def test_returns_local_when_model_says_local():
    fake_post = AsyncMock(return_value=fake_response("LOCAL"))
    with patch("httpx.AsyncClient.post", fake_post):
        result = await llm_classify("anything", "http://ollama:11434", "gemma4:e4b")
    assert result == "local"


@pytest.mark.asyncio
async def test_handles_lowercase_response():
    fake_post = AsyncMock(return_value=fake_response("cloud"))
    with patch("httpx.AsyncClient.post", fake_post):
        result = await llm_classify("anything", "http://ollama:11434", "gemma4:e4b")
    assert result == "cloud"


@pytest.mark.asyncio
async def test_handles_extra_whitespace_and_extra_words():
    fake_post = AsyncMock(return_value=fake_response("  CLOUD because complex"))
    with patch("httpx.AsyncClient.post", fake_post):
        result = await llm_classify("anything", "http://ollama:11434", "gemma4:e4b")
    assert result == "cloud"


@pytest.mark.asyncio
async def test_unknown_response_defaults_to_local():
    fake_post = AsyncMock(return_value=fake_response("MAYBE"))
    with patch("httpx.AsyncClient.post", fake_post):
        result = await llm_classify("anything", "http://ollama:11434", "gemma4:e4b")
    assert result == "local"


@pytest.mark.asyncio
async def test_http_error_falls_back_to_local():
    fake_post = AsyncMock(side_effect=Exception("connection refused"))
    with patch("httpx.AsyncClient.post", fake_post):
        result = await llm_classify("anything", "http://ollama:11434", "gemma4:e4b")
    assert result == "local"
