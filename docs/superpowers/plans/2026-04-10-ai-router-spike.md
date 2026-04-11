# AI Router Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI-based AI router that exposes an OpenAI-compatible API and routes prompts between Ollama (local) and Claude (cloud), accessible through Open WebUI — all running in Docker Compose.

**Architecture:** FastAPI server at `/v1/chat/completions` receives requests from Open WebUI, classifies the prompt using simple heuristics (length + code detection + manual `/local` `/cloud` overrides), and forwards to either Ollama's OpenAI-compatible endpoint or the Anthropic Messages API (with format translation). Non-streaming only for this spike.

**Tech Stack:** Python 3.12, FastAPI, httpx, anthropic SDK, PyYAML, Pydantic, Docker Compose, Ollama, Open WebUI

**Hardware:** Ubuntu 24.04, i7-11800H, 64GB RAM, RTX 3060 Laptop 6GB VRAM

---

## File Structure

```
localrouter/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, /v1/chat/completions endpoint
│   ├── config.py                # Config loading from YAML + env
│   ├── classifier.py            # Simple heuristic routing classifier
│   ├── models.py                # Pydantic models for OpenAI chat format
│   └── clients/
│       ├── __init__.py
│       ├── ollama_client.py     # Async httpx client for Ollama
│       └── anthropic_client.py  # Anthropic SDK client + format translation
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py       # Classifier unit tests
│   └── test_anthropic_translator.py  # Translation logic tests
├── config.yaml                  # Routing config (models, thresholds, patterns)
├── .env                         # ANTHROPIC_API_KEY (user-provided)
├── .env.example                 # Template for .env
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Router container image
├── docker-compose.yml           # Router + Ollama + Open WebUI
└── scripts/
    └── bootstrap.sh             # Pulls model in Ollama container
```

---

## Task 0: GPU + Ollama Validation (Spike 0)

**This is a pass/fail gate.** If the model is too slow, stop and reassess before building the router.

- [ ] **Step 1: Verify NVIDIA driver is working**

```bash
nvidia-smi
```

Expected: Output showing your RTX 3060 with driver version and CUDA version. If this fails, install NVIDIA drivers first — nothing else will work.

- [ ] **Step 2: Check if nvidia-container-toolkit is installed**

```bash
dpkg -l | grep nvidia-container-toolkit
```

If not installed, install it:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

- [ ] **Step 3: Verify GPU is visible inside Docker**

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

Expected: Same nvidia-smi output as Step 1, but from inside the container. If this fails, the toolkit install didn't work — debug before continuing.

- [ ] **Step 4: Start Ollama container with GPU**

```bash
docker run -d --gpus all -v ollama_data:/root/.ollama -p 11434:11434 --name ollama-test ollama/ollama
```

Wait a few seconds, then verify it's running:

```bash
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

Expected: JSON response with `"models": []` (empty, no models pulled yet).

- [ ] **Step 5: Find the correct Gemma 4 E4B model tag**

```bash
docker exec ollama-test ollama search gemma4
```

If `search` is not available in your Ollama version, try:

```bash
docker exec ollama-test ollama list
# And check https://ollama.com/library for "gemma4" to find the exact tag
```

Note the exact model tag (e.g., `gemma4:e4b`, `gemma4-e4b`, etc.). If Gemma 4 E4B is not available yet, use `gemma3:4b` as the fallback — it's proven on low VRAM.

- [ ] **Step 6: Pull the model and test speed**

```bash
# Replace with the correct tag from Step 5
docker exec ollama-test ollama pull gemma3:4b
```

Wait for download to complete (3-5GB depending on model). Then test:

```bash
time curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:4b",
    "messages": [{"role": "user", "content": "What is the capital of Albania?"}],
    "stream": false
  }' | python3 -m json.tool
```

**Pass criteria:** Response completes in <5 seconds with a correct answer ("Tirana").

**If it fails (too slow or errors):**
- Check `docker logs ollama-test` for GPU detection
- Try a smaller model: `ollama pull gemma3:1b`
- If GPU is not detected, try adding `--runtime=nvidia` to the docker run command

- [ ] **Step 7: Clean up test container**

```bash
docker stop ollama-test && docker rm ollama-test
```

Keep the `ollama_data` volume — it has the downloaded model and will be reused by Docker Compose.

- [ ] **Step 8: Record the working model name**

Update `config.yaml` (created in Task 1) with whichever model tag actually worked. Note the response time for reference.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `.env.example`
- Create: `.env`
- Create: `app/__init__.py`
- Create: `app/clients/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Initialize git repo**

```bash
cd /home/iury-lazoski/projects/aispikes/localrouter
git init
```

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p app/clients tests scripts
```

- [ ] **Step 3: Create requirements.txt**

Create `requirements.txt`:

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
httpx==0.28.0
anthropic==0.42.0
pyyaml==6.0.2
pydantic==2.10.0
pytest==8.3.0
pytest-asyncio==0.24.0
```

- [ ] **Step 4: Create config.yaml**

Create `config.yaml`:

```yaml
ollama:
  base_url: "http://ollama:11434"
  model: "gemma3:4b"  # Update with the model tag from Task 0, Step 5

claude:
  model: "claude-sonnet-4-20250514"

classifier:
  max_local_length: 500
  code_patterns:
    - "```"
    - "def "
    - "class "
    - "import "
    - "function "
    - "const "
    - "var "
    - "SELECT "
    - "CREATE TABLE"
    - "async "
    - "await "
```

- [ ] **Step 5: Create .env.example and .env**

Create `.env.example`:

```
ANTHROPIC_API_KEY=your-api-key-here
```

Create `.env`:

```
ANTHROPIC_API_KEY=your-api-key-here
```

- [ ] **Step 6: Create empty __init__.py files**

Create `app/__init__.py` (empty file).
Create `app/clients/__init__.py` (empty file).
Create `tests/__init__.py` (empty file).

- [ ] **Step 7: Create .gitignore**

Create `.gitignore`:

```
__pycache__/
*.pyc
.env
.venv/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 8: Commit scaffolding**

```bash
git add requirements.txt config.yaml .env.example .gitignore app/__init__.py app/clients/__init__.py tests/__init__.py
git commit -m "chore: project scaffolding with deps and config"
```

---

## Task 2: Pydantic Models for OpenAI Chat Format

**Files:**
- Create: `app/models.py`

- [ ] **Step 1: Create the models**

Create `app/models.py`:

```python
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "local-router"
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
```

- [ ] **Step 2: Commit**

```bash
git add app/models.py
git commit -m "feat: add Pydantic models for OpenAI chat completion format"
```

---

## Task 3: Config Loader

**Files:**
- Create: `app/config.py`

- [ ] **Step 1: Create config loader**

Create `app/config.py`:

```python
import os
from pathlib import Path

import yaml


class Settings:
    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.ollama_base_url = self.config["ollama"]["base_url"]
        self.ollama_model = self.config["ollama"]["model"]
        self.claude_model = self.config["claude"]["model"]

        classifier = self.config["classifier"]
        self.max_local_length = classifier["max_local_length"]
        self.code_patterns = classifier["code_patterns"]


settings = Settings()
```

- [ ] **Step 2: Commit**

```bash
git add app/config.py
git commit -m "feat: add config loader for YAML + env"
```

---

## Task 4: Classifier with Tests

**Files:**
- Create: `app/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_classifier.py`:

```python
from app.classifier import classify_prompt


class FakeSettings:
    max_local_length = 500
    code_patterns = ["```", "def ", "class ", "import ", "function "]


settings = FakeSettings()


def test_cloud_prefix_routes_to_cloud():
    assert classify_prompt("/cloud explain quantum physics", settings) == "cloud"


def test_local_prefix_routes_to_local():
    assert classify_prompt("/local what is 2+2", settings) == "local"


def test_short_simple_prompt_routes_to_local():
    assert classify_prompt("What is the capital of Albania?", settings) == "local"


def test_prompt_with_code_routes_to_cloud():
    assert classify_prompt("def fibonacci(n):\n    pass", settings) == "cloud"


def test_prompt_with_code_block_routes_to_cloud():
    assert classify_prompt("Fix this:\n```python\nprint('hello')\n```", settings) == "cloud"


def test_long_prompt_routes_to_cloud():
    long_prompt = "Explain in detail " + "word " * 200
    assert classify_prompt(long_prompt, settings) == "cloud"


def test_short_prompt_without_code_routes_to_local():
    assert classify_prompt("Translate hello to French", settings) == "local"


def test_prefix_stripped_content():
    route, content = classify_prompt("/cloud tell me a joke", settings, strip_prefix=True)
    assert route == "cloud"
    assert content == "tell me a joke"


def test_no_prefix_content_unchanged():
    route, content = classify_prompt("hello world", settings, strip_prefix=True)
    assert route == "local"
    assert content == "hello world"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/iury-lazoski/projects/aispikes/localrouter
python -m pytest tests/test_classifier.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.classifier'`

- [ ] **Step 3: Implement the classifier**

Create `app/classifier.py`:

```python
def classify_prompt(
    message: str, settings, strip_prefix: bool = False
) -> str | tuple[str, str]:
    content = message

    # Manual overrides
    if message.startswith("/cloud"):
        content = message[len("/cloud"):].lstrip()
        if strip_prefix:
            return "cloud", content
        return "cloud"

    if message.startswith("/local"):
        content = message[len("/local"):].lstrip()
        if strip_prefix:
            return "local", content
        return "local"

    # Code detection
    for pattern in settings.code_patterns:
        if pattern in message:
            if strip_prefix:
                return "cloud", content
            return "cloud"

    # Length check
    if len(message) > settings.max_local_length:
        if strip_prefix:
            return "cloud", content
        return "cloud"

    if strip_prefix:
        return "local", content
    return "local"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_classifier.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/classifier.py tests/test_classifier.py
git commit -m "feat: add simple heuristic classifier with /local /cloud overrides"
```

---

## Task 5: Ollama Client

**Files:**
- Create: `app/clients/ollama_client.py`

- [ ] **Step 1: Create the Ollama client**

Create `app/clients/ollama_client.py`:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add app/clients/ollama_client.py
git commit -m "feat: add async Ollama client via OpenAI-compatible endpoint"
```

---

## Task 6: Anthropic Client + Translator with Tests

**Files:**
- Create: `app/clients/anthropic_client.py`
- Create: `tests/test_anthropic_translator.py`

This is the hardest part of the spike — translating between OpenAI and Anthropic formats. The key differences:
- **System prompt:** OpenAI puts it in `messages` with `role: "system"`. Anthropic uses a separate top-level `system` field.
- **Response format:** Anthropic `content` is an array of blocks. OpenAI `choices[0].message.content` is a string.
- **Field names:** `stop` vs `stop_sequences`, `input_tokens` vs `prompt_tokens`, `stop_reason` vs `finish_reason`.
- **max_tokens:** Required in Anthropic, optional in OpenAI.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_anthropic_translator.py`:

```python
from app.clients.anthropic_client import translate_to_anthropic, translate_from_anthropic


def test_system_message_extracted():
    openai_request = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 100,
    }
    result = translate_to_anthropic(openai_request, model="claude-sonnet-4-20250514")
    assert result["system"] == "You are helpful."
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"


def test_multiple_system_messages_concatenated():
    openai_request = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 100,
    }
    result = translate_to_anthropic(openai_request, model="claude-sonnet-4-20250514")
    assert result["system"] == "You are helpful.\n\nBe concise."


def test_no_system_message():
    openai_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
    }
    result = translate_to_anthropic(openai_request, model="claude-sonnet-4-20250514")
    assert "system" not in result


def test_max_tokens_default_when_missing():
    openai_request = {
        "messages": [{"role": "user", "content": "Hello"}],
    }
    result = translate_to_anthropic(openai_request, model="claude-sonnet-4-20250514")
    assert result["max_tokens"] == 4096


def test_stop_string_wrapped_in_list():
    openai_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "stop": "\n",
    }
    result = translate_to_anthropic(openai_request, model="claude-sonnet-4-20250514")
    assert result["stop_sequences"] == ["\n"]


def test_stop_list_passed_through():
    openai_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "stop": ["\n", "END"],
    }
    result = translate_to_anthropic(openai_request, model="claude-sonnet-4-20250514")
    assert result["stop_sequences"] == ["\n", "END"]


def test_temperature_clamped_to_one():
    openai_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 1.5,
    }
    result = translate_to_anthropic(openai_request, model="claude-sonnet-4-20250514")
    assert result["temperature"] == 1.0


def test_response_translation():
    # Simulate an Anthropic API response object
    class ContentBlock:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class UsageObj:
        input_tokens = 10
        output_tokens = 20

    class FakeResponse:
        id = "msg_abc123"
        model = "claude-sonnet-4-20250514"
        content = [ContentBlock("Tirana is the capital of Albania.")]
        stop_reason = "end_turn"
        usage = UsageObj()

    result = translate_from_anthropic(FakeResponse())
    assert result["id"] == "chatcmpl-msg_abc123"
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "Tirana is the capital of Albania."
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 20
    assert result["usage"]["total_tokens"] == 30


def test_stop_reason_mapping():
    class ContentBlock:
        type = "text"
        text = ""

    class UsageObj:
        input_tokens = 0
        output_tokens = 0

    class FakeResponse:
        id = "msg_x"
        model = "claude-sonnet-4-20250514"
        content = [ContentBlock()]
        usage = UsageObj()

    for anthropic_reason, openai_reason in [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("stop_sequence", "stop"),
    ]:
        FakeResponse.stop_reason = anthropic_reason
        result = translate_from_anthropic(FakeResponse())
        assert result["choices"][0]["finish_reason"] == openai_reason
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_anthropic_translator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.clients.anthropic_client'`

- [ ] **Step 3: Implement the translator and client**

Create `app/clients/anthropic_client.py`:

```python
import time

import anthropic


def translate_to_anthropic(openai_request: dict, model: str) -> dict:
    messages = openai_request.get("messages", [])
    system_parts = []
    anthropic_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        else:
            anthropic_messages.append(
                {"role": msg["role"], "content": msg["content"]}
            )

    result = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": openai_request.get("max_tokens") or 4096,
    }

    if system_parts:
        result["system"] = "\n\n".join(system_parts)

    temperature = openai_request.get("temperature")
    if temperature is not None:
        result["temperature"] = min(float(temperature), 1.0)

    stop = openai_request.get("stop")
    if stop is not None:
        result["stop_sequences"] = [stop] if isinstance(stop, str) else stop

    return result


def translate_from_anthropic(anthropic_response) -> dict:
    content = ""
    for block in anthropic_response.content:
        if block.type == "text":
            content += block.text

    stop_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }

    return {
        "id": f"chatcmpl-{anthropic_response.id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": anthropic_response.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": stop_reason_map.get(
                    anthropic_response.stop_reason, "stop"
                ),
            }
        ],
        "usage": {
            "prompt_tokens": anthropic_response.usage.input_tokens,
            "completion_tokens": anthropic_response.usage.output_tokens,
            "total_tokens": (
                anthropic_response.usage.input_tokens
                + anthropic_response.usage.output_tokens
            ),
        },
    }


async def call_claude(request_data: dict, api_key: str, model: str) -> dict:
    client = anthropic.AsyncAnthropic(api_key=api_key)
    anthropic_request = translate_to_anthropic(request_data, model=model)
    response = await client.messages.create(**anthropic_request)
    return translate_from_anthropic(response)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_anthropic_translator.py -v
```

Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/clients/anthropic_client.py tests/test_anthropic_translator.py
git commit -m "feat: add Anthropic client with OpenAI format translation"
```

---

## Task 7: FastAPI Router Endpoint

**Files:**
- Create: `app/main.py`

- [ ] **Step 1: Create the FastAPI app**

Create `app/main.py`:

```python
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
```

- [ ] **Step 2: Quick smoke test with uvicorn**

```bash
cd /home/iury-lazoski/projects/aispikes/localrouter
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 2
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8000/v1/models | python3 -m json.tool
kill %1
```

Expected: `{"status": "ok"}` and a models list with `"local-router"`.

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat: add FastAPI router with /v1/chat/completions endpoint"
```

---

## Task 8: Dockerfile + Docker Compose

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create the Dockerfile**

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create docker-compose.yml**

Create `docker-compose.yml`:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

  router:
    build: .
    container_name: localrouter
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - ollama
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:v0.6.5
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      OPENAI_API_BASE_URL: "http://localrouter:8000/v1"
      OPENAI_API_KEY: "sk-dummy"
      ENABLE_OLLAMA_API: "false"
      OLLAMA_BASE_URL: ""
      WEBUI_AUTH: "false"
    volumes:
      - open-webui_data:/app/backend/data
    depends_on:
      - router
    restart: unless-stopped

volumes:
  ollama_data:
  open-webui_data:
```

Note: The `ollama_data` volume will reuse the model downloaded in Task 0 if you used the same volume name. If Task 0 used a different volume name (e.g., `ollama_data` via `-v ollama_data:/root/.ollama`), the model is already there. Otherwise, you'll need to pull it again via the bootstrap script.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: add Dockerfile and Docker Compose with Ollama + Open WebUI"
```

---

## Task 9: Bootstrap Script + End-to-End Test

**Files:**
- Create: `scripts/bootstrap.sh`

- [ ] **Step 1: Create the bootstrap script**

Create `scripts/bootstrap.sh`:

```bash
#!/bin/bash
set -e

MODEL="${1:-gemma3:4b}"

echo "=== AI Router Bootstrap ==="
echo "Waiting for Ollama to be ready..."

until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "  Ollama not ready, retrying in 3s..."
    sleep 3
done

echo "Ollama is ready!"
echo "Pulling model: $MODEL"
docker exec ollama ollama pull "$MODEL"
echo "Model $MODEL pulled successfully!"
echo ""
echo "=== Bootstrap complete ==="
echo "Open WebUI: http://localhost:3000"
echo "Router API: http://localhost:8000"
echo "Ollama API: http://localhost:11434"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/bootstrap.sh
```

- [ ] **Step 3: Put your API key in .env**

```bash
# Edit .env and replace with your real Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-your-real-key-here" > .env
```

- [ ] **Step 4: Launch everything**

```bash
cd /home/iury-lazoski/projects/aispikes/localrouter
docker compose up -d --build
```

Wait for containers to start:

```bash
docker compose ps
```

Expected: All 3 services (ollama, localrouter, open-webui) showing as "running".

- [ ] **Step 5: Run the bootstrap script to pull the model**

```bash
./scripts/bootstrap.sh gemma3:4b
```

Replace `gemma3:4b` with whichever model worked in Task 0. Wait for the pull to complete.

- [ ] **Step 6: Test the router via curl**

Test local routing (simple question):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-router",
    "messages": [{"role": "user", "content": "What is the capital of Albania?"}],
    "stream": false
  }' | python3 -m json.tool
```

Expected: Response from Ollama with "Tirana". Check router logs for `Route: local`.

Test cloud routing (code question):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-router",
    "messages": [{"role": "user", "content": "def fibonacci(n): write this function in Python with memoization"}],
    "stream": false
  }' | python3 -m json.tool
```

Expected: Response from Claude with a Python implementation. Check router logs for `Route: cloud`.

Test manual override:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-router",
    "messages": [{"role": "user", "content": "/cloud What is the capital of Albania?"}],
    "stream": false
  }' | python3 -m json.tool
```

Expected: Response from Claude (forced cloud). Check router logs for `Route: cloud`.

- [ ] **Step 7: Test through Open WebUI**

Open your browser to `http://localhost:3000`. You should see the Open WebUI chat interface.

1. Select "local-router" from the model dropdown (it should appear automatically)
2. Type: "What is the capital of Albania?" — should route locally (fast)
3. Type: "Write a Python function that implements binary search" — should route to Claude (code detected)
4. Type: "/local Write a Python function that prints hello" — should route locally (manual override)
5. Check router logs: `docker compose logs router -f`

- [ ] **Step 8: Commit everything**

```bash
git add scripts/bootstrap.sh
git commit -m "feat: add bootstrap script and complete spike"
```

- [ ] **Step 9: Verify all tests still pass**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

---

## Deferred to Iteration (not in this spike)

- Streaming SSE support for both Ollama and Claude backends
- Sophisticated classifier with tiktoken token counting
- Multi-step reasoning detection in classifier
- Config YAML for classifier thresholds (hot reload)
- README with full setup instructions
- Usage/cost tracking and structured logging
- Error retry logic for upstream failures
- Model fallback (if Ollama fails, try Claude automatically)
- Conversation history truncation for long chats
