# Local AI Router

A FastAPI service that exposes an OpenAI-compatible API and routes prompts between a local model (Ollama + Gemma 4 E4B) and the Claude API based on prompt complexity. Designed to run on a developer laptop with a modest GPU.

## How it works

The router sits behind a standard `POST /v1/chat/completions` endpoint. A built-in heuristic classifier analyzes each prompt and decides where to send it:

- **Local (Gemma 4 E4B via Ollama)** — short prompts, translation, simple Q&A
- **Cloud (Claude Sonnet via Anthropic API)** — code generation, long prompts, complex reasoning

You can also force a route by prefixing your prompt with `/cloud` or `/local`.

## Stack

- **Python 3.12**, FastAPI, httpx, anthropic SDK, Pydantic
- **Ollama** with Gemma 4 E4B for local inference (GPU-accelerated)
- **Open WebUI** as the chat frontend
- **Docker Compose** to orchestrate all three services

## Requirements

- Ubuntu (or another Linux with NVIDIA drivers)
- Docker and Docker Compose
- NVIDIA Container Toolkit ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- An NVIDIA GPU with at least 6 GB VRAM (developed on RTX 3060 Laptop)
- An Anthropic API key with credits

## Quick start

1. **Verify GPU is visible inside Docker:**

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```

   You should see your GPU listed. If not, fix your NVIDIA Container Toolkit setup before continuing.

2. **Set your Anthropic API key:**

   ```bash
   cp .env.example .env
   # Edit .env and paste your key — no quotes, no trailing spaces
   ```

3. **Launch all three services:**

   ```bash
   docker compose up -d --build
   ```

4. **Pull the local model** (one-time, ~9.6 GB download):

   ```bash
   ./scripts/bootstrap.sh
   ```

5. **Open the chat UI:** [http://localhost:3000](http://localhost:3000)

   Select **`local-router`** from the model dropdown and start chatting.

## Service endpoints

| Service           | URL                       |
|-------------------|---------------------------|
| Open WebUI (chat) | http://localhost:3000     |
| Router API        | http://localhost:8000     |
| Ollama API        | http://localhost:11434    |

## Watching the routing decisions

```bash
docker compose logs router -f
```

You'll see one log line per request showing the route and timing:

```
Route: local | Prompt: what is the capital of albania
Route: local | Completed in 0.63s

Route: cloud | Prompt: def fibonacci(n): write this with memoization...
Route: cloud | Completed in 1.84s
```

## Forcing a route

| Prefix    | Effect                          |
|-----------|---------------------------------|
| `/local`  | Force local (Ollama)            |
| `/cloud`  | Force cloud (Claude)            |
| _none_    | Auto-decide via classifier      |

Example: `/cloud what is 2+2?` will hit Claude even though it's a trivially simple question.

## Configuration

`config.yaml` controls the classifier and model choices:

```yaml
ollama:
  base_url: "http://ollama:11434"
  model: "gemma4:e4b"

claude:
  model: "claude-sonnet-4-20250514"

classifier:
  max_local_length: 500
  code_patterns:
    - "```"
    - "def "
    - "class "
    # ...
```

Anything longer than `max_local_length` characters or containing one of the `code_patterns` substrings is sent to Claude. Everything else stays local.

After editing `config.yaml`, rebuild the router:

```bash
docker compose up -d --build router
```

## After editing the API key

Restart isn't enough — Docker Compose caches env vars at container creation. Force a recreate:

```bash
docker compose up -d --force-recreate router
```

## Troubleshooting

- **502 Bad Gateway with `404 Not Found` from Ollama** — model isn't pulled. Run `./scripts/bootstrap.sh`.
- **502 with `authentication_error`** — your API key is wrong or stale. Check `.env`, then `docker compose up -d --force-recreate router`.
- **502 with `Your credit balance is too low`** — your Anthropic workspace has no credits, even if billing shows a limit. Check the workspace your API key belongs to.
- **GPU not used (slow responses)** — verify `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi` works. If not, fix the NVIDIA Container Toolkit setup.
- **Open WebUI shows no models** — wait a few seconds for it to call `/v1/models`, then refresh.

## Running tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Project structure

```
app/
├── main.py                   # FastAPI app, /v1/chat/completions
├── classifier.py             # Routing heuristics
├── config.py                 # YAML + env loader
├── models.py                 # Pydantic models
└── clients/
    ├── ollama_client.py      # Async Ollama proxy
    └── anthropic_client.py   # Claude client + format translation

tests/                        # pytest test suite
config.yaml                   # Runtime config
docker-compose.yml            # Three services: ollama, router, open-webui
scripts/bootstrap.sh          # Pulls the local model
```

## Known limitations (this is a spike)

- **No streaming** — responses arrive all at once, not token-by-token
- **Simple classifier** — no token counting, no semantic analysis, just length + literal patterns
- **No retries** — upstream errors bubble up as 502
- **No conversation truncation** — long histories may exceed model context
