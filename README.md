# Local AI Router

A FastAPI service that exposes an OpenAI-compatible API and routes prompts between a local model (Ollama + Gemma 4 E4B) and the Claude API based on prompt complexity. Designed to run on a developer laptop with a modest GPU.

## The interesting bit: a two-stage classifier

Most routers stop at regex heuristics. This one doesn't. The problem with heuristics is that they completely miss semantic complexity — *"explain the Taylor series of cos(x) step by step"* has no code, no `def`, no `import`, is under 500 characters — it looks trivial to regex, but it's firmly in cloud territory.

So the router runs **two classifier stages**:

```
                      ┌────────────────────────────────────┐
     user prompt ──▶  │  1. Heuristics (fast, ~0ms)        │
                      │     • /cloud, /local overrides     │
                      │     • force_local_prefixes         │
                      │     • code patterns                │
                      │     • length > threshold           │
                      └──────────────┬─────────────────────┘
                                     │
                    ┌────────────────┴─────────────────┐
                    ▼                                  ▼
          says "cloud" → Claude             says "local" → stage 2
                                                       │
                      ┌────────────────────────────────┴───┐
                      │  2. LLM classifier (~300-500ms)    │
                      │                                    │
                      │  Asks Gemma itself:                │
                      │  "LOCAL or CLOUD? one word."       │
                      │                                    │
                      │  • max_tokens=5, temperature=0     │
                      │  • can only escalate local → cloud │
                      │  • fails safe to local on error    │
                      └──────────────┬─────────────────────┘
                                     │
                    ┌────────────────┴─────────────────┐
                    ▼                                  ▼
                 "cloud" → Claude              "local" → Gemma
```

Stage 1 catches the obvious stuff cheaply. Stage 2 catches semantic complexity — the tax is ~400ms of local compute (free, GPU-accelerated), and it can **only promote** a prompt from local to cloud, never the reverse. If anything goes wrong (timeout, parse error, Ollama down), it falls back to local so the router keeps working.

Stage 2 is feature-flagged in `config.yaml` (`llm_classifier.enabled`) so you can run pure heuristics if you want to benchmark or save latency.

## Routes at a glance

- **Local (Gemma 4 E4B via Ollama)** — short prompts, translation, simple Q&A, definitions
- **Cloud (Claude Sonnet via Anthropic API)** — code, complex reasoning, specialized expertise, long analysis

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

When the LLM classifier escalates a prompt that heuristics would have kept local:

```
LLM classifier raw output: 'CLOUD' -> first_word='CLOUD'
LLM classifier escalated local -> cloud
Route: cloud | Prompt: explain the Taylor series of cos(x) step by step
Route: cloud | Completed in 5.21s
```

## Forcing a route

| Prefix    | Effect                          |
|-----------|---------------------------------|
| `/local`  | Force local (Ollama)            |
| `/cloud`  | Force cloud (Claude)            |
| _none_    | Auto-decide via classifier      |

Example: `/cloud what is 2+2?` will hit Claude even though it's a trivially simple question.

## Configuration

`config.yaml` controls both classifier stages and the model choices:

```yaml
ollama:
  base_url: "http://ollama:11434"
  model: "gemma4:e4b"

claude:
  model: "claude-sonnet-4-20250514"

# Stage 2: ask Gemma itself to classify prompts that pass heuristics
llm_classifier:
  enabled: true          # flip to false for pure heuristic mode
  timeout: 10.0          # fall back to local if Gemma doesn't respond in time

# Stage 1: cheap regex heuristics
classifier:
  max_local_length: 500  # longer than this → cloud
  force_local_prefixes:  # always local, even if code-like (e.g. Open WebUI tasks)
    - "### Task:"
  code_patterns:         # substring match → cloud
    - "```"
    - "def "
    - "class "
    # ...
```

**Tuning ideas:**

- Lower `max_local_length` to route more to cloud; raise it to trust local more
- Add domain-specific `force_local_prefixes` (e.g. chat UI wrapper prompts)
- Disable `llm_classifier.enabled` to measure pure heuristic accuracy
- Edit the classifier system prompt in `app/llm_classifier.py` to tune Gemma's judgment

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
- **No retries** — upstream errors bubble up as 502
- **No conversation truncation** — long histories may exceed model context
- **LLM classifier judgment is only as good as Gemma's self-awareness** — small models can be over-confident about what they can answer well. The system prompt in `app/llm_classifier.py` is a reasonable starting point, not a final tuning.
