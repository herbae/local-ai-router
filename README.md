# Local AI Router

A Python service that runs locally and routes prompts between a local model (Ollama) and the Claude API, exposing an OpenAI-compatible API.

## How it works

The router sits behind a standard `POST /v1/chat/completions` endpoint. A built-in classifier analyzes each prompt and decides whether to handle it locally or send it to the cloud:

- **Local (Gemma 4 E4B via Ollama)** — translation, simple summaries, factual questions
- **Cloud (Claude Sonnet via Anthropic API)** — code generation, multi-step reasoning, long analysis

Routing thresholds are configurable via `config.yaml`. You can also force a route by prefixing your prompt with `/cloud` or `/local`.

## Stack

- Python 3.12+, FastAPI, httpx, tiktoken
- Ollama for local inference
- Open WebUI as the chat frontend
- Docker Compose to run everything together

## Quick start

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

docker compose up -d
```

This starts three services:

| Service | Port |
|---------|------|
| Router (FastAPI) | 8000 |
| Ollama | 11434 |
| Open WebUI | 3000 |

Open WebUI connects to the router, which handles model selection automatically.

## Configuration

Edit `config.yaml` to adjust routing thresholds, token limits, and model names. See the comments in the file for details.

## Requirements

- Docker and Docker Compose
- An Anthropic API key (for cloud routing)
- ~6 GB VRAM for the local model
