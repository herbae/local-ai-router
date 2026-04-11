I want to create an "AI Router" in Python that runs locally and routes prompts between a local model (Ollama) and the Claude API for complex tasks.

My machine: Ubuntu 24.04, i7-11800H, 64GB RAM, RTX 3060 Laptop 6GB VRAM. 
Docker and Docker Compose are already installed.

## Architecture
- FastAPI server exposing an OpenAI-compatible API (POST /v1/chat/completions)
- Local model: Gemma 4 E4B via Ollama (localhost:11434)
- Cloud model: Claude Sonnet via Anthropic API
- A classifier that analyzes the prompt and decides whether to route locally or to the cloud

## Routing criteria (configurable via YAML)
- Prompt size (tokens)
- Presence of code or complex coding requests
- Multi-step reasoning / long analysis requests
- Translation, simple summaries, factual questions → local
- User can force cloud with a "/cloud" prefix or local with "/local"

## Technical requirements
- Python 3.12+, FastAPI, httpx, tiktoken
- Config via .env (ANTHROPIC_API_KEY) and config.yaml (thresholds, models)
- Clear logging showing which route was chosen and why
- Streaming support (SSE) for both local and cloud
- Docker Compose with the service + Ollama + Open WebUI all together

## Deliverables
- Router code
- docker-compose.yml with all 3 services (router, ollama, open-webui)
- README with setup instructions
- Bootstrap script that pulls the Gemma 4 E4B model in Ollama

Start by researching the Ollama API and the Anthropic API to ensure compatibility, then make a plan before coding.