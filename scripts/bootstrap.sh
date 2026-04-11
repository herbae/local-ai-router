#!/bin/bash
set -e

MODEL="${1:-gemma4:e4b}"

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
