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
        self.force_local_prefixes = classifier.get("force_local_prefixes", [])


settings = Settings()
