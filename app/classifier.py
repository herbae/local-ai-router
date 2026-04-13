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

    # Force-local prefixes (e.g. Open WebUI background tasks)
    for prefix in getattr(settings, "force_local_prefixes", []):
        if message.startswith(prefix):
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
