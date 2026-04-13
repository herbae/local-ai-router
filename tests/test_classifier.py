from app.classifier import classify_prompt


class FakeSettings:
    max_local_length = 500
    code_patterns = ["```", "def ", "class ", "import ", "function "]
    force_local_prefixes = ["### Task:"]


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


def test_force_local_prefix_overrides_code_pattern():
    # Open WebUI's title/tag generation prompts contain code-like patterns
    # but should still route locally to avoid burning cloud credits.
    msg = "### Task:\nGenerate a concise 3-5 word title with class names and def stuff"
    assert classify_prompt(msg, settings) == "local"


def test_force_local_prefix_overrides_length():
    long_task = "### Task:\n" + "word " * 200
    assert classify_prompt(long_task, settings) == "local"
