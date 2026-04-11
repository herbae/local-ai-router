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
