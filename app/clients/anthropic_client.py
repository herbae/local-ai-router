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
