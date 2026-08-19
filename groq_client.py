"""Small adapter exposing the response shape expected by the application."""

from dataclasses import dataclass
from typing import Any, Optional

from groq import Groq


MODEL_NAME = "openai/gpt-oss-120b"


@dataclass
class UsageMetadata:
    prompt_token_count: Optional[int] = None
    candidates_token_count: Optional[int] = None
    total_token_count: Optional[int] = None


@dataclass
class TextResponse:
    text: str
    usage_metadata: UsageMetadata


class _Models:
    def __init__(self, client: Groq):
        self._client = client

    def generate_content(self, model: str, contents: str, config: Any = None):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": contents}],
        }
        if isinstance(config, dict):
            if "temperature" in config:
                kwargs["temperature"] = config["temperature"]
            if "max_output_tokens" in config:
                kwargs["max_tokens"] = config["max_output_tokens"]

        response = self._client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        return TextResponse(
            text=response.choices[0].message.content or "",
            usage_metadata=UsageMetadata(
                prompt_token_count=getattr(usage, "prompt_tokens", None),
                candidates_token_count=getattr(usage, "completion_tokens", None),
                total_token_count=getattr(usage, "total_tokens", None),
            ),
        )


class GroqClient:
    def __init__(self, api_key: str):
        self._client = Groq(api_key=api_key)
        self.models = _Models(self._client)
