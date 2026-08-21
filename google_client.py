"""Small adapter exposing the response shape expected by the application."""

from dataclasses import dataclass
from typing import Any, Optional

from google import genai


MODEL_NAME = "gemini-3.6-flash"


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
    def __init__(self, client):
        self._client = client

    def generate_content(self, model: str, contents: str, config: Any = None):
        response = self._client.models.generate_content(
            model=model, contents=contents, config=config
        )
        usage = getattr(response, "usage_metadata", None)
        return TextResponse(
            text=response.text or "",
            usage_metadata=UsageMetadata(
                prompt_token_count=getattr(usage, "prompt_token_count", None),
                candidates_token_count=getattr(usage, "candidates_token_count", None),
                total_token_count=getattr(usage, "total_token_count", None),
            ),
        )


class GroqClient:
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)
        self.models = _Models(self._client)
