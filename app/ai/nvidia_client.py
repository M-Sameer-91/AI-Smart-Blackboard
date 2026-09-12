"""NVIDIA hosted NIM client; credentials are read only from environment variables."""

import json
import os
import socket
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.ai.provider import AIProvider, AIProviderError

try:
    from dotenv import load_dotenv
except ImportError:  # Keeps the application usable when only OS env vars are used.
    load_dotenv = None


DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NVIDIAProvider(AIProvider):
    """Small OpenAI-compatible client for NVIDIA's hosted NIM catalog API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 20.0,
                 opener: Callable = urlopen) -> None:
        self._api_key = (api_key if api_key is not None else os.getenv("NVIDIA_API_KEY", "")).strip()
        self._model = (model if model is not None else os.getenv("NVIDIA_MODEL", "")).strip()
        self._base_url = (base_url if base_url is not None else os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)).rstrip("/")
        self._timeout = timeout
        self._opener = opener

    @classmethod
    def from_environment(cls) -> "NVIDIAProvider":
        if load_dotenv is not None:
            load_dotenv()
        return cls()

    @property
    def is_available(self) -> bool:
        return bool(self._api_key and self._model)

    def explain(self, board_context: str) -> str:
        if not self._api_key:
            raise AIProviderError("NVIDIA API is unavailable: set NVIDIA_API_KEY in your environment.")
        if not self._model:
            raise AIProviderError("NVIDIA API is unavailable: set NVIDIA_MODEL to your selected catalog model.")
        if not board_context.strip():
            raise AIProviderError("Nothing on the board is available to explain yet.")

        payload = {
            "model": self._model,
            "temperature": 0.2,
            "max_tokens": 500,
            "messages": [
                {"role": "system", "content": "You are a concise, encouraging teacher. Explain the provided smart-blackboard context. Do not invent unseen content."},
                {"role": "user", "content": board_context},
            ],
        }
        request = Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise AIProviderError("NVIDIA API authentication failed. Check NVIDIA_API_KEY.") from exc
            if exc.code == 429:
                raise AIProviderError("NVIDIA API rate limit reached. Please try again shortly.") from exc
            raise AIProviderError(f"NVIDIA API request failed (HTTP {exc.code}).") from exc
        except (URLError, socket.timeout, TimeoutError) as exc:
            raise AIProviderError("NVIDIA API is unavailable or timed out. Check your network and try again.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AIProviderError("NVIDIA API returned an unreadable response.") from exc

        try:
            content = str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise AIProviderError("NVIDIA API returned an empty response.") from exc
        if not content:
            raise AIProviderError("NVIDIA API returned an empty response.")
        return content
