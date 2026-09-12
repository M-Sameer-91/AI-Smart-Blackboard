"""Centralized, server-only Gemini integration for Blackboard AI features."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any

from app.ai.provider import AIProviderError

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiService:
    """Reusable Gemini boundary; API credentials never leave the server."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout_seconds: float | None = None, client: Any = None) -> None:
        if load_dotenv is not None:
            load_dotenv()
        self._api_key = (api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
        self.model = (model if model is not None else os.getenv("GEMINI_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
        try:
            configured_timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "25"))
        except ValueError:
            configured_timeout = 25.0
        self.timeout_seconds = timeout_seconds or min(max(configured_timeout, 3.0), 60.0)
        self._client = client

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def _client_or_error(self) -> Any:
        if not self._api_key:
            raise AIProviderError("Gemini is not configured. Set GEMINI_API_KEY on the server and restart it.")
        if self._client is not None:
            return self._client
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            return self._client
        except ImportError as exc:
            raise AIProviderError("Gemini support is not installed on the server. Install the project requirements.") from exc
        except Exception as exc:
            raise AIProviderError("Gemini client could not be initialized. Verify the server configuration.") from exc

    def _generate(self, prompt: str, image_bytes: bytes | None = None, mime_type: str = "image/png") -> str:
        if not prompt.strip():
            raise AIProviderError("Nothing was provided for Gemini to analyze.")
        client = self._client_or_error()
        contents: list[Any] = [prompt]
        if image_bytes:
            if len(image_bytes) > 12_000_000:
                raise AIProviderError("The selected canvas image is too large for AI analysis.")
            try:
                from google.genai import types
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            except ImportError as exc:
                raise AIProviderError("Gemini image support is not installed on the server. Install the project requirements.") from exc

        def call() -> Any:
            return client.models.generate_content(model=self.model, contents=contents)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                response = pool.submit(call).result(timeout=self.timeout_seconds)
        except FutureTimeout as exc:
            raise AIProviderError("Gemini timed out. Please try again shortly.") from exc
        except Exception as exc:
            message = str(exc).lower()
            if "429" in message or "resource_exhausted" in message:
                raise AIProviderError("Gemini rate limit reached. Please try again shortly.") from exc
            if "401" in message or "403" in message or "api key" in message:
                raise AIProviderError("Gemini authentication failed. Check the server-side GEMINI_API_KEY.") from exc
            raise AIProviderError("Gemini could not complete this request. Please try again.") from exc
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise AIProviderError("Gemini returned an empty or malformed response.")
        return text

    def explain(self, subject: str, content: str, image_bytes: bytes | None = None) -> str:
        prompt = ("You are a careful, encouraging educational assistant for an AI smart blackboard. "
                  f"Subject: {subject or 'General'}. Explain the selected board content clearly, "
                  "using short steps when useful. Do not claim the image says something you cannot see.\n"
                  f"Selected text/context: {content or '(image only)'}")
        return self._generate(prompt, image_bytes)

    def analyze_canvas(self, subject: str, image_bytes: bytes, local_result: dict[str, Any] | None = None) -> str:
        prompt = ("Interpret this selected smart-blackboard image for a teacher. Identify handwriting, equations, "
                  "diagrams, or educational drawings only when visible. Return a concise interpretation and state "
                  f"uncertainty. Subject: {subject or 'General'}. Local result: {local_result or {}}.")
        return self._generate(prompt, image_bytes)

    def lecture_assistance(self, subject: str, content: str, task: str = "summarize") -> str:
        prompt = ("You are preparing student-friendly lecture material from smart-blackboard content. "
                  f"Subject: {subject or 'General'}. Task: {task}. Produce concise Markdown with a title, key points, "
                  "and, where appropriate, an example or question.\nContent:\n" + content)
        return self._generate(prompt)
