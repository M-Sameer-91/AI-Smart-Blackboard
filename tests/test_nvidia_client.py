"""Offline tests for secure NVIDIA NIM client configuration and response handling."""

import json
import socket
import unittest
from urllib.error import HTTPError

from app.ai.nvidia_client import NVIDIAProvider
from app.ai.provider import AIProviderError


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class NVIDIAProviderTests(unittest.TestCase):
    def test_missing_credentials_are_unavailable_and_safe(self):
        provider = NVIDIAProvider(api_key="", model="")
        self.assertFalse(provider.is_available)
        with self.assertRaisesRegex(AIProviderError, "NVIDIA_API_KEY"):
            provider.explain("Recognized text: 2x + 5 = 15")

    def test_parses_openai_compatible_response_without_exposing_key(self):
        provider = NVIDIAProvider(
            api_key="test-secret", model="configured-model",
            opener=lambda *_args, **_kwargs: _Response({"choices": [{"message": {"content": "Solve for x by subtracting 5."}}]}),
        )
        self.assertTrue(provider.is_available)
        self.assertEqual(provider.explain("Recognized text: 2x + 5 = 15"), "Solve for x by subtracting 5.")

    def test_http_and_timeout_failures_are_friendly(self):
        def unauthorized(*_args, **_kwargs):
            raise HTTPError("https://example.invalid", 401, "Unauthorized", {}, None)
        with self.assertRaisesRegex(AIProviderError, "authentication failed"):
            NVIDIAProvider(api_key="test", model="model", opener=unauthorized).explain("context")

        def timeout(*_args, **_kwargs):
            raise socket.timeout()
        with self.assertRaisesRegex(AIProviderError, "timed out"):
            NVIDIAProvider(api_key="test", model="model", opener=timeout).explain("context")


if __name__ == "__main__":
    unittest.main()
