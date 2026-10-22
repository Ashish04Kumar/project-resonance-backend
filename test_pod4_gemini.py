"""Pod 4's Gemini calls keep thinking low (fast, cheap) and cope with models that refuse the setting."""

from __future__ import annotations

import io
import json
import unittest
from dataclasses import replace
from unittest.mock import patch
from urllib.error import HTTPError

from app import pod4_providers
from app.config import settings
from app.pod4_providers import Message, _gemini, _gemini_generation_config


class GeminiThinkingTests(unittest.TestCase):
    def test_thinking_setting_per_model(self) -> None:
        with patch.object(pod4_providers, "settings", replace(settings, pod4_gemini_model="gemini-3.1-flash-lite", pod4_gemini_thinking="low")):
            self.assertEqual(_gemini_generation_config(0.7, 100)["thinkingConfig"], {"thinkingLevel": "low"})
        with patch.object(pod4_providers, "settings", replace(settings, pod4_gemini_model="gemini-2.5-flash-lite", pod4_gemini_thinking="low")):
            self.assertEqual(_gemini_generation_config(0.7, 100)["thinkingConfig"], {"thinkingBudget": 0})
        self.assertNotIn("thinkingConfig", _gemini_generation_config(0.7, 100, thinking=False))

    def test_model_that_refuses_thinking_is_asked_again_without_it(self) -> None:
        sent: list[dict] = []

        class Reply(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def fake_urlopen(request, timeout=None):
            body = json.loads(request.data)
            sent.append(body)
            if "thinkingConfig" in body["generationConfig"]:
                raise HTTPError(request.full_url, 400, "bad thinking", None, None)
            return Reply(json.dumps({"candidates": [{"content": {"parts": [{"text": "Hello!"}]}}]}).encode())

        fake = replace(settings, pod4_gemini_api_key="test-key", pod4_gemini_model="gemini-3.1-flash-lite")
        with patch.object(pod4_providers, "settings", fake), patch.object(pod4_providers, "urlopen", fake_urlopen):
            self.assertEqual(_gemini([Message("user", "hi")], 0.7, 50), "Hello!")
        self.assertEqual(len(sent), 2)
        self.assertNotIn("thinkingConfig", sent[1]["generationConfig"])


if __name__ == "__main__":
    unittest.main()
