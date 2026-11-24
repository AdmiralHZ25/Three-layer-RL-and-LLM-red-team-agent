"""Offline tests: the menu's Ollama check reads RED_LLM_ENDPOINT and RED_LLM_MODEL (red.config). No network.

Run from the RED_AGENT folder:  python -m unittest tests.test_menu_ollama -v
"""
import contextlib
import io
import json
import unittest
from unittest import mock

from red import config
import run_menu


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode('utf-8')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


class OllamaCheck(unittest.TestCase):
    def test_checks_the_configured_endpoint_and_model(self):
        seen = []

        def fake_urlopen(url, timeout=None):
            seen.append(url)
            return FakeResponse({'models': [{'name': 'qwen3-red:latest'}]})

        with mock.patch.multiple(config, LLM_ENDPOINT='http://192.168.56.1:11434', LLM_MODEL='qwen3-red'), \
                mock.patch('urllib.request.urlopen', side_effect=fake_urlopen), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(run_menu.check_ollama())
        self.assertEqual(seen, ['http://192.168.56.1:11434/api/tags'])

    def test_v1_suffix_is_removed_like_the_llm_client_does(self):
        with mock.patch.object(config, 'LLM_ENDPOINT', 'http://192.168.56.1:11434/v1'):
            self.assertEqual(run_menu.ollama_endpoint(), 'http://192.168.56.1:11434')

    def test_default_is_this_pc_when_no_endpoint_is_set(self):
        with mock.patch.object(config, 'LLM_ENDPOINT', ''):
            self.assertEqual(run_menu.ollama_endpoint(), 'http://127.0.0.1:11434')

    def test_model_missing_is_reported_as_false(self):
        with mock.patch.multiple(config, LLM_ENDPOINT='http://192.168.56.1:11434', LLM_MODEL='qwen3-red'), \
                mock.patch('urllib.request.urlopen', return_value=FakeResponse({'models': [{'name': 'llama3:8b'}]})), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run_menu.check_ollama())

    def test_unreachable_is_reported_as_false(self):
        with mock.patch.multiple(config, LLM_ENDPOINT='http://192.168.56.1:11434'), \
                mock.patch('urllib.request.urlopen', side_effect=OSError('refused')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run_menu.check_ollama())


if __name__ == '__main__':
    unittest.main()
