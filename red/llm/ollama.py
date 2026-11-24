"""On-prem LLM via Ollama's HTTP API, implementing the ``complete(prompt) -> LLMResult`` interface.

stdlib ``urllib`` only (no new dependency). POST /api/generate with ``stream=False`` and a timeout; token
counts come from Ollama's ``prompt_eval_count`` / ``eval_count`` (left as None if absent). On any timeout,
transport, HTTP or decode error it returns ``ok=False`` with a short ``error`` and NEVER raises, so the
hybrid pipeline falls through deterministically. A ``transport`` callable can be injected so tests run with
no network. The model (D5: Qwen3-14B prototype, not yet frozen) and host are configurable.
"""
import http.client
import json
import time
import urllib.error
import urllib.request

from .client import LLMClient, LLMResult


class OllamaClient(LLMClient):
    def __init__(self, model='qwen3', base_url='http://127.0.0.1:11434', timeout=30.0,
                 temperature=0.0, options=None, transport=None):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.temperature = temperature
        self.options = dict(options or {})
        self._transport = transport          # callable(url, body:dict, timeout) -> dict; default: urllib

    def _http(self, url, body):
        if self._transport is not None:
            return self._transport(url, body, self.timeout)
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = r.read().decode()
        return json.loads(raw) if raw else {}

    def complete(self, prompt, num_predict=None):
        url = f'{self.base_url}/api/generate'
        # Default reply cap is 16 tokens (qwen3 thinking used the budget with no index, measured 2026-11-18). That default
        # is kept for the index-only pipeline. The Caldera agent loop passes its own cap per call, because a plan, a stop
        # reason or an evidence quote does not fit in 16 tokens (it was cut off in the 19 Nov runs).
        cap = 16 if num_predict is None else int(num_predict)
        # shortcut: thinking off, 8192 context for the deduplicated option list
        body = {'model': self.model, 'prompt': prompt, 'stream': False, 'think': False,
                'options': {'temperature': self.temperature, 'num_predict': cap, 'num_ctx': 8192, **self.options}}
        t0 = time.perf_counter()
        try:
            data = self._http(url, body)
        except urllib.error.HTTPError as e:
            return LLMResult('', False, time.perf_counter() - t0, self.model, error=f'http {e.code}')
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as e:
            return LLMResult('', False, time.perf_counter() - t0, self.model,
                             error=f'{type(e).__name__}: {e}')
        except (ValueError, TypeError) as e:
            return LLMResult('', False, time.perf_counter() - t0, self.model,
                             error=f'bad response: {e}')
        if not isinstance(data, dict) or data.get('error'):
            return LLMResult('', False, time.perf_counter() - t0, self.model,
                             error=f'bad response: {str(data)[:80]}')
        return LLMResult(text=data.get('response', ''), ok=True, latency_s=time.perf_counter() - t0,
                         model=data.get('model', self.model),
                         prompt_tokens=data.get('prompt_eval_count'),
                         completion_tokens=data.get('eval_count'))
