"""Deterministic offline LLM for tests and dry runs. No network, no randomness."""
from .client import LLMClient, LLMResult


class FakeLLM(LLMClient):
    """``responder(prompt) -> str`` maps a prompt to the model's raw text (default: always 'ACTION: 0').
    Set ``ok=False`` with an ``error`` to simulate a timeout or backend failure."""

    def __init__(self, responder=None, model='fake', ok=True, error=None, latency_s=0.0,
                 prompt_tokens=None, completion_tokens=None):
        self.responder = responder
        self.model = model
        self.ok = ok
        self.error = error
        self.latency_s = latency_s
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.num_predicts = []      # the num_predict of each call, in order (the reply itself is not capped here)

    def complete(self, prompt, num_predict=None):
        self.num_predicts.append(num_predict)
        if not self.ok:
            return LLMResult(text='', ok=False, latency_s=self.latency_s, model=self.model,
                             error=self.error or 'fake failure')
        text = self.responder(prompt) if self.responder is not None else 'ACTION: 0'
        return LLMResult(text=text, ok=True, latency_s=self.latency_s, model=self.model,
                         prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens)
