"""One interface for every language model used by the hybrid red agent: on-prem (Ollama) and API.

``complete(prompt) -> LLMResult`` is the whole contract. Token counts are ``None`` when the backend does
not report them (never 0, per docs/EVALUATION_PROTOCOL.md). No real backend is called here; the Ollama and
API clients (added later) and :class:`~red.llm.fake.FakeLLM` all implement this same interface, so
the decision pipeline and its tests never depend on a network.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMResult:
    text: str                                   # raw model output (empty when ok is False)
    ok: bool                                    # False on timeout / transport / backend error
    latency_s: float                            # end-to-end wall time of the call
    model: str                                  # model identifier, for the decision log
    prompt_tokens: Optional[int] = None         # None when the backend does not report it
    completion_tokens: Optional[int] = None
    error: Optional[str] = None                 # short reason when ok is False


class LLMClient:
    """Interface only. Implementations return an :class:`LLMResult` and never raise on a backend failure;
    they set ``ok=False`` and an ``error`` string instead, so the pipeline can fall through deterministically."""

    def complete(self, prompt: str) -> LLMResult:
        raise NotImplementedError
