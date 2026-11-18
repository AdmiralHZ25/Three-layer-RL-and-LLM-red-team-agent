"""Pick the LLM client from the project config (red.caldera.config).

Default (backend 'auto'/'ollama'/'endpoint') routes to the on-prem Ollama server on localhost using the
configured model -- on this PC that is ``qwen3-red`` (Qwen3-14B, Q2_K). 'stub' returns the offline
FakeLLM. Every backend implements the same complete(prompt)->LLMResult interface, so the decision pipeline
and its tests do not change. No network here -- clients are only constructed, not called.
"""
from .fake import FakeLLM
from .ollama import OllamaClient

_DEFAULT_OLLAMA = 'http://127.0.0.1:11434'


def make_llm(backend=None, model=None, base_url=None, timeout=None):
    """Return an LLM client. Arguments override the config; ``backend`` is one of
    auto | ollama | endpoint | stub | llama_cpp (the last not wired yet)."""
    from red import config as C
    backend = (backend or C.LLM_BACKEND or 'auto').lower()
    model = model or C.LLM_MODEL
    timeout = C.LLM_TIMEOUT if timeout is None else timeout
    if backend == 'stub':
        return FakeLLM(model=model)
    if backend in ('auto', 'ollama', 'endpoint'):
        url = (base_url or C.LLM_ENDPOINT or _DEFAULT_OLLAMA).split('/v1')[0].rstrip('/')
        return OllamaClient(model=model, base_url=url, timeout=timeout)
    if backend == 'llama_cpp':
        raise NotImplementedError(
            "backend 'llama_cpp' belum disambungkan; pakai Ollama (RED_LLM_BACKEND=ollama, qwen3-red) "
            "atau RED_LLM_BACKEND=stub untuk uji offline")
    raise ValueError(f'RED_LLM_BACKEND tak dikenal: {backend!r}')
