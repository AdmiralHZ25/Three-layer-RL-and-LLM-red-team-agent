"""Language-model layer of the hybrid red agent: one `complete(prompt) -> LLMResult` interface, a prompt
builder over the legal-candidate set, the Acceptance Gate, and a deterministic offline fake for tests."""
from .acceptance import AcceptanceGate, parse_action_index
from .client import LLMClient, LLMResult
from .factory import make_llm
from .fake import FakeLLM
from .ollama import OllamaClient
from .prompt import build_prompt, candidate_label

__all__ = ['LLMClient', 'LLMResult', 'FakeLLM', 'OllamaClient', 'make_llm', 'AcceptanceGate',
           'parse_action_index', 'build_prompt', 'candidate_label']
