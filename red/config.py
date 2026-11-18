"""LLM settings for the red package. Environment variables override the defaults.

Only the LLM part of the former Caldera config is kept. Caldera is not part of this package.
"""
import os

LLM_BACKEND = os.environ.get('RED_LLM_BACKEND', 'auto')       # auto | ollama | endpoint | stub
LLM_ENDPOINT = os.environ.get('RED_LLM_ENDPOINT', '')
LLM_MODEL = os.environ.get('RED_LLM_MODEL', 'qwen3-red')       # Ollama model name (Qwen3-14B, Q2_K)
LLM_TIMEOUT = float(os.environ.get('RED_LLM_TIMEOUT', '180'))
