"""Settings for the red package. Environment variables override the defaults.

LLM settings (on-prem Ollama by default) and Caldera settings (Phase 2, lab only).
Secrets such as the Caldera API key come only from environment variables. They are never written to a file.
"""
import os


def _csv(name):
    return tuple(s.strip() for s in os.environ.get(name, '').split(',') if s.strip())


# LLM (on-prem Ollama by default)
LLM_BACKEND = os.environ.get('RED_LLM_BACKEND', 'auto')        # auto | ollama | endpoint | stub
LLM_ENDPOINT = os.environ.get('RED_LLM_ENDPOINT', '')
LLM_MODEL = os.environ.get('RED_LLM_MODEL', 'qwen3-red')        # Ollama model name (Qwen3-14B, Q2_K)
LLM_TIMEOUT = float(os.environ.get('RED_LLM_TIMEOUT', '180'))

# Caldera (Phase 2, lab only). Empty allow-lists mean nothing may run on any target.
CALDERA_URL = os.environ.get('CALDERA_URL', 'http://127.0.0.1:8888').rstrip('/')
CALDERA_API_KEY = os.environ.get('CALDERA_API_KEY', '')
CALDERA_PAW = os.environ.get('CALDERA_PAW', '').strip()
CALDERA_LAB_HOSTS = _csv('CALDERA_LAB_HOSTS')
CALDERA_ALLOWED_ABILITY_IDS = _csv('CALDERA_ALLOWED_ABILITY_IDS')
CALDERA_ADVERSARY_ID = os.environ.get('CALDERA_ADVERSARY_ID', '')   # empty = ad-hoc, the planner adds no actions
CALDERA_PLANNER_ID = os.environ.get('CALDERA_PLANNER_ID', '')
CALDERA_SOURCE_ID = os.environ.get('CALDERA_SOURCE_ID', '')
CALDERA_HTTP_TIMEOUT = float(os.environ.get('CALDERA_HTTP_TIMEOUT', '30'))
CALDERA_POLL_INTERVAL = float(os.environ.get('CALDERA_POLL_INTERVAL', '1.0'))
CALDERA_LINK_TIMEOUT = float(os.environ.get('CALDERA_LINK_TIMEOUT', '120'))
