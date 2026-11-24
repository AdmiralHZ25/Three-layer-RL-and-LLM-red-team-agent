"""Limits for one agentic Caldera run (menu g).

The objective is written in words by a person before each run. The limits are set once in red/config.py, not for each
run. They are a ceiling and a safety stop. They do not decide what the agent does. Nothing here contacts Caldera.
"""

MAX_STEPS_LIMIT = 20   # the same limit as scope.MAX_STEPS_LIMIT


def check_run_settings(max_steps, max_seconds, max_failures):
    """Raise ValueError when a run limit cannot be used. The menu calls this before it creates an operation."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_STEPS_LIMIT:
        raise ValueError(f'CALDERA_RUN_MAX_STEPS harus angka 1 sampai {MAX_STEPS_LIMIT}')
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or max_seconds <= 0:
        raise ValueError('CALDERA_RUN_MAX_SECONDS harus lebih dari 0')
    if isinstance(max_failures, bool) or not isinstance(max_failures, int) or max_failures < 1:
        raise ValueError('CALDERA_MAX_CONSECUTIVE_FAILURES harus angka 1 atau lebih')
