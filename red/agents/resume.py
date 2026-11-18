"""Resume support for multi-head DDQN training.

A weights checkpoint (``episode_NNNNNN.pt``) is saved with ``weights_only=True`` and does NOT hold the
replay buffer or RNG streams. This module writes a companion ``episode_NNNNNN.state`` next to it that does,
so an interrupted run can continue bit-for-bit: the replay buffer, the agent's own ``rng``/``mask_rng`` and
training ``progress``, and the global python/numpy/torch RNG.

The companion is a single npz file with **no pickle** (``np.load(..., allow_pickle=False)``): a JSON
manifest describes the structure and scalars; every ndarray is stored as its own npz entry and referenced by
key. Tuples are tagged so ``random``/``numpy`` RNG states round-trip exactly.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from red.agents.transition import coerce_py_rng_state

RESUME_SCHEMA = 'ddqn-multihead-resume-v1'


# ---- generic no-pickle codec (nested dicts/lists/tuples/scalars/ndarrays) ---------------------

def _encode(obj, arrays):
    if isinstance(obj, np.ndarray):
        key = f'a{len(arrays)}'
        arrays[key] = obj
        return {'__arr__': key}
    if isinstance(obj, tuple):
        return {'__tuple__': [_encode(v, arrays) for v in obj]}
    if isinstance(obj, list):
        return [_encode(v, arrays) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _encode(v, arrays) for k, v in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj                                   # int, float, bool, str, None


def _decode(obj, arrays):
    if isinstance(obj, dict):
        if '__arr__' in obj:
            return arrays[obj['__arr__']]
        if '__tuple__' in obj:
            return tuple(_decode(v, arrays) for v in obj['__tuple__'])
        return {k: _decode(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v, arrays) for v in obj]
    return obj


def save_npz(path, obj):
    """Write a nested structure to one npz file (no pickle). Atomic via a temp file + os.replace."""
    path = Path(path)
    arrays = {}
    manifest = _encode(obj, arrays)
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        with open(tmp, 'wb') as f:
            np.savez(f, __meta__=np.array(json.dumps(manifest)), **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def load_npz(path):
    with np.load(Path(path), allow_pickle=False) as z:
        manifest = json.loads(str(z['__meta__'].item()))
        arrays = {k: z[k] for k in z.files if k != '__meta__'}
    return _decode(manifest, arrays)


# ---- agent runtime state ----------------------------------------------------------------------

def collect_runtime(agent):
    """Everything needed to continue training that is not in the weights checkpoint."""
    return {'schema': RESUME_SCHEMA,
            'config_json': json.dumps(agent.cfg.to_dict()),
            'progress': float(agent.progress),
            'agent_rng': agent.rng.getstate(),
            'mask_rng': agent.mask_rng.getstate(),
            'replay': agent.replay.state_dict(),
            'nstep': (agent.nstep.state_dict() if agent.nstep is not None else None),
            'py_rng': random.getstate(),
            'np_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state().numpy().copy()}


def _np_state(s):
    name, keys, pos, has_gauss, cached = s
    return (str(name), np.asarray(keys, dtype=np.uint32), int(pos), int(has_gauss), float(cached))


def apply_runtime(agent, state):
    if state.get('schema') != RESUME_SCHEMA:
        raise ValueError(f"not a {RESUME_SCHEMA} state: {state.get('schema')!r}")
    agent.progress = float(state['progress'])
    agent.rng.setstate(coerce_py_rng_state(state['agent_rng']))
    agent.mask_rng.setstate(coerce_py_rng_state(state['mask_rng']))
    agent.replay.load_state_dict(state['replay'])
    if agent.nstep is not None and state.get('nstep') is not None:
        agent.nstep.load_state_dict(state['nstep'])
    random.setstate(coerce_py_rng_state(state['py_rng']))
    np.random.set_state(_np_state(state['np_rng']))
    torch.set_rng_state(torch.tensor(np.asarray(state['torch_rng'], dtype=np.uint8), dtype=torch.uint8))
    return agent


def save_runtime(agent, path, extra=None):
    """Write the companion ``.state`` file. ``path`` must not exist."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'runtime state already exists: {path}')
    state = collect_runtime(agent)
    state['extra_json'] = json.dumps(extra or {})
    return save_npz(path, state)


def load_runtime(agent, path):
    """Restore the companion ``.state`` written by :func:`save_runtime` into ``agent`` (in place)."""
    return apply_runtime(agent, load_npz(path))
