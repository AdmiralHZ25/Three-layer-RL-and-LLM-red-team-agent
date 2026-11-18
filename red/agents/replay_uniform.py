"""Uniform FIFO replay with a per-transition bootstrap mask (one Bernoulli bit per head)."""
import random
from collections import deque

import numpy as np

from red.agents.network import FEATURE_DIM

from .transition import Transition, coerce_py_rng_state, pack_transitions, unpack_transitions


class ReplayBuffer:
    def __init__(self, capacity, seed):
        self.buffer = deque(maxlen=int(capacity))
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.buffer)

    def add(self, feature, reward, next_features, done, duration, mask):
        nxt = np.asarray(next_features, dtype=np.float32).reshape(-1, FEATURE_DIM)
        self.buffer.append(Transition(np.asarray(feature, dtype=np.float32).copy(), float(reward),
                                      nxt.copy(), bool(done), int(duration),
                                      tuple(int(b) for b in mask)))

    def sample(self, batch_size):
        return self.rng.sample(self.buffer, batch_size)

    def state_dict(self):
        """Buffer contents and sampling RNG (not the weights). Round-trips without pickle."""
        return {'kind': 'uniform', 'capacity': int(self.buffer.maxlen),
                'rng': self.rng.getstate(), 'transitions': pack_transitions(self.buffer)}

    def load_state_dict(self, state):
        if state.get('kind') != 'uniform':
            raise ValueError(f"expected 'uniform' replay state, got {state.get('kind')!r}")
        self.buffer = deque(unpack_transitions(state['transitions']), maxlen=int(state['capacity']))
        self.rng.setstate(coerce_py_rng_state(state['rng']))
