"""Proportional prioritized experience replay (Schaul et al. 2016, arXiv:1511.05952) over a sum tree.

Same ``add`` signature and FIFO overwrite as the uniform buffer. New transitions get the current maximum
priority. ``sample`` draws one item per equal-mass segment of the total priority (stratified) with the
buffer's own seeded ``random.Random`` and returns importance-sampling weights
    w_i = (N * P(i)) ** -beta,  P(i) = p_i / sum_j p_j,   normalised by the largest weight IN THE BATCH,
so every weight is in (0, 1]. Beta is annealed linearly from ``beta_start`` to ``beta_end`` with the
``progress`` in [0, 1] given by the caller. ``update_priorities`` sets p = (|td| + eps) ** alpha.
"""
import random

import numpy as np

from red.agents.network import FEATURE_DIM

from .sum_tree import SumTree
from .transition import Transition, coerce_py_rng_state, pack_transitions, unpack_transitions


class PrioritizedReplayBuffer:
    def __init__(self, capacity, seed, alpha=0.6, beta_start=0.4, beta_end=1.0, eps=1e-6):
        if alpha < 0 or eps <= 0:
            raise ValueError('alpha >= 0 and eps > 0')
        self.capacity = int(capacity)
        self.alpha, self.beta_start, self.beta_end, self.eps = alpha, beta_start, beta_end, eps
        self.rng = random.Random(seed)
        self.tree = SumTree(self.capacity)
        self.data = [None] * self.capacity
        self.size = 0
        self._next = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def add(self, feature, reward, next_features, done, duration, mask):
        nxt = np.asarray(next_features, dtype=np.float32).reshape(-1, FEATURE_DIM)
        self.data[self._next] = Transition(np.asarray(feature, dtype=np.float32).copy(), float(reward),
                                           nxt.copy(), bool(done), int(duration),
                                           tuple(int(b) for b in mask))
        self.tree.update(self._next, self.max_priority)
        self._next = (self._next + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def beta(self, progress=0.0):
        p = min(max(float(progress), 0.0), 1.0)
        return self.beta_start + (self.beta_end - self.beta_start) * p

    def sample(self, batch_size, progress=0.0):
        """Returns (batch, indices, is_weights float32 [B]); sampling is with replacement."""
        if self.size == 0:
            raise ValueError('cannot sample from an empty buffer')
        total = self.tree.total
        segment = total / batch_size
        indices = []
        for i in range(batch_size):
            s = self.rng.uniform(i * segment, (i + 1) * segment)
            idx = self.tree.find(s)
            indices.append(min(idx, self.size - 1))
        probs = np.array([self.tree.priority(i) for i in indices], dtype=np.float64) / total
        weights = (self.size * probs) ** (-self.beta(progress))
        weights = (weights / weights.max()).astype(np.float32)
        return [self.data[i] for i in indices], indices, weights

    def update_priorities(self, indices, td_abs):
        for idx, td in zip(indices, td_abs):
            p = (abs(float(td)) + self.eps) ** self.alpha
            self.tree.update(int(idx), p)
            self.max_priority = max(self.max_priority, p)

    def state_dict(self):
        """Ring buffer, sum tree, sampling RNG and hyper-parameters. Round-trips without pickle."""
        present = [t is not None for t in self.data]
        return {'kind': 'prioritized', 'capacity': self.capacity, 'size': self.size,
                'next': self._next, 'max_priority': self.max_priority,
                'alpha': self.alpha, 'beta_start': self.beta_start, 'beta_end': self.beta_end,
                'eps': self.eps, 'rng': self.rng.getstate(), 'tree': self.tree.state_dict(),
                'present': np.array(present, dtype=bool),
                'transitions': pack_transitions([t for t in self.data if t is not None])}

    def load_state_dict(self, state):
        if state.get('kind') != 'prioritized':
            raise ValueError(f"expected 'prioritized' replay state, got {state.get('kind')!r}")
        self.capacity = int(state['capacity'])
        self.size = int(state['size'])
        self._next = int(state['next'])
        self.max_priority = float(state['max_priority'])
        self.alpha = float(state['alpha'])
        self.beta_start = float(state['beta_start'])
        self.beta_end = float(state['beta_end'])
        self.eps = float(state['eps'])
        self.rng.setstate(coerce_py_rng_state(state['rng']))
        self.tree = SumTree(self.capacity)
        self.tree.load_state_dict(state['tree'])
        present = np.asarray(state['present'])
        restored = iter(unpack_transitions(state['transitions']))
        self.data = [None] * self.capacity
        for i in range(self.capacity):
            if bool(present[i]):
                self.data[i] = next(restored)
