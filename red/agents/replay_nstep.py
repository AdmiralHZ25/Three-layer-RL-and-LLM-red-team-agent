"""n-step return accumulator for variable-duration decisions.

A window of the last <= n one-step transitions (s_t, a_t, r_t, s_{t+1}, done_t, d_t, mask_t) is folded into
    R      = sum_i gamma ** (d_0 + ... + d_{i-1}) * r_i          (i = 0 .. m-1, m <= n steps in the window)
    D      = d_0 + ... + d_{m-1}                                  (bootstrap discount gamma ** D, as in agent.py)
    s'     = next candidate set of the LAST step in the window
    done   = True iff the episode terminated inside the window
    mask   = bootstrap mask of the FIRST step (so the mask stream drawn by the agent is unchanged)
With n = 1 every pushed transition is returned unchanged. Near the end of an episode the remaining windows
are emitted truncated (m < n): on ``done`` they end in a terminal (no bootstrap); on ``flush()`` without
``done`` (episode cut off) they keep done=False and bootstrap from s' with gamma ** D.
"""
from collections import deque

import numpy as np

from red.agents.network import FEATURE_DIM

from .transition import Transition, pack_transitions, unpack_transitions


class NStepAccumulator:
    def __init__(self, n, gamma):
        if int(n) < 1:
            raise ValueError('n >= 1')
        self.n = int(n)
        self.gamma = float(gamma)
        self._window = deque()

    def __len__(self):
        return len(self._window)

    def push(self, feature, reward, next_features, done, duration, mask):
        """Add one decision; returns the list of n-step Transitions that are ready (oldest first)."""
        self._window.append(Transition(
            np.asarray(feature, dtype=np.float32).copy(), float(reward),
            np.asarray(next_features, dtype=np.float32).reshape(-1, FEATURE_DIM).copy(),
            bool(done), int(duration), tuple(int(b) for b in mask)))
        out = []
        if done:
            while self._window:
                out.append(self._fold(len(self._window)))
                self._window.popleft()
        elif len(self._window) >= self.n:
            out.append(self._fold(self.n))
            self._window.popleft()
        return out

    def flush(self):
        """Episode ended without a terminal flag: emit all pending truncated windows, then clear."""
        out = []
        while self._window:
            out.append(self._fold(len(self._window)))
            self._window.popleft()
        return out

    def _fold(self, steps):
        """Transition covering the first ``steps`` entries of the window."""
        items = list(self._window)[:steps]
        ret, elapsed = 0.0, 0
        for t in items:
            ret += self.gamma ** elapsed * t.reward
            elapsed += t.duration
        first, last = items[0], items[-1]
        return Transition(first.feature, ret, last.next_features, last.done, elapsed, first.mask)

    def state_dict(self):
        """Pending (not-yet-emitted) window, plus n and gamma. Round-trips without pickle."""
        return {'n': self.n, 'gamma': self.gamma, 'window': pack_transitions(self._window)}

    def load_state_dict(self, state):
        self.n = int(state['n'])
        self.gamma = float(state['gamma'])
        self._window = deque(unpack_transitions(state['window']))
