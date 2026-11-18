"""Array-based sum tree: O(log n) priority update and prefix-sum lookup (for proportional PER)."""
import numpy as np


class SumTree:
    def __init__(self, capacity):
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError('capacity >= 1')
        self.capacity = capacity
        size = 1
        while size < capacity:
            size *= 2
        self._size = size                       # leaves live at [size, 2 * size); unused leaves stay 0
        self._tree = [0.0] * (2 * size)

    @property
    def total(self):
        return self._tree[1]

    def priority(self, idx):
        self._check(idx)
        return self._tree[self._size + idx]

    def update(self, idx, priority):
        self._check(idx)
        if not priority >= 0.0:                 # also rejects NaN
            raise ValueError('priority must be >= 0')
        i = self._size + idx
        self._tree[i] = float(priority)
        i //= 2
        while i >= 1:
            self._tree[i] = self._tree[2 * i] + self._tree[2 * i + 1]
            i //= 2

    def find(self, prefix_sum):
        """Leaf index whose cumulative-priority interval contains ``prefix_sum`` (clamped to [0, total))."""
        if self.total <= 0.0:
            raise ValueError('sum tree is empty (total priority 0)')
        s = min(max(float(prefix_sum), 0.0), self.total)
        i = 1
        while i < self._size:
            left = 2 * i
            if s < self._tree[left] or self._tree[left + 1] <= 0.0:
                i = left
            else:
                s -= self._tree[left]
                i = left + 1
        return i - self._size

    def _check(self, idx):
        if not 0 <= idx < self.capacity:
            raise IndexError(f'index {idx} outside [0, {self.capacity})')

    def state_dict(self):
        return {'capacity': self.capacity, 'tree': np.asarray(self._tree, dtype=np.float64)}

    def load_state_dict(self, state):
        if int(state['capacity']) != self.capacity:
            raise ValueError('sum tree capacity mismatch')
        tree = [float(x) for x in state['tree']]
        if len(tree) != len(self._tree):
            raise ValueError('sum tree size mismatch')
        self._tree = tree
