"""Step 9: Record Caldera run steps as RL training experiences.

Each executed step produces one Transition that can later be added to a replay buffer
for offline RL fine-tuning (see red/caldera/retrain.py).

Reward heuristic (matches CAGE-4 Weights as closely as possible):
  -0.05  per step (matches Weights.step penalty)
  +1.00  when the ability succeeded (status == '0')
  -0.10  when the ability failed
  +0.20  per new unique fact discovered in the output
  +100.0 at the step that makes objective_reached True (matches Weights.impact)

The 'pending record' pattern:
  - record_step() is called at the START of a step (action chosen, feature built).
  - close_step() is called AFTER results arrive (reward known, next-state features built).
  - flush() seals the last open record when the run ends.

All records are stored in memory and optionally saved to a .npz file at the end.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from red.agents.transition import Transition, pack_transitions, unpack_transitions
from red.caldera.features import caldera_features
from red.caldera.mapping import cage_class

# Matches CAGE-4 reward weights used during simulation training.
STEP_COST = -0.05
SUCCESS_BONUS = 1.0
FAIL_PENALTY = -0.10
FACT_BONUS = 0.20
OBJECTIVE_BONUS = 100.0
OK_STATUS = '0'

# All bootstrap bits on: every head learns from real-world experience.
_ALL_HEADS_MASK = None   # filled lazily based on agent's K value


@dataclass
class _Pending:
    feature: np.ndarray
    tick: int
    n_facts_before: int


class CalderaExperienceRecorder:
    """Accumulates Transitions from a single agentic Caldera run.

    Usage inside run_agentic():
        rec = CalderaExperienceRecorder(max_steps, heads=5)
        # start of step loop:
        rec.record_step(link, tick, n_facts_so_far)
        # after executing and collecting results:
        rec.close_step(status, n_facts_after, next_short, tick+1, max_steps, objective_reached)
        # after the loop:
        rec.flush()
        transitions = rec.transitions
    """

    def __init__(self, max_steps: int, heads: int = 5):
        self.max_steps = max_steps
        self.heads = heads
        self._mask: Tuple[int, ...] = tuple(1 for _ in range(heads))
        self.transitions: List[Transition] = []
        self._pending: Optional[_Pending] = None

    # ------------------------------------------------------------------
    def _feature_for(self, link, tick: int) -> Optional[np.ndarray]:
        cls = cage_class(getattr(link, 'technique_id', None))
        if cls is None:
            return None
        return caldera_features(cls, tick, self.max_steps)

    def _next_features(self, next_links, next_tick: int) -> np.ndarray:
        feats = []
        for lnk in (next_links or []):
            f = self._feature_for(lnk, next_tick)
            if f is not None:
                feats.append(f)
        if not feats:
            return np.zeros((0, feats[0].shape[0] if feats else 22), dtype=np.float32)
        return np.stack(feats, axis=0)

    # ------------------------------------------------------------------
    def record_step(self, link, tick: int, n_facts: int):
        """Call at the START of a step, before submitting to Caldera.

        link        – the PotentialLink being executed
        tick        – 1-based step index
        n_facts     – number of unique facts known before this step
        """
        feat = self._feature_for(link, tick)
        if feat is None:
            self._pending = None   # unmapped technique: skip this step
            return
        self._pending = _Pending(feature=feat, tick=tick, n_facts_before=n_facts)

    def close_step(self, status: str, n_facts_after: int,
                   next_links, next_tick: int, objective_reached: bool = False):
        """Call AFTER results arrive to seal the pending record.

        status           – Caldera link status string ('0' = success)
        n_facts_after    – number of unique facts known after this step
        next_links       – shortlist for the NEXT step (used as next-state features)
        next_tick        – tick value that will be used on the next step
        objective_reached – True only on the step that satisfied the objective
        """
        if self._pending is None:
            return
        p = self._pending
        self._pending = None

        reward = STEP_COST
        if status == OK_STATUS:
            reward += SUCCESS_BONUS
        else:
            reward += FAIL_PENALTY
        new_facts = max(0, n_facts_after - p.n_facts_before)
        reward += FACT_BONUS * new_facts
        if objective_reached:
            reward += OBJECTIVE_BONUS

        done = objective_reached
        nxt = self._next_features(next_links, next_tick)
        self.transitions.append(Transition(
            feature=p.feature.copy(),
            reward=float(reward),
            next_features=nxt,
            done=done,
            duration=1,
            mask=self._mask,
        ))

    def flush(self):
        """Seal any open pending record when the run ends without a normal close_step."""
        if self._pending is None:
            return
        p = self._pending
        self._pending = None
        # Terminal: no next actions, no extra reward — just the step cost.
        self.transitions.append(Transition(
            feature=p.feature.copy(),
            reward=float(STEP_COST + FAIL_PENALTY),
            next_features=np.zeros((0, p.feature.shape[0]), dtype=np.float32),
            done=True,
            duration=1,
            mask=self._mask,
        ))

    # ------------------------------------------------------------------
    def save(self, path: Path):
        """Save all transitions to a .npz file for offline use."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        packed = pack_transitions(self.transitions)
        meta = {'n': len(self.transitions), 'heads': self.heads}
        np.savez_compressed(str(path), meta=json.dumps(meta), **packed)

    @classmethod
    def load(cls, path: Path) -> List[Transition]:
        """Load transitions from a .npz file previously saved by ``save()``."""
        data = np.load(str(path), allow_pickle=False)
        d = {k: data[k] for k in data.files if k != 'meta'}
        return unpack_transitions(d)
