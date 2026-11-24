"""Step 9: Record Caldera run steps as RL training experiences.

Each executed link produces one Transition that can later be added to a replay buffer
for offline RL fine-tuning (see red/caldera/retrain.py).

Reward heuristic (matches CAGE-4 Weights as closely as possible):
  -0.05  per step (matches Weights.step penalty)
  +1.00  when the ability succeeded (status == '0')
  -0.10  when the ability failed
  +0.20  per new unique fact discovered in the output of that link
  +100.0 once per batch, on its last mapped link, when objective_reached is True (matches Weights.impact)

The 'open batch' pattern (one record per link, closed with its own outcome):
  - record_batch() is called before submitting a batch: one open record per link, in batch order.
  - close_batch() is called once the results of the whole batch are known, with (status, new_facts) per link
    in the same order. Failed links are closed too, so no record is lost.
  - flush() seals records of a batch that never reached close_batch (for example after an exception).

All records are stored in memory and optionally saved to a .npz file at the end.
"""
import json
from dataclasses import dataclass
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


class CalderaExperienceRecorder:
    """Accumulates Transitions from a single agentic Caldera run.

    Usage inside run_agentic():
        rec = CalderaExperienceRecorder(max_steps, heads=5)
        # before submitting a batch:
        rec.record_batch(batch, tick)
        # after the results of the whole batch are known (one (status, new_facts) per link, same order):
        rec.close_batch(list(zip(statuses, facts_gained)), next_short, tick + 1, done, objective_reached)
        # after the loop:
        rec.flush()
        transitions = rec.transitions
    """

    def __init__(self, max_steps: int, heads: int = 5):
        self.max_steps = max_steps
        self.heads = heads
        self._mask: Tuple[int, ...] = tuple(1 for _ in range(heads))
        self.transitions: List[Transition] = []
        self._open: List[Optional[_Pending]] = []

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
    def record_batch(self, links, tick: int):
        """Call before submitting a batch: one open record per link, in batch order (unmapped links get None)."""
        self._open = []
        for link in links:
            feat = self._feature_for(link, tick)
            self._open.append(None if feat is None else _Pending(feature=feat, tick=tick))

    def close_batch(self, outcomes, next_links, next_tick: int, done: bool = False, objective_reached: bool = False):
        """Call after the results of the whole batch are known.

        outcomes          – (status, new_facts) per link, in the order given to record_batch
        next_links        – shortlist for the NEXT step (empty when the run ends here)
        next_tick         – tick value used on the next step
        done              – True when the run ends with this batch (terminal: no bootstrap)
        objective_reached – True only when this batch satisfied the objective. The bonus is added once, to the
                            last mapped link; if no link in the batch is mapped, the bonus is not recorded.
        """
        open_records, self._open = self._open, []
        if len(outcomes) != len(open_records):
            raise ValueError(f'close_batch: {len(outcomes)} outcomes for {len(open_records)} recorded links')
        records = [(p, status, new_facts) for p, (status, new_facts) in zip(open_records, outcomes) if p is not None]
        done = done or objective_reached
        for i, (p, status, new_facts) in enumerate(records):
            reward = STEP_COST + (SUCCESS_BONUS if status == OK_STATUS else FAIL_PENALTY)
            reward += FACT_BONUS * new_facts
            if objective_reached and i == len(records) - 1:
                reward += OBJECTIVE_BONUS
            self.transitions.append(Transition(
                feature=p.feature.copy(),
                reward=float(reward),
                next_features=self._next_features(next_links, next_tick),
                done=done,
                duration=1,
                mask=self._mask,
            ))

    def flush(self):
        """Seal records of a batch that never reached close_batch (for example after an exception).

        Terminal: no next actions, only the step cost and the fail penalty.
        """
        open_records, self._open = self._open, []
        for p in open_records:
            if p is None:
                continue
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
