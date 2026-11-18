"""Transition record and batch collation shared by the replay buffers of the multi-head DDQN agent."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    feature: np.ndarray        # (D,) features of the executed candidate
    reward: float
    next_features: np.ndarray  # (n, D) legal candidate set of s'; n may be 0
    done: bool                 # True -> no bootstrap
    duration: int              # ticks consumed by the action; discount gamma ** duration
    mask: Tuple[int, ...]      # (K,) bootstrap bits; head k learns from this transition iff mask[k] == 1


def collate(batch):
    """Stack a batch; pad next-candidate sets with a boolean legality mask.

    Returns feats [B,D], rewards [B], next_feats [B,W,D], next_mask [B,W], dones [B], durations [B],
    bootstrap_mask [B,K] (bool)."""
    feats = torch.tensor(np.stack([t.feature for t in batch]))
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
    dones = torch.tensor([t.done for t in batch], dtype=torch.bool)
    durations = torch.tensor([t.duration for t in batch], dtype=torch.float32)
    bmask = torch.tensor([list(t.mask) for t in batch], dtype=torch.bool)
    width = max(1, max(len(t.next_features) for t in batch))
    nxt = np.zeros((len(batch), width, feats.shape[1]), dtype=np.float32)
    mask = np.zeros((len(batch), width), dtype=bool)
    for i, t in enumerate(batch):
        n = len(t.next_features)
        if n:
            nxt[i, :n] = t.next_features
            mask[i, :n] = True
    return feats, rewards, torch.tensor(nxt), torch.tensor(mask), dones, durations, bmask


# ---- resume serialisation (no pickle: plain arrays + scalars) --------------------------------

def pack_transitions(seq):
    """Pack a sequence of Transitions into flat arrays. The next-candidate sets vary in length, so they
    are concatenated and recovered with a per-transition count. Inverse: ``unpack_transitions``."""
    items = list(seq)
    if not items:
        return {'feature': np.zeros((0, 0), np.float32), 'reward': np.zeros((0,), np.float64),
                'done': np.zeros((0,), bool), 'duration': np.zeros((0,), np.int64),
                'mask': np.zeros((0, 0), np.int64), 'next_count': np.zeros((0,), np.int64),
                'next_feat': np.zeros((0, 0), np.float32)}
    k = len(items[0].mask)
    feature = np.stack([np.asarray(t.feature, np.float32) for t in items])
    dim = feature.shape[1]
    reward = np.array([t.reward for t in items], np.float64)
    done = np.array([t.done for t in items], bool)
    duration = np.array([t.duration for t in items], np.int64)
    mask = np.array([list(t.mask) for t in items], np.int64).reshape(len(items), k)
    counts = np.array([len(t.next_features) for t in items], np.int64)
    pieces = [np.asarray(t.next_features, np.float32).reshape(-1, dim)
              for t in items if len(t.next_features)]
    next_feat = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, dim), np.float32)
    return {'feature': feature, 'reward': reward, 'done': done, 'duration': duration,
            'mask': mask, 'next_count': counts, 'next_feat': next_feat}


def unpack_transitions(d):
    """Rebuild the list of Transitions packed by ``pack_transitions`` (same order)."""
    reward = np.asarray(d['reward'])
    n = int(reward.shape[0])
    if n == 0:
        return []
    feature = np.asarray(d['feature'], np.float32)
    done = np.asarray(d['done'])
    duration = np.asarray(d['duration'])
    mask = np.asarray(d['mask'])
    counts = np.asarray(d['next_count'])
    next_feat = np.asarray(d['next_feat'], np.float32)
    dim = feature.shape[1]
    out, off = [], 0
    for i in range(n):
        c = int(counts[i])
        nxt = next_feat[off:off + c].reshape(c, dim).copy() if c else np.zeros((0, dim), np.float32)
        off += c
        out.append(Transition(feature[i].astype(np.float32).copy(), float(reward[i]), nxt,
                              bool(done[i]), int(duration[i]), tuple(int(b) for b in mask[i])))
    return out


def coerce_py_rng_state(state):
    """``random.Random.setstate`` wants ``(version:int, tuple[int], gauss)``. Accept a list/tuple that has
    round-tripped through JSON or npz (where the inner tuple became a list)."""
    version, internal, gauss = state
    return (int(version), tuple(int(x) for x in internal), None if gauss is None else float(gauss))
