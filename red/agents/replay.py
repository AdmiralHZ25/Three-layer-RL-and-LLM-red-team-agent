"""Replay buffers for the multi-head DDQN agent: uniform (default) and proportional prioritized."""
from .replay_nstep import NStepAccumulator
from .replay_prioritized import PrioritizedReplayBuffer
from .sum_tree import SumTree
from .transition import Transition, collate
from .replay_uniform import ReplayBuffer


def make_replay(cfg):
    """Uniform FIFO buffer unless ``cfg.prioritized`` (then a proportional PER buffer)."""
    if getattr(cfg, 'prioritized', False):
        return PrioritizedReplayBuffer(cfg.replay_capacity, cfg.seed, cfg.per_alpha, cfg.per_beta_start,
                                       cfg.per_beta_end, cfg.per_eps)
    return ReplayBuffer(cfg.replay_capacity, cfg.seed)


__all__ = ['Transition', 'collate', 'ReplayBuffer', 'PrioritizedReplayBuffer', 'SumTree',
           'NStepAccumulator', 'make_replay']
