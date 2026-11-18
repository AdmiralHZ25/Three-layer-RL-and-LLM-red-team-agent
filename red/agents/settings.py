"""Hyperparameters of the multi-head (bootstrapped) DDQN agent."""
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional

REFERENCE_ENV_TICKS = 154922  # budget reference of the DDQN config; kept here so the ddqn package is not needed


@dataclass
class MultiHeadConfig:
    heads: int = 5                      # K
    bootstrap_p: float = 0.5            # P(mask bit k = 1) per transition and head
    main_head: int = 0                  # head that chooses actions (and is the reference of the gate)
    gamma: float = 0.97                 # bootstrap uses gamma ** action_duration_ticks
    lr: float = 1e-3                    # Adam
    batch_size: int = 32
    replay_capacity: int = 20000
    warmup: int = 64                    # no update before len(replay) >= max(batch_size, warmup)
    target_update_interval: int = 100   # hard target sync every this many updates
    grad_clip_norm: float = 10.0
    seed: int = 42
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    train_every: int = 1                # one gradient step every `train_every` decisions
    # --- runner fields (mirror red.agents.settings.DDQNConfig) ----------------------------
    scenario_seed_offset: Optional[int] = None  # training scenario seed = offset + episode; None -> seed
    budget_env_ticks: Optional[int] = REFERENCE_ENV_TICKS  # stop starting episodes once reached
    max_episodes: Optional[int] = None  # optional hard cap; if budget_env_ticks is None it sets the schedule
    max_steps: int = 100                # episode horizon in simulator ticks
    eval_seed: int = 20000              # held-out scenario seeds eval_seed .. eval_seed+eval_episodes-1
    eval_episodes: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.08
    epsilon_fraction: float = 0.7       # linear decay over this fraction of training progress
    curriculum_fractions: List[float] = field(default_factory=lambda: [0.2, 0.3])
    checkpoint_interval: int = 100      # episodes
    log_train_events: bool = False      # per-decision JSONL during training (large)
    reward_version: str = 'v1'
    mask_last_withdraw: bool = False
    # --- optional Rainbow components (defaults = the original behaviour, bit for bit) ----------
    n_step: int = 1                     # multi-step return length (1 = ordinary one-step transitions)
    prioritized: bool = False           # proportional prioritized experience replay
    per_alpha: float = 0.6              # priority exponent
    per_beta_start: float = 0.4         # importance-sampling exponent at progress 0
    per_beta_end: float = 1.0           # ... annealed linearly to this at progress 1
    per_eps: float = 1e-6               # added to |TD error| before the exponent

    def validate(self):
        errors = []
        if self.heads < 1:
            errors.append('heads >= 1')
        if not 0 <= self.main_head < self.heads:
            errors.append('0 <= main_head < heads')
        if not 0.0 <= self.bootstrap_p <= 1.0:
            errors.append('0 <= bootstrap_p <= 1')
        if not 0 < self.gamma <= 1:
            errors.append('0 < gamma <= 1')
        if min(self.batch_size, self.replay_capacity, self.warmup, self.target_update_interval,
               self.train_every) < 1:
            errors.append('batch_size, replay_capacity, warmup, target_update_interval, train_every >= 1')
        if self.batch_size > self.replay_capacity:
            errors.append('batch_size <= replay_capacity')
        if self.lr <= 0 or self.grad_clip_norm <= 0:
            errors.append('lr > 0 and grad_clip_norm > 0')
        if self.max_steps < 4:
            errors.append('max_steps >= 4')
        if not self.budget_env_ticks and not self.max_episodes:
            errors.append('budget_env_ticks or max_episodes is required')
        if min(self.eval_episodes, self.checkpoint_interval) < 1:
            errors.append('eval_episodes, checkpoint_interval >= 1')
        if self.reward_version not in ('v1', 'v2'):
            errors.append("reward_version must be 'v1' or 'v2'")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or self.epsilon_fraction <= 0:
            errors.append('0 <= epsilon_end <= epsilon_start <= 1 and epsilon_fraction > 0')
        f = self.curriculum_fractions
        if len(f) != 2 or min(f) < 0 or sum(f) > 1:
            errors.append('curriculum_fractions = [a, b], a,b >= 0, a+b <= 1')
        if self.n_step < 1:
            errors.append('n_step >= 1')
        if self.per_alpha < 0 or self.per_eps <= 0:
            errors.append('per_alpha >= 0 and per_eps > 0')
        if not 0.0 <= self.per_beta_start <= self.per_beta_end <= 1.0:
            errors.append('0 <= per_beta_start <= per_beta_end <= 1')
        if errors:
            raise ValueError('Invalid MultiHeadConfig: ' + '; '.join(errors))
        return self

    @property
    def scenario_base(self):
        return self.seed if self.scenario_seed_offset is None else self.scenario_seed_offset

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
