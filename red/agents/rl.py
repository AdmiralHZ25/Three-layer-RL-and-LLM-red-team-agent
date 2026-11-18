"""Multi-head (bootstrapped) Double DQN learner over legal-candidate sets.

K heads share one torso. Each stored transition carries K Bernoulli(bootstrap_p) mask bits drawn from the
agent's own seeded RNG; head k is trained only on transitions whose bit k is 1. Actions come from the
MAIN head; the other heads exist for the Confidence Gate (see red.agents.confidence).

Per-head Double-DQN target for (s, a, r, s', done, Delta):
    a*_k = argmax_{a' in Legal(s')} Q_online,k(s', a')        (online head k SELECTS)
    y_k  = r + gamma**Delta * (1 - done) * Q_target,k(s', a*_k) (target head k EVALUATES)
No bootstrap when done or when Legal(s') is empty. Conventions (22-dim candidate features, padding masks,
candidate-set targets, ``add(feature, reward, next_features, done, ticks)``) follow red.rl.ddqn.
Loss = (1/K) * sum over heads with at least one selected sample of the masked mean Huber loss.

Optional Rainbow components, both OFF by default (then the code path and RNG streams are the original ones):
  * cfg.n_step > 1: decisions pass through red.agents.replay.NStepAccumulator; the stored transition is
    the n-step return (reward sum with gamma**elapsed ticks, summed duration, bootstrap from s_{t+n}) with
    the bootstrap mask drawn for its FIRST step. Call ``end_episode()`` after each episode.
  * cfg.prioritized: proportional PER. Each head's Huber term is weighted per sample by the importance-
    sampling weight, and priorities are updated with the mean |TD error| over the heads whose bootstrap
    bit is 1 (all heads if none). Beta is annealed with ``set_progress(p)``, p in [0, 1].
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import POLICY_LABEL, SCHEMA
from .settings import MultiHeadConfig
from .network import FEATURE_DIM, MultiHeadQNet
from red.agents.replay import NStepAccumulator, collate, make_replay


def _masked_q(net, next_feats, next_mask):
    """Q for every legal candidate and head: [B, W, K], -inf at padding. Net sees legal rows only."""
    heads = net.num_heads
    q = torch.full(next_mask.shape + (heads,), float('-inf'))
    if bool(next_mask.any()):
        q[next_mask] = net(next_feats[next_mask])
    return q


def multihead_double_dqn_targets(online, target, rewards, next_feats, next_mask, dones, durations, gamma):
    """Per-head Double-DQN targets [B, K]. Padding can never be selected."""
    q_on = _masked_q(online, next_feats, next_mask)
    a_star = q_on.argmax(dim=1)                                   # [B, K], online head k selects
    q_tg = _masked_q(target, next_feats, next_mask)
    boot = q_tg.gather(1, a_star.unsqueeze(1)).squeeze(1)         # [B, K], target head k evaluates
    valid = (next_mask.any(dim=1) & ~dones).unsqueeze(1)          # [B, 1]
    boot = torch.where(valid, boot, torch.zeros_like(boot))
    return rewards.unsqueeze(1) + (gamma ** durations).unsqueeze(1) * boot


def _select_index(scores, epsilon, rng, legal_mask=None):
    scores = torch.as_tensor(scores)
    legal = (torch.ones_like(scores, dtype=torch.bool) if legal_mask is None
             else torch.as_tensor(legal_mask, dtype=torch.bool))
    idx = torch.nonzero(legal).flatten().tolist()
    if not idx:
        return None
    if rng.random() < epsilon:
        return idx[rng.randrange(len(idx))]
    return int(scores.masked_fill(~legal, float('-inf')).argmax().item())


class MultiHeadDDQNAgent:
    def __init__(self, cfg: MultiHeadConfig):
        self.cfg = cfg.validate()
        with torch.random.fork_rng():
            torch.manual_seed(cfg.seed)
            self.online = MultiHeadQNet(FEATURE_DIM, tuple(cfg.hidden_sizes), cfg.heads)
            self.target = MultiHeadQNet(FEATURE_DIM, tuple(cfg.hidden_sizes), cfg.heads)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.replay = make_replay(cfg)
        self.nstep = NStepAccumulator(cfg.n_step, cfg.gamma) if cfg.n_step > 1 else None
        self.progress = 0.0                           # training progress in [0, 1] (PER beta annealing)
        self.rng = random.Random(cfg.seed + 1)        # epsilon exploration (default for act)
        self.mask_rng = random.Random(cfg.seed + 2)   # bootstrap masks only; global streams untouched
        self.updates = 0
        self.decisions = 0
        self.target_syncs = 0
        self.checkpoint_extra = {}

    # ---- acting ---------------------------------------------------------------------------
    @torch.no_grad()
    def q_matrix(self, features):
        """[N candidates, K heads] online Q-values (input to the Confidence Gate)."""
        x = torch.as_tensor(np.asarray(features, dtype=np.float32)).reshape(-1, FEATURE_DIM)
        return self.online(x)

    def act(self, features, epsilon=0.0, rng=None, legal_mask=None):
        """Index of the candidate chosen by the MAIN head (epsilon-random over legal ones)."""
        q = self.q_matrix(features)
        if q.shape[0] == 0:
            return None
        return _select_index(q[:, self.cfg.main_head], epsilon, rng or self.rng, legal_mask)

    # ---- learning -------------------------------------------------------------------------
    def draw_mask(self):
        return tuple(int(self.mask_rng.random() < self.cfg.bootstrap_p) for _ in range(self.cfg.heads))

    def set_progress(self, progress):
        """Training progress in [0, 1]; only used to anneal the PER importance-sampling exponent."""
        self.progress = min(max(float(progress), 0.0), 1.0)

    def end_episode(self):
        """Flush pending n-step windows (no-op with n_step == 1) so none leaks into the next episode."""
        if self.nstep is not None:
            for t in self.nstep.flush():
                self.replay.add(t.feature, t.reward, t.next_features, t.done, t.duration, t.mask)

    def add(self, feature, reward, next_features, done, ticks):
        mask = self.draw_mask()
        if self.nstep is None:
            self.replay.add(feature, reward, next_features, done, ticks, mask)
        else:
            for t in self.nstep.push(feature, reward, next_features, done, ticks, mask):
                self.replay.add(t.feature, t.reward, t.next_features, t.done, t.duration, t.mask)
        self.decisions += 1
        if len(self.replay) < max(self.cfg.batch_size, self.cfg.warmup):
            return
        if self.decisions % self.cfg.train_every == 0:
            self.train_step()

    def train_step(self, batch=None):
        """One gradient step. Returns the loss, or None if no head had a selected sample (no update)."""
        cfg = self.cfg
        indices = weights = None
        if batch is None:
            if cfg.prioritized:
                batch, indices, weights = self.replay.sample(cfg.batch_size, self.progress)
                weights = torch.as_tensor(weights, dtype=torch.float32)
            else:
                batch = self.replay.sample(cfg.batch_size)
        feats, rewards, nxt, mask, dones, durations, bmask = collate(batch)
        with torch.no_grad():
            y = multihead_double_dqn_targets(self.online, self.target, rewards, nxt, mask, dones,
                                             durations, cfg.gamma)
        q = self.online(feats)                                    # [B, K]
        loss = None
        for k in range(cfg.heads):
            sel = bmask[:, k]
            if not bool(sel.any()):
                continue    # no loss term -> no gradient -> Adam skips this head's parameters
            if weights is None:
                term = nn.functional.smooth_l1_loss(q[sel, k], y[sel, k])
            else:
                per_sample = nn.functional.smooth_l1_loss(q[sel, k], y[sel, k], reduction='none')
                term = (per_sample * weights[sel]).mean()
            loss = term if loss is None else loss + term
        if loss is None:
            if indices is not None:
                self._update_priorities(indices, q.detach(), y, bmask)
            return None
        loss = loss / cfg.heads
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss; training stopped')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip_norm)
        self.optimizer.step()
        if indices is not None:
            self._update_priorities(indices, q.detach(), y, bmask)
        self.updates += 1
        if self.updates % cfg.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target_syncs += 1
        return float(loss.detach())

    def _update_priorities(self, indices, q, y, bmask):
        """PER priority = mean |TD| over heads whose bootstrap bit is 1 (all heads if the sample has none)."""
        td = (y - q).abs()                                        # [B, K]
        bits = bmask.to(td.dtype)
        count = bits.sum(dim=1)
        masked_mean = (td * bits).sum(dim=1) / count.clamp(min=1.0)
        priority = torch.where(count > 0, masked_mean, td.mean(dim=1))
        self.replay.update_priorities(indices, priority.tolist())

    # ---- checkpoints (replay buffer is not saved) ----------------------------------------------
    def save(self, path, extra=None):
        path = Path(path)
        if path.exists():
            raise FileExistsError(f'Checkpoint already exists: {path}')
        payload = {'schema': SCHEMA, 'policy': POLICY_LABEL,
                   'state_dict': self.online.state_dict(),
                   'target_state_dict': self.target.state_dict(),
                   'optimizer': self.optimizer.state_dict(),
                   'feature_dim': FEATURE_DIM, 'heads': self.cfg.heads,
                   'config_json': json.dumps(self.cfg.to_dict()),
                   'updates': self.updates, 'decisions': self.decisions,
                   'target_syncs': self.target_syncs,
                   'extra_json': json.dumps(extra or {})}
        tmp = path.with_suffix('.tmp')
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return path

    @classmethod
    def load(cls, path, cfg=None):
        ck = torch.load(Path(path), map_location='cpu', weights_only=True)
        if ck.get('schema') != SCHEMA:
            raise ValueError(f"Not a {SCHEMA} checkpoint: {ck.get('schema')!r}")
        if ck.get('feature_dim') != FEATURE_DIM:
            raise ValueError('Checkpoint feature dimension differs from the current candidate features.')
        cfg = cfg or MultiHeadConfig.from_dict(json.loads(ck['config_json']))
        agent = cls(cfg)
        agent.online.load_state_dict(ck['state_dict'])
        agent.target.load_state_dict(ck['target_state_dict'])
        agent.optimizer.load_state_dict(ck['optimizer'])
        agent.updates, agent.decisions = int(ck['updates']), int(ck['decisions'])
        agent.target_syncs = int(ck['target_syncs'])
        agent.checkpoint_extra = json.loads(ck['extra_json'])
        return agent
