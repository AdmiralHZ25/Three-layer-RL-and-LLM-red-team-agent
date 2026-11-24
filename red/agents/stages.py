"""Stage-specialized RL layer (M5): one approved multi-head DDQN per stage, routed by a provided stage.

``StageSpecialists`` holds the three stage agents (recon/advance/impact) plus the unmasked ``general`` agent.
``SpecialistChooser`` is a Chooser seam adapter: a ``stage_provider`` gives the stage for the step, the matching specialist scores the candidates,
the per-stage mask forces off-stage actions out of contention, and the main head's argmax is returned.

The stage is supplied explicitly (by the caller's ``stage_provider``), NOT by the shadow-mode Stage Router
whose runtime integration needs separate authorization (M2 gate C). An unknown/None stage, or an empty
specialist set, routes to ``default_stage`` (``general``, the unmasked agent; it must be registered, else
ValueError -- never a silent fall to ``advance``). If the stage mask leaves no legal candidate, the step goes to
``general`` (no stage mask) instead of unmasking the specialist; ``general`` missing -> ValueError.
No RNG, no exploration.
"""
import numpy as np

from red.simulator.chooser import Choice
from red.router.snapshot import GENERAL

from .stage_mask import STAGES, stage_action_mask


class StageSpecialists:
    """``agents``: mapping stage -> MultiHeadDDQNAgent (the approved model)."""

    def __init__(self, agents):
        self.agents = dict(agents)

    def agent_for(self, stage):
        return self.agents.get(stage)

    def __contains__(self, stage):
        return stage in self.agents


def route_and_mask(specialists, stage, candidates, default_stage=GENERAL, apply_mask=True):
    """(routed stage, bool mask [N]). Unknown stage -> ``default_stage``. A stage mask with no legal candidate hands
    the step to ``general`` (all candidates legal), so the transition is stored by the general learner."""
    routed = stage if stage in specialists else default_stage
    mask = stage_action_mask(candidates, routed) if apply_mask else np.ones(len(candidates), dtype=bool)
    if len(candidates) and not mask.any():
        if GENERAL not in specialists:
            raise ValueError(f'stage mask for {routed!r} leaves no legal action and no {GENERAL!r} agent is registered')
        return GENERAL, np.ones(len(candidates), dtype=bool)
    return routed, mask


class SpecialistChooser:
    def __init__(self, specialists, stage_provider, default_stage=GENERAL, main_head=None):
        self.specialists = specialists
        self.stage_provider = stage_provider          # callable(candidates, knowledge, tick) -> stage|None
        self.default_stage = default_stage
        self.main_head = main_head

    def choose(self, candidates, knowledge=None, tick=None):
        stage = self.stage_provider(candidates, knowledge, tick)
        routed, mask = route_and_mask(self.specialists, stage, candidates, self.default_stage)
        agent = self.specialists.agent_for(routed)
        if agent is None:
            raise ValueError(f'no specialist for stage {routed!r}')
        mh = agent.cfg.main_head if self.main_head is None else self.main_head
        q = agent.q_matrix([c[2] for c in candidates])[:, mh].detach().cpu().numpy().astype(np.float64)
        idx = int(np.argmax(np.where(mask, q, -np.inf)))
        return Choice(index=idx, source='rl:' + routed,
                      meta={'stage': stage, 'routed_stage': routed, 'n_masked': int((~mask).sum()),
                            'main_head': mh})


class SpecialistEpsilonGreedyChooser:
    """Training selector. Same RNG order as red.simulator.chooser.EpsilonGreedyChooser (``rng.random()``, then
    ``rng.randrange`` only when exploring). Exploration is uniform over ALL candidates (Baru-1); the greedy
    branch scores with the routed specialist's main head, stage mask applied (``apply_mask=False`` = no mask).
    ``meta['routed_stage']`` tells the TransitionRouter which specialist owns the transition."""

    def __init__(self, specialists, stage_provider, eps, rng, default_stage=GENERAL, apply_mask=True,
                 main_head=None):
        self.specialists = specialists
        self.stage_provider = stage_provider
        self.eps = eps
        self.rng = rng
        self.default_stage = default_stage
        self.apply_mask = apply_mask
        self.main_head = main_head

    def choose(self, candidates, knowledge=None, tick=None):
        stage = self.stage_provider(candidates, knowledge, tick)
        routed, mask = route_and_mask(self.specialists, stage, candidates, self.default_stage, self.apply_mask)
        meta = {'stage': stage, 'routed_stage': routed}
        if self.rng.random() < self.eps:
            return Choice(index=self.rng.randrange(len(candidates)), source='random', meta=meta)
        agent = self.specialists.agent_for(routed)
        mh = agent.cfg.main_head if self.main_head is None else self.main_head
        q = agent.q_matrix([c[2] for c in candidates])[:, mh].detach().cpu().numpy().astype(np.float64)
        return Choice(index=int(np.argmax(np.where(mask, q, -np.inf))), source='rl:' + routed, meta=meta)


def max_q_stage_provider(specialists, main_head=None):
    """Stage provider with no stage prediction (decision 25 Nov 2026: 'Nilai Q tertinggi').

    Every specialist scores the candidates that its action types own (STAGE_ACTIONS; Sleep is shared) with its main
    head. The stage of the best-scoring candidate over all specialists is returned, and the chooser routes the step
    to that specialist. The stage comes from the action type, never from the situation: no predicate, no router.
    Returns None when no specialist can score any candidate, which routes the step to 'general'. Ties go to the first
    stage in STAGES order, so the result is deterministic."""
    def provider(candidates, knowledge=None, tick=None):
        best_stage, best_q = None, -np.inf
        for stage in STAGES:
            agent = specialists.agent_for(stage)
            if agent is None:
                continue
            mask = stage_action_mask(candidates, stage)
            if not mask.any():
                continue
            mh = agent.cfg.main_head if main_head is None else main_head
            q = agent.q_matrix([c[2] for c in candidates])[:, mh].detach().cpu().numpy().astype(np.float64)
            value = float(np.max(np.where(mask, q, -np.inf)))
            if value > best_q:
                best_stage, best_q = stage, value
        return best_stage
    return provider
