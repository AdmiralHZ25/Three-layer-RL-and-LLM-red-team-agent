"""Regime-B snapshot builder + router-driven stage provider (Task 2, 18 Nov).

``make_snapshot`` copies ONLY allowlisted primitives (spec M2 section 3) from the policy-visible
``Knowledge`` and the candidate list -- never env, action objects, rewards, counters or seeds. Call it
right before ``inner.choose`` (the Chooser seam).

``router_stage_provider`` plugs the Stage Router into ``StagedHybridChooser`` as its ``stage_provider``:
it returns the key of exactly ONE specialist. UNKNOWN / AMBIGUOUS go to the general agent (``GENERAL``).
"""
from .stage_router import BSnapshot, Stage

GENERAL = 'general'   # multi-head DDQN without a stage mask; must be registered in StageSpecialists

SPECIALIST_FOR_STAGE = {
    Stage.RECON: 'recon',
    Stage.INITIAL_ACCESS: 'advance',
    Stage.PRIVILEGE_ESCALATION: 'advance',
    Stage.LATERAL_MOVEMENT: 'advance',
    Stage.IMPACT: 'impact',
}


def make_snapshot(knowledge, candidates, tick=None, horizon=None, cutoff_seq=None):
    """Immutable, deterministic copy (sorted, so equal states give equal snapshots)."""
    return BSnapshot(
        known=tuple(sorted(knowledge.known)),
        ips=tuple(sorted((str(k), str(v)) for k, v in knowledge.ips.items())),
        current_sessions={a: frozenset(s) for a, s in knowledge.current_sessions.items()},
        attempts={k: v for k, v in knowledge.attempts.items() if v},
        candidates=tuple((c[0], type(c[1]).__name__ if c[1] is not None else '', c[3]) for c in candidates),
        tick=tick, horizon=horizon, cutoff_seq=cutoff_seq)


def router_stage_provider(router, horizon=None):
    """``stage_provider(candidates, knowledge, tick) -> specialist key`` for ``StagedHybridChooser``.

    shortcut: one estimate per agent with an unresolved target scope; agents that disagree -> AMBIGUOUS
    (spec section 5: never the 'highest' stage). Upgrade when per-(agent, target) routing is needed.
    """
    def provider(candidates, knowledge=None, tick=None):
        snap = make_snapshot(knowledge, candidates, tick, horizon)
        agents = sorted({c[0] for c in candidates})
        stages = {router.predict(snap, a, ('UNRESOLVED', '')).stage_t for a in agents}
        stage = stages.pop() if len(stages) == 1 else Stage.AMBIGUOUS
        return SPECIALIST_FOR_STAGE.get(stage, GENERAL)
    return provider
