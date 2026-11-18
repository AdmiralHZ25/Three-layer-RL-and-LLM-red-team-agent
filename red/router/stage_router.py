"""Stage Router v1 -- offline shadow-mode estimator.

Follows docs/experiments/M2_STAGE_ROUTER_SPECIFICATION.md (PROVISIONAL; CONDITIONAL GO for a standalone
offline prototype with fakes only). The router reads an immutable, allowlist-only Regime-B snapshot and
returns a StageEstimate for logging. Contract, by design:

  * It NEVER selects or changes a DQN action, loads a policy, or reads reward / counters / Green events /
    simulator state / seeds (spec sections 4, 8). Only the allowlist snapshot fields are read.
  * It draws NO randomness (no simulator/policy RNG, no python/numpy/torch global RNG) and never mutates
    its input -- same snapshot + config give the same StageEstimate (spec section 8).
  * Deterministic abstention v1 (spec 7.3): with the current Regime-B fields NO single-stage predicate is
    proven sufficient, so the safe default is UNKNOWN. Positive stage predicates are injectable test
    doubles ONLY, for exercising the five-label contract -- never a claim that B identifies stages.

Runtime integration (feeding a policy, capturing live snapshots) is a SEPARATE authorization (spec gate C)
and is intentionally NOT done here.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

SCHEMA_VERSION = 'B-v1'


class Stage(str, Enum):
    RECON = 'RECON'
    INITIAL_ACCESS = 'INITIAL_ACCESS'
    PRIVILEGE_ESCALATION = 'PRIVILEGE_ESCALATION'
    LATERAL_MOVEMENT = 'LATERAL_MOVEMENT'
    IMPACT = 'IMPACT'
    UNKNOWN = 'UNKNOWN'
    AMBIGUOUS = 'AMBIGUOUS'


# The five single stages a positive predicate could support (spec section 2); UNKNOWN/AMBIGUOUS are outcomes.
SINGLE_STAGES = (Stage.RECON, Stage.INITIAL_ACCESS, Stage.PRIVILEGE_ESCALATION,
                 Stage.LATERAL_MOVEMENT, Stage.IMPACT)


@dataclass(frozen=True)
class BSnapshot:
    """Immutable, allowlist-only Regime-B snapshot (spec section 3). Copies of primitive values only --
    never the env, action objects, controller, a mutable Knowledge, model, reward, counters or seed.
    ``current_sessions`` maps an agent id to the set of hosts seen on its last pre-cutoff update."""
    known: tuple = ()
    ips: tuple = ()
    current_sessions: dict = field(default_factory=dict)
    attempts: dict = field(default_factory=dict)
    candidates: tuple = ()                       # (agent, action_name, target) summaries only
    tick: Optional[int] = None
    horizon: Optional[int] = None
    cutoff_seq: Optional[int] = None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class StageEstimate:
    """Spec section 6. ``stage_confidence_t`` and ``recommended_policy_family_t`` stay None in M2 shadow."""
    stage_t: Stage
    access_state_t: str                          # OBSERVED_PRESENT | ABSENT_CONFIRMED | UNKNOWN
    uncertainty_reason_t: tuple = ()
    possible_stages: tuple = ()                  # the supported set when AMBIGUOUS
    progression_evidence_t: tuple = ()
    stage_evidence_t: tuple = ()
    stage_confidence_t: Optional[float] = None
    recommended_policy_family_t: Optional[str] = None


class StageRouter:
    """``predict`` is a pure function of its arguments. ``predicates`` is a test double only: a mapping
    ``Stage -> callable(snapshot, agent_id, target_scope) -> bool`` supplying explicit synthetic positive
    evidence (spec section 8). With the default (none), every real B snapshot abstains to UNKNOWN."""

    def __init__(self, config_version='router-v1', predicates=None):
        self.config_version = config_version
        self.predicates = dict(predicates or {})

    def predict(self, snapshot, agent_id, target_scope, config_version=None):
        if snapshot is None or getattr(snapshot, 'schema_version', None) != SCHEMA_VERSION:
            return StageEstimate(Stage.UNKNOWN, 'UNKNOWN', ('MISSING_OBSERVATION',))
        if config_version is not None and config_version != self.config_version:
            return StageEstimate(Stage.UNKNOWN, 'UNKNOWN', ('CONFIG_MISMATCH',))

        access = self._access_state(snapshot, agent_id, target_scope)
        matched = [s for s in SINGLE_STAGES
                   if self._predicate_holds(s, snapshot, agent_id, target_scope)]
        if not matched:
            return StageEstimate(Stage.UNKNOWN, access, ('NO_POSITIVE_PREDICATE',))
        if len(matched) == 1:
            return StageEstimate(matched[0], access, (), stage_evidence_t=('positive_predicate',))
        return StageEstimate(Stage.AMBIGUOUS, access, ('MULTIPLE_STAGES',), possible_stages=tuple(matched))

    def _predicate_holds(self, stage, snapshot, agent_id, target_scope):
        pred = self.predicates.get(stage)
        return bool(pred(snapshot, agent_id, target_scope)) if pred is not None else False

    @staticmethod
    def _access_state(snapshot, agent_id, target_scope):
        # OBSERVED_PRESENT only if the target host is in the agent's last pre-cutoff session set; an empty
        # or absent set is UNKNOWN, never ABSENT_CONFIRMED (spec section 7, freshness).
        host = target_scope[1] if isinstance(target_scope, tuple) and len(target_scope) == 2 else target_scope
        sessions = snapshot.current_sessions.get(agent_id)
        if sessions is None:
            return 'UNKNOWN'
        return 'OBSERVED_PRESENT' if host in sessions else 'UNKNOWN'



def fixed_stage(stage):
    """stage_provider konstan. Provider berbasis-bukti = REVIEW_REQUIRED (gate-C)."""
    return lambda candidates, knowledge=None, tick=None: stage
