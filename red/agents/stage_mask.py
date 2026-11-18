"""Per-stage action masking for the stage-specialized RL layer (M5, mapping D11).

A specialist for a kill-chain stage scores only that stage's action classes (plus Sleep, always allowed);
Withdraw stays governed by the existing ``--mask-last-withdraw`` elsewhere. The mask is applied at SCORING
(off-stage candidates get -inf), so native action indices and the 22-feature contract are never changed.
Action class names match red.rl.hdqn.goals.ACTION_NAMES.
"""
import numpy as np

STAGE_ACTIONS = {
    'recon': frozenset({'DiscoverRemoteSystems', 'AggressiveServiceDiscovery',
                        'StealthServiceDiscovery', 'DiscoverDeception'}),
    'advance': frozenset({'ExploitRemoteService', 'PrivilegeEscalate'}),
    'impact': frozenset({'Impact', 'DegradeServices'}),
}
ALWAYS_ALLOWED = frozenset({'Sleep'})
STAGES = ('recon', 'advance', 'impact')


def action_name(candidate):
    """Action class name of a (agent, action, feature, target) candidate tuple."""
    action = candidate[1]
    return type(action).__name__ if action is not None else ''


def stage_action_mask(candidates, stage):
    """Boolean mask [N]: True where the candidate's action belongs to ``stage`` (or is always allowed).
    An unknown stage returns all-True (no masking)."""
    legal = STAGE_ACTIONS.get(stage)
    if not legal:
        return np.ones(len(candidates), dtype=bool)
    allowed = legal | ALWAYS_ALLOWED
    return np.array([action_name(c) in allowed for c in candidates], dtype=bool)
