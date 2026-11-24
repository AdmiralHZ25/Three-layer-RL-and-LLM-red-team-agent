"""Choosers for one step of a Caldera operation.

Every chooser returns (index or None, info). None means no valid choice. The runner then stops (fail closed) and
never picks a default candidate.

Modes:
  llm     the LLM chooses among all allowed candidates
  rl      the RL main head chooses among candidates that have a CAGE class (gate values are logged only)
  rl+llm  RL when the Confidence Gate passes; otherwise the LLM chooses among all allowed candidates
"""
import json
from pathlib import Path

import numpy as np

from red.agents.confidence import ConfidenceGate
from red.llm import AcceptanceGate

from .executor import build_link_prompt
from .features import caldera_features
from .mapping import cage_class

MODES = ('llm', 'rl', 'rl+llm')


def read_gate_lock(path):
    """(theta, delta) from the gate lock file. The file is read only, never changed here."""
    gate = json.loads(Path(path).read_text(encoding='utf-8'))['gate']
    return float(gate['theta']), float(gate['delta'])


class LlmChooser:
    def __init__(self, llm, acceptance=None):
        self.llm = llm
        self.acceptance = acceptance or AcceptanceGate()

    def choose(self, links, stage=None, tick=0, horizon=1, history=None):
        res = self.llm.complete(build_link_prompt(links, stage, history))
        info = {'source': 'LLM', 'llm_ok': res.ok, 'llm_error': res.error, 'latency_s': res.latency_s}
        if not res.ok:
            info['source'] = 'llm_failed'
            return None, info
        verdict = self.acceptance.accept(res.text, len(links))
        if verdict['accepted']:
            return verdict['index'], info
        info['source'] = 'llm_rejected'
        info['reason'] = verdict['reason']
        return None, info


class RlChooser:
    def __init__(self, agent, theta, delta):
        self.agent = agent
        self.gate = ConfidenceGate(theta, delta)

    def choose(self, links, stage=None, tick=0, horizon=1, history=None):
        # history is accepted so that every chooser has one signature; RL uses numeric features, not text.
        eligible = [i for i, l in enumerate(links) if cage_class(l.technique_id)]
        if not eligible:
            return None, {'source': 'rl_no_mapped_candidate'}
        classes = [cage_class(links[i].technique_id) for i in eligible]
        feats = np.stack([caldera_features(c, tick, horizon) for c in classes])
        qm = self.agent.q_matrix(feats).detach().cpu().numpy().astype(np.float64)
        d = self.gate.decide(qm, self.agent.cfg.main_head)
        pick = d['action']
        info = {'source': 'RL', 'cage_class': classes[pick], 'tau': d['tau'], 'sigma': d['sigma'],
                'gate_passed': d['passed']}
        return eligible[pick], info

    def vote(self, links, index, tick=0, horizon=1):
        """Acceptance Gate for an LLM action (solution overview, figure 6): at least half of the K heads must rank the
        action first among the mapped candidates. An action without a CAGE class has no head to vote for it, so refused."""
        eligible = [i for i, l in enumerate(links) if cage_class(l.technique_id)]
        if index not in eligible:
            return {'accepted': False, 'mapped': False, 'votes': 0, 'heads': None}
        classes = [cage_class(links[i].technique_id) for i in eligible]
        feats = np.stack([caldera_features(c, tick, horizon) for c in classes])
        qm = self.agent.q_matrix(feats).detach().cpu().numpy().astype(np.float64)
        pos = eligible.index(index)
        heads = qm.shape[1]
        votes = sum(int(qm[:, k].argmax()) == pos for k in range(heads))
        return {'accepted': 2 * votes >= heads, 'mapped': True, 'votes': votes, 'heads': heads}


class HybridChooser:
    """RL > gate > LLM: the RL pick when the gate passes, otherwise the LLM."""

    def __init__(self, rl, llm):
        self.rl = rl
        self.llm = llm

    def choose(self, links, stage=None, tick=0, horizon=1, history=None):
        rl_idx, rl_info = self.rl.choose(links, stage, tick, horizon)
        if rl_idx is not None and rl_info.get('gate_passed'):
            return rl_idx, {**rl_info, 'route': 'RL'}
        llm_idx, llm_info = self.llm.choose(links, stage, tick, horizon, history=history)
        return llm_idx, {**llm_info, 'route': 'LLM', 'rl_info': rl_info}


def make_chooser(mode, llm=None, agent=None, theta=None, delta=None):
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode!r}; use one of {MODES}')
    if mode == 'llm':
        return LlmChooser(llm)
    rl = RlChooser(agent, theta, delta)
    if mode == 'rl':
        return rl
    return HybridChooser(rl, LlmChooser(llm))
