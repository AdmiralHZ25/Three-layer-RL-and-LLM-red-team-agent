"""Full three-layer pipeline with stage-routed RL specialists (M5.4).

Layer 1 is a stage specialist (routed by a ``stage_provider``) plus the Confidence Gate over that
specialist's MASKED K-head Q: the gate passes -> use the RL action; otherwise on-prem LLM -> Acceptance
Gate -> API LLM -> fallback to the specialist's gated pick. The stage comes from the caller's ``stage_provider``
(``fixed_stage`` or ``router.snapshot.router_stage_provider``); an unknown stage routes to the general agent
(``GENERAL``), which must be registered in the specialist set. Every selection is a legal candidate index, so
native indices and the 22-feature contract never change, and the RL is the approved multi-head DDQN only.
"""
import numpy as np

from red.simulator.chooser import Choice
from red.agents.confidence import ConfidenceGate
from red.llm.acceptance import AcceptanceGate
from red.llm.prompt import build_prompt, llm_options
from red.agents.stages import route_and_mask
from red.router.snapshot import GENERAL


class StagedHybridChooser:
    def __init__(self, specialists, stage_provider, theta, delta, default_stage=GENERAL,
                 llm_local=None, llm_api=None, acceptance=None, main_head=None, llm_only=False, prompt_goal=None):
        self.specialists = specialists
        self.stage_provider = stage_provider
        self.gate = ConfidenceGate(theta, delta)
        self.default_stage = default_stage
        self.llm_local = llm_local
        self.llm_api = llm_api
        self.acceptance = acceptance or AcceptanceGate()
        self.main_head = main_head
        self.llm_only = llm_only    # LLM-only framework: the LLM chooses every step; the gate is logged, not used
        self.prompt_goal = prompt_goal  # optional mission line for the LLM prompt; None = original prompt

    def choose(self, candidates, knowledge=None, tick=None):
        stage = self.stage_provider(candidates, knowledge, tick)
        routed, mask = route_and_mask(self.specialists, stage, candidates, self.default_stage)
        agent = self.specialists.agent_for(routed)
        if agent is None:
            raise ValueError(f'no specialist for stage {routed!r}')
        mh = agent.cfg.main_head if self.main_head is None else self.main_head
        q = agent.q_matrix([c[2] for c in candidates]).detach().cpu().numpy().astype(np.float64)  # [N, K]
        qm = np.where(mask[:, None], q, -np.inf)                 # off-stage actions out of contention
        d = self.gate.decide(qm, mh)
        meta = {'stage': stage, 'routed_stage': routed, 'tau': d['tau'], 'sigma': d['sigma'],
                'passed': d['passed'], 'n_masked': int((~mask).sum())}
        if d['passed'] and not self.llm_only:
            return Choice(index=d['action'], source='rl:' + routed, meta={**meta, 'layer': 'rl'})

        opts = llm_options(candidates, qm[:, mh], mask)   # legal, one per label; the LLM index refers to opts
        main_head_q = qm[opts, mh] if self.acceptance.top_m is not None else None
        prompt = build_prompt(candidates, knowledge, tick, opts, goal=self.prompt_goal)
        for layer, llm in (('llm', self.llm_local), ('api', self.llm_api)):
            if llm is None:
                continue
            res = llm.complete(prompt)
            meta.setdefault('llm_calls', []).append(
                {'layer': layer, 'ok': res.ok, 'latency_s': res.latency_s, 'model': res.model,
                 'prompt_tokens': res.prompt_tokens, 'completion_tokens': res.completion_tokens,
                 'error': res.error, 'text': res.text})
            if not res.ok:
                continue
            verdict = self.acceptance.accept(res.text, len(opts), main_head_q)
            if verdict['accepted']:
                return Choice(index=opts[verdict['index']], source=layer, meta={**meta, 'layer': layer})
            meta.setdefault('rejections', []).append({'layer': layer, 'reason': verdict['reason']})
        if self.llm_only:
            return self._sleep_choice(candidates, mask, meta)
        return Choice(index=d['action'], source='fallback', meta={**meta, 'layer': 'fallback'})

    @staticmethod
    def _sleep_choice(candidates, mask, meta):
        """LLM-only fallback when no LLM answer is accepted: the legal Sleep action, never an RL pick."""
        idx = next((i for i, c in enumerate(candidates) if mask[i] and type(c[1]).__name__ == 'Sleep'), None)
        if idx is None:
            raise ValueError('LLM-only fallback needs a legal Sleep action; none is among the candidates')
        return Choice(index=idx, source='sleep', meta={**meta, 'layer': 'sleep'})
