"""Offline tests for the RL confidence gate + LLM in the agentic Caldera loop (menu g), RL+LLM mode (solution overview,
figure 6): gate passes -> RL action alone; gate fails -> one LLM action, sent only when half of the heads agree.

No network, no Caldera, no real RL model: a fake chooser stands in for RlChooser, and the vote tests use a fake Q-agent.
Run from the RED_AGENT folder:  python -B -m unittest tests.test_rl_llm_gate -v
"""
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from red.caldera.agent_loop import run_agentic
from red.caldera.choosers import RlChooser
from red.caldera.models import PotentialLink
from red.llm import FakeLLM

PLAN = 'RENCANA:\n1. Cari host di jaringan\n2. Eksploit layanan yang ditemukan'
CHOICE = 'ACTION: 0'
DONE_NO = 'DONE: no\nEVIDENCE:'
REFUSED_BY_ACCEPTANCE = '3 langkah ditolak Acceptance Gate berturut-turut'
# two mapped candidates x five heads: heads 0-2 rank candidate 0 first, heads 3-4 rank candidate 1 first
SPLIT_ROWS = [[0.9, 0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.1, 0.9, 0.9]]


def _link(ability, name=None, tactic='discovery', technique='T1087'):
    return PotentialLink(paw='p1', ability_id=ability, name=name or ability,
                         tactic=tactic, technique_id=technique,
                         target='lab-vm-1', command=f'echo {ability}', executor='psh')


class FakeClient:
    def __init__(self, links, outputs=None, statuses=None):
        self.links = links
        self.outputs = outputs or {}
        self.statuses = statuses or {}
        self.submitted = []
        self._ability_of = {}

    @contextmanager
    def operation(self, name='x'):
        yield 'op-1'

    def is_done(self, op_id):
        return False

    def get_all_candidates(self, op_id, paw):
        return list(self.links)

    def submit(self, op_id, pl, allowed=None):
        self.submitted.append(pl.ability_id)
        lid = f'link-{len(self.submitted)}'
        self._ability_of[lid] = pl.ability_id
        return lid

    def wait(self, op_id, link_id):
        return self.statuses.get(self._ability_of[link_id], '0')

    def get_link_result(self, op_id, link_id):
        return self.outputs.get(self._ability_of[link_id], 'output ok')


class FakeRlChooser:
    """Deterministic fake: picks ``pick_idx``; the gate passes when ``gate_passed``; the vote is ``vote_accepted``."""

    def __init__(self, pick_idx=0, gate_passed=True, vote_accepted=True):
        self.pick_idx = pick_idx
        self.gate_passed = gate_passed
        self.vote_accepted = vote_accepted
        self.calls = []
        self.votes = []

    def choose(self, links, stage=None, tick=0, horizon=1, history=None):
        self.calls.append([l.ability_id for l in links])
        if self.pick_idx is None or not links:
            return None, {'source': 'rl_no_mapped_candidate'}
        idx = min(self.pick_idx, len(links) - 1)
        return idx, {'source': 'RL', 'gate_passed': self.gate_passed, 'tau': 1.0, 'sigma': 0.1,
                     'cage_class': 'StealthServiceDiscovery'}

    def vote(self, links, index, tick=0, horizon=1):
        self.votes.append(links[index].ability_id)
        return {'accepted': self.vote_accepted, 'mapped': True,
                'votes': 3 if self.vote_accepted else 1, 'heads': 5}


class _Q:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeQAgent:
    """Stands in for MultiHeadDDQNAgent: q_matrix returns the given rows (one per candidate, one column per head)."""

    def __init__(self, rows):
        self.cfg = SimpleNamespace(main_head=0)
        self.rows = np.array(rows, dtype=float)

    def q_matrix(self, features):
        return _Q(self.rows)


def responder(plan=PLAN, choice=CHOICE, done=DONE_NO, seen=None):
    def respond(prompt):
        if seen is not None:
            seen.append(prompt)
        if 'Buat rencana singkat' in prompt:
            return plan
        if 'Pilih sampai' in prompt:
            return choice
        return done
    return respond


def _run(links, llm_responder, rl_chooser=None, max_steps=3, outputs=None, statuses=None):
    client = FakeClient(links, outputs=outputs, statuses=statuses)
    llm = FakeLLM(responder=llm_responder)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / 'run'
        stop = Path(tmp) / 'STOP'
        summary = run_agentic(client, llm, 'Find hosts in the lab', 'p1', ('lab-vm-1',),
                              run_dir, stop_file=stop, max_steps=max_steps, rl_chooser=rl_chooser)
    return summary, client


class ModeTests(unittest.TestCase):
    def test_mode_is_llm_when_no_chooser(self):
        links = [_link('A')]
        summary, _ = _run(links, responder())
        self.assertEqual(summary['mode'], 'llm')

    def test_mode_is_rl_llm_when_chooser_given(self):
        links = [_link('A')]
        summary, _ = _run(links, responder(), rl_chooser=FakeRlChooser(0))
        self.assertEqual(summary['mode'], 'rl+llm')


class GatePassedTests(unittest.TestCase):
    def test_rl_action_is_the_only_action_when_gate_passes(self):
        links = [_link('A'), _link('B'), _link('C')]
        rl = FakeRlChooser(pick_idx=1, gate_passed=True)
        summary, client = _run(links, responder(choice='ACTION: 0'), rl_chooser=rl, max_steps=1)
        self.assertEqual(client.submitted, ['B'])
        self.assertEqual((summary['rl_actions'], summary['llm_actions']), (1, 0))

    def test_llm_is_not_asked_to_choose_when_gate_passes(self):
        links = [_link('A'), _link('B')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=True)
        seen = []
        _run(links, responder(seen=seen), rl_chooser=rl, max_steps=1)
        self.assertEqual([p for p in seen if 'Pilih sampai' in p], [])

    def test_rl_sees_the_full_shortlist(self):
        links = [_link('A'), _link('B'), _link('C')]
        rl = FakeRlChooser(pick_idx=2, gate_passed=True)
        _run(links, responder(), rl_chooser=rl, max_steps=1)
        self.assertEqual(len(rl.calls[0]), 3)


class GateFailedTests(unittest.TestCase):
    def test_llm_action_is_sent_when_half_the_heads_agree(self):
        links = [_link('A'), _link('B'), _link('C')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=False, vote_accepted=True)
        summary, client = _run(links, responder(choice='ACTION: 2'), rl_chooser=rl, max_steps=1)
        self.assertEqual(client.submitted, ['C'])
        self.assertEqual(rl.votes, ['C'])
        self.assertEqual(summary['llm_actions'], 1)

    def test_llm_action_is_refused_without_half_the_heads(self):
        links = [_link('A'), _link('B'), _link('C')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=False, vote_accepted=False)
        summary, client = _run(links, responder(choice='ACTION: 2'), rl_chooser=rl, max_steps=3)
        self.assertEqual(client.submitted, [])
        self.assertEqual(summary['llm_rejected'], 3)
        self.assertEqual(summary['stop'], REFUSED_BY_ACCEPTANCE)

    def test_llm_must_give_exactly_one_action(self):
        links = [_link('A'), _link('B'), _link('C')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=False, vote_accepted=True)
        summary, client = _run(links, responder(choice='ACTION: 0, 2'), rl_chooser=rl, max_steps=1)
        self.assertEqual(client.submitted, [])
        self.assertEqual(summary['llm_rejected'], 1)
        self.assertEqual(rl.votes, [])

    def test_llm_stop_still_ends_the_run_when_gate_fails(self):
        links = [_link('A'), _link('B')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=False)
        stop_reply = 'ACTION: STOP\nALASAN: not reachable'
        summary, client = _run(links, responder(choice=stop_reply), rl_chooser=rl, max_steps=2)
        self.assertEqual(client.submitted, [])
        self.assertEqual(summary['stop'], 'planner_stopped: not reachable')

    def test_rl_is_called_on_each_step(self):
        links = [_link('A'), _link('B')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=False, vote_accepted=True)
        _run(links, responder(choice='ACTION: 0'), rl_chooser=rl, max_steps=2)
        self.assertEqual(len(rl.calls), 2)


class VoteTests(unittest.TestCase):
    """The real RlChooser.vote: the heads, not the LLM, decide whether an LLM action is sent."""

    def setUp(self):
        self.links = [_link('a1', technique='T1018'), _link('a2', technique='T1210'), _link('a3', technique='T9999')]
        self.rl = RlChooser(FakeQAgent(SPLIT_ROWS), theta=0.85, delta=0.8)

    def test_three_of_five_heads_accept_the_action(self):
        verdict = self.rl.vote(self.links, 0)
        self.assertEqual((verdict['accepted'], verdict['votes'], verdict['heads']), (True, 3, 5))

    def test_two_of_five_heads_refuse_the_action(self):
        verdict = self.rl.vote(self.links, 1)
        self.assertEqual((verdict['accepted'], verdict['votes']), (False, 2))

    def test_action_without_a_cage_class_cannot_be_voted(self):
        verdict = self.rl.vote(self.links, 2)
        self.assertEqual((verdict['accepted'], verdict['mapped']), (False, False))

    def test_unmapped_llm_action_is_refused_in_the_loop(self):
        links = [_link('A', technique='T1018'), _link('B', technique='T9999'), _link('C', technique='T1210')]
        rl = RlChooser(FakeQAgent(SPLIT_ROWS), theta=0.85, delta=0.8)   # gate fails: 3 of 5 heads agree, theta 0.85
        summary, client = _run(links, responder(choice='ACTION: 1'), rl_chooser=rl, max_steps=3)
        self.assertEqual(client.submitted, [])
        self.assertEqual(summary['stop'], REFUSED_BY_ACCEPTANCE)


if __name__ == '__main__':
    unittest.main()
