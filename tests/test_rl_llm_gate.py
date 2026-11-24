"""Offline tests for the RL confidence gate + LLM integration in the agentic Caldera loop (menu g).

No network, no Caldera, no real RL model: a fake chooser stands in for RlChooser.
Run from the RED_AGENT folder:  python -B -m unittest tests.test_rl_llm_gate -v
"""
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from red.caldera.agent_loop import run_agentic
from red.caldera.models import PotentialLink
from red.llm import FakeLLM

PLAN = 'RENCANA:\n1. Cari host di jaringan\n2. Eksploit layanan yang ditemukan'
CHOICE = 'ACTION: 0'
DONE_NO = 'DONE: no\nEVIDENCE:'
DONE_YES = 'DONE: yes\nEVIDENCE: host found'


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
    """Deterministic fake: always picks ``pick_idx``, passes gate when ``gate_passed=True``."""

    def __init__(self, pick_idx=0, gate_passed=True):
        self.pick_idx = pick_idx
        self.gate_passed = gate_passed
        self.calls = []

    def choose(self, links, stage=None, tick=0, horizon=1, history=None):
        self.calls.append([l.ability_id for l in links])
        if self.pick_idx is None or not links:
            return None, {'source': 'rl_no_mapped_candidate'}
        idx = min(self.pick_idx, len(links) - 1)
        return idx, {'source': 'RL', 'gate_passed': self.gate_passed, 'tau': 1.0, 'sigma': 0.1,
                     'cage_class': 'StealthServiceDiscovery'}


def responder(plan=PLAN, choice=CHOICE, done=DONE_NO):
    def respond(prompt):
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
    def test_rl_pick_is_first_in_batch_when_gate_passes(self):
        # 3 links: A, B, C. RL picks index 1 (B). LLM sees [A, C] and picks index 0 (A).
        links = [_link('A'), _link('B'), _link('C')]
        rl = FakeRlChooser(pick_idx=1, gate_passed=True)
        summary, client = _run(links, responder(choice='ACTION: 0'), rl_chooser=rl, max_steps=1)
        # Both B (RL) and A (LLM) should be submitted; B first.
        self.assertIn('B', client.submitted)
        self.assertIn('A', client.submitted)
        self.assertEqual(client.submitted[0], 'B')

    def test_rl_pick_is_excluded_from_llm_shortlist(self):
        # RL picks index 0 (A). LLM should be shown [B, C] only. LLM picks index 1 → C.
        links = [_link('A'), _link('B'), _link('C')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=True)
        # 'ACTION: 1' selects index 1 from the LLM shortlist [B, C], which is 'C'.
        summary, client = _run(links, responder(choice='ACTION: 1'), rl_chooser=rl, max_steps=1)
        self.assertIn('A', client.submitted)   # RL pick
        self.assertIn('C', client.submitted)   # LLM pick (index 1 from [B, C])
        self.assertNotIn('B', client.submitted)

    def test_rl_pick_alone_when_llm_gives_invalid_reply(self):
        # RL passes, LLM returns garbage → only RL pick is sent, run does not stop.
        links = [_link('A'), _link('B')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=True)
        summary, client = _run(links, responder(choice='apa ya'), rl_chooser=rl, max_steps=1)
        self.assertEqual(client.submitted, ['A'])

    def test_llm_stop_wins_even_when_rl_gate_passes(self):
        # RL passes, but LLM decides objective done → nothing sent, run stops.
        links = [_link('A'), _link('B')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=True)
        stop_reply = 'ACTION: STOP\nALASAN: not reachable'
        summary, client = _run(links, responder(choice=stop_reply), rl_chooser=rl, max_steps=2)
        self.assertEqual(client.submitted, [])
        self.assertTrue(summary['stop'].startswith('planner_stopped'))


class GateFailedTests(unittest.TestCase):
    def test_llm_picks_all_when_gate_fails(self):
        # RL gate fails → LLM sees all links and picks normally.
        links = [_link('A'), _link('B'), _link('C')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=False)
        summary, client = _run(links, responder(choice='ACTION: 0, 2'), rl_chooser=rl, max_steps=1)
        self.assertIn('A', client.submitted)
        self.assertIn('C', client.submitted)
        # B was not chosen by LLM
        self.assertNotIn('B', client.submitted)

    def test_llm_picks_all_when_rl_has_no_eligible_candidate(self):
        # RL returns None → LLM handles all.
        links = [_link('A'), _link('B')]
        rl = FakeRlChooser(pick_idx=None, gate_passed=False)
        summary, client = _run(links, responder(choice='ACTION: 1'), rl_chooser=rl, max_steps=1)
        self.assertEqual(client.submitted, ['B'])


class RlCallTests(unittest.TestCase):
    def test_rl_is_called_on_each_step(self):
        # 2 links A and B. Step 1: LLM picks A (gate fails). Step 2: B remains, RL called again.
        links = [_link('A'), _link('B')]
        rl = FakeRlChooser(pick_idx=0, gate_passed=False)
        # LLM always picks index 0 from whatever it sees
        _run(links, responder(choice='ACTION: 0'), rl_chooser=rl, max_steps=2)
        self.assertEqual(len(rl.calls), 2)

    def test_rl_sees_the_same_shortlist_as_llm_before_removal(self):
        # RL receives the full shortlist; LLM receives it minus the RL pick.
        links = [_link('A'), _link('B'), _link('C')]
        rl = FakeRlChooser(pick_idx=2, gate_passed=True)
        seen_prompts = []
        _run(links, responder(choice='ACTION: 0', plan=PLAN), rl_chooser=rl, max_steps=1)
        # RL saw all 3; LLM shortlist has 2 (A and B, C removed).
        self.assertEqual(len(rl.calls[0]), 3)


if __name__ == '__main__':
    unittest.main()
