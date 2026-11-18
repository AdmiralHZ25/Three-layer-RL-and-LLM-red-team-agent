"""Offline tests for the unattended Caldera runner, choosers, scope, mapping, and features.

No network and no Caldera: the client, the RL agent, and the LLM are all fakes.
Run from the RED_AGENT folder:  python -m unittest tests.test_caldera_runner -v
"""
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from red.caldera.choosers import HybridChooser, LlmChooser, RlChooser, make_chooser
from red.caldera.features import caldera_features
from red.caldera.mapping import CAGE_CLASSES, cage_class
from red.caldera.models import PotentialLink
from red.caldera.runner import run_operation
from red.caldera.scope import Scope, ScopeViolation
from red.llm import FakeLLM
from red.simulator import cage4


def _link(ability, technique, host='lab-vm-1'):
    return PotentialLink(paw='p1', ability_id=ability, technique_id=technique, target=host, name=ability,
                         tactic='discovery', command=f'echo {ability}')


class FakeClient:
    """Records what would have been sent. Statuses are returned in order; the default is success."""

    def __init__(self, links, statuses=()):
        self.links = links
        self.statuses = list(statuses)
        self.submitted = []
        self.finished = []

    @contextmanager
    def operation(self, name='x'):
        try:
            yield 'op-1'
        finally:
            self.finished.append('op-1')

    def is_done(self, op_id):
        return False

    def get_candidates(self, op_id, paw):
        return list(self.links)

    def submit(self, op_id, pl):
        self.submitted.append(pl.ability_id)
        return f'link-{len(self.submitted)}'

    def wait(self, op_id, link_id):
        return self.statuses.pop(0) if self.statuses else '0'


class FixedChooser:
    """Returns the given index on each call, or None when the list runs out."""

    def __init__(self, indexes):
        self.indexes = list(indexes)

    def choose(self, links, stage=None, tick=0, horizon=1):
        if not self.indexes:
            return None, {'source': 'test'}
        return self.indexes.pop(0), {'source': 'test'}


class CrashingChooser:
    def choose(self, links, stage=None, tick=0, horizon=1):
        raise RuntimeError('chooser crashed')


class FakeAgent:
    """Stands in for the RL agent: q_matrix returns the rows given, one row per feature vector."""

    def __init__(self, rows):
        self.q = torch.tensor(rows, dtype=torch.float32)
        self.cfg = types.SimpleNamespace(main_head=0)

    def q_matrix(self, features):
        return self.q


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.links = [_link('a1', 'T1018'), _link('a2', 'T1046')]

    def tearDown(self):
        self.tmp.cleanup()

    def _scope(self, max_steps=3, allowed=('a1', 'a2'), hosts=('lab-vm-1',)):
        return Scope(paw='p1', lab_hosts=hosts, allowed_abilities=frozenset(allowed), max_steps=max_steps,
                     stop_file=self.root / 'STOP')

    def _run(self, client, scope, chooser, name='run'):
        return run_operation(client, scope, chooser, 'test', 'recon', self.root / name, log=lambda *_: None)

    def test_budget_stops_the_run_after_max_steps(self):
        client = FakeClient(self.links)
        summary = self._run(client, self._scope(max_steps=3), FixedChooser([0, 1, 0]))
        self.assertEqual(client.submitted, ['a1', 'a2', 'a1'])
        self.assertEqual(summary['stop'], 'budget_used')
        self.assertEqual(client.finished, ['op-1'])

    def test_stop_file_prevents_any_submission(self):
        client = FakeClient(self.links)
        scope = self._scope()
        scope.stop_file.write_text('stop', encoding='utf-8')
        summary = self._run(client, scope, FixedChooser([0, 0, 0]))
        self.assertEqual(client.submitted, [])
        self.assertEqual(summary['stop'], 'stop_file')

    def test_ability_outside_the_allow_list_is_refused(self):
        client = FakeClient([_link('a9', 'T1018')])
        summary = self._run(client, self._scope(), FixedChooser([0]))
        self.assertEqual(client.submitted, [])
        self.assertTrue(summary['stop'].startswith('scope:'))

    def test_host_outside_the_lab_is_refused(self):
        client = FakeClient([_link('a1', 'T1018', host='office-pc')])
        summary = self._run(client, self._scope(), FixedChooser([0]))
        self.assertEqual(client.submitted, [])
        self.assertTrue(summary['stop'].startswith('scope:'))

    def test_no_valid_choice_fails_closed(self):
        client = FakeClient(self.links)
        summary = self._run(client, self._scope(), FixedChooser([None]))
        self.assertEqual(client.submitted, [])
        self.assertEqual(summary['stop'], 'no_valid_choice')

    def test_a_link_that_needs_a_human_stops_the_run(self):
        client = FakeClient(self.links, statuses=['untrusted'])
        summary = self._run(client, self._scope(), FixedChooser([0, 0, 0]))
        self.assertEqual(client.submitted, ['a1'])
        self.assertEqual(summary['stop'], 'link status untrusted')

    def test_operation_is_finished_even_when_the_chooser_crashes(self):
        client = FakeClient(self.links)
        with self.assertRaises(RuntimeError):
            self._run(client, self._scope(), CrashingChooser())
        self.assertEqual(client.finished, ['op-1'])
        self.assertEqual(client.submitted, [])

    def test_scope_rejects_an_empty_allow_list_and_bad_budget(self):
        with self.assertRaises(ValueError):
            self._scope(allowed=())
        with self.assertRaises(ValueError):
            self._scope(max_steps=0)
        with self.assertRaises(ValueError):
            self._scope(max_steps=99)


class MappingAndFeatures(unittest.TestCase):
    def test_cage_classes_match_the_simulator_names(self):
        self.assertEqual(tuple(CAGE_CLASSES), tuple(cage4.NAMES))

    def test_technique_lookup_accepts_sub_techniques_and_rejects_unknown(self):
        self.assertEqual(cage_class('T1046.001'), 'StealthServiceDiscovery')
        self.assertIsNone(cage_class('T9999'))
        self.assertIsNone(cage_class(''))

    def test_feature_vector_has_the_cage_layout(self):
        vec = caldera_features('Impact', tick=2, horizon=4)
        self.assertEqual(vec.shape, (22,))
        self.assertEqual(int(np.argmax(vec[:10])), cage4.NAMES.index('Impact'))
        self.assertAlmostEqual(float(vec[10]), 0.5)


class Choosers(unittest.TestCase):
    def setUp(self):
        self.links = [_link('a1', 'T1018'), _link('a2', 'T1210'), _link('a3', 'T9999')]

    def test_rl_picks_among_mapped_candidates_when_heads_agree(self):
        # rows for the two mapped candidates; both heads prefer row 0, so the gate passes
        agent = FakeAgent([[0.9, 0.9], [0.1, 0.1]])
        idx, info = RlChooser(agent, theta=0.85, delta=0.8).choose(self.links)
        self.assertEqual(idx, 0)
        self.assertTrue(info['gate_passed'])

    def test_hybrid_uses_the_llm_when_the_gate_fails(self):
        # heads disagree (tau 0.5 < theta), so the gate fails and the LLM chooses among all candidates
        agent = FakeAgent([[0.9, 0.1], [0.1, 0.9]])
        rl = RlChooser(agent, theta=0.85, delta=0.8)
        idx, info = HybridChooser(rl, LlmChooser(FakeLLM(responder=lambda p: 'ACTION: 2'))).choose(self.links)
        self.assertEqual(idx, 2)
        self.assertEqual(info['route'], 'LLM')

    def test_llm_only_mode_fails_closed_on_a_bad_answer(self):
        chooser = make_chooser('llm', llm=FakeLLM(responder=lambda p: 'I choose one'))
        idx, info = chooser.choose(self.links)
        self.assertIsNone(idx)
        self.assertEqual(info['source'], 'llm_rejected')

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            make_chooser('random')


if __name__ == '__main__':
    unittest.main()
