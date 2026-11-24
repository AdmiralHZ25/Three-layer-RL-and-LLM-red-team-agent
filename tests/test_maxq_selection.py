"""Offline tests for the max-Q specialist selection (decision 25 Nov 2026): the specialist that owns the candidate with the
highest main-head Q acts. No stage prediction, no router, no torch: fake Q-agents with one value per candidate.

Run from the RED_AGENT folder:  python -B -m unittest tests.test_maxq_selection -v
"""
import unittest
from types import SimpleNamespace

import numpy as np

from red.agents.stages import StageSpecialists, max_q_stage_provider
from red.pipeline.chooser import StagedHybridChooser
from red.router.snapshot import GENERAL

K = 5   # heads, as in the trained agents


def _action(name):
    return type(name, (), {})()


class _Q:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return _Q(self.values[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeAgent:
    """q_matrix returns one row per candidate (all K heads equal to that candidate's value for this agent)."""

    def __init__(self, values, main_head=0):
        self.cfg = SimpleNamespace(main_head=main_head)
        self.values = list(values)

    def q_matrix(self, features):
        assert len(features) == len(self.values), 'one value per candidate'
        return _Q(np.array([[v] * K for v in self.values], dtype=float))


def _cands(*action_names):
    return [('red_agent_0', _action(n), np.zeros(22), 'host1') for n in action_names]


class ProviderTests(unittest.TestCase):
    def test_stage_of_the_best_scoring_candidate_wins(self):
        cands = _cands('DiscoverRemoteSystems', 'ExploitRemoteService', 'Impact')
        spec = StageSpecialists({'recon': FakeAgent([0.2, 0.0, 0.0]),
                                 'advance': FakeAgent([0.0, 0.9, 0.0]),
                                 'impact': FakeAgent([0.0, 0.0, 0.5]),
                                 GENERAL: FakeAgent([0.0, 0.0, 0.0])})
        self.assertEqual(max_q_stage_provider(spec)(cands), 'advance')

    def test_shared_sleep_goes_to_the_specialist_with_the_higher_value(self):
        cands = _cands('Sleep')
        spec = StageSpecialists({'recon': FakeAgent([0.1]), 'impact': FakeAgent([0.7])})
        self.assertEqual(max_q_stage_provider(spec)(cands), 'impact')

    def test_missing_specialist_is_skipped(self):
        cands = _cands('DiscoverRemoteSystems', 'ExploitRemoteService')
        spec = StageSpecialists({'advance': FakeAgent([0.0, 0.3])})
        self.assertEqual(max_q_stage_provider(spec)(cands), 'advance')

    def test_no_specialist_can_score_returns_none(self):
        cands = _cands('Withdraw')   # not in any stage and not Sleep
        spec = StageSpecialists({'recon': FakeAgent([0.9]), 'advance': FakeAgent([0.9])})
        self.assertIsNone(max_q_stage_provider(spec)(cands))

    def test_same_input_gives_the_same_stage(self):
        cands = _cands('DiscoverRemoteSystems', 'Impact')
        spec = StageSpecialists({'recon': FakeAgent([0.4, 0.0]), 'impact': FakeAgent([0.0, 0.4])})
        provider = max_q_stage_provider(spec)
        self.assertEqual(provider(cands), provider(cands))


class ChooserTests(unittest.TestCase):
    def test_chooser_acts_with_the_owner_of_the_global_best_action(self):
        # the advance specialist owns the best candidate (index 1, value 0.9); theta 0.5 and delta 1.0 let the gate pass
        cands = _cands('DiscoverRemoteSystems', 'ExploitRemoteService', 'Impact')
        spec = StageSpecialists({'recon': FakeAgent([0.2, 0.0, 0.0]),
                                 'advance': FakeAgent([0.0, 0.9, 0.0]),
                                 'impact': FakeAgent([0.0, 0.0, 0.5]),
                                 GENERAL: FakeAgent([0.0, 0.0, 0.0])})
        chooser = StagedHybridChooser(spec, max_q_stage_provider(spec), theta=0.5, delta=1.0)
        choice = chooser.choose(cands)
        self.assertEqual(choice.index, 1)
        self.assertEqual(choice.source, 'rl:advance')


if __name__ == '__main__':
    unittest.main()
