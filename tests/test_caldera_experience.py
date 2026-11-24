"""Offline tests for step 9 (experience recording) and step 10 (retrain) of the Caldera loop.

No network and no Caldera: links are plain objects with a technique_id, the agent is a fake.
Run from the RED_AGENT folder:  python -B -m unittest tests.test_caldera_experience -v
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from red.caldera import retrain as retrain_mod
from red.caldera.experience import (FACT_BONUS, FAIL_PENALTY, OBJECTIVE_BONUS, STEP_COST, SUCCESS_BONUS,
                                    CalderaExperienceRecorder)
from red.caldera.features import caldera_features
from red.caldera.agent_loop import run_agentic
from red.caldera.mapping import cage_class
from red.llm import FakeLLM
from tests.test_agentic_planner import NEUTRAL_PLAN, FakeClient, responder
from tests.test_agentic_planner import _link as plan_link

MAX_STEPS = 10
MAPPED = 'T1047'      # ExploitRemoteService
UNMAPPED = 'T9999'


def _link(technique_id):
    return SimpleNamespace(technique_id=technique_id)


def _quiet(*_args):
    pass


class FakeReplay(list):
    def add(self, *item):
        self.append(item)


class FakeAgent:
    """Stands in for MultiHeadDDQNAgent: a replay buffer, a config, and a train_step that counts calls."""

    def __init__(self, batch_size=32, warmup=64):
        self.replay = FakeReplay()
        self.cfg = SimpleNamespace(batch_size=batch_size, warmup=warmup)
        self.online = mock.Mock()
        self.train_calls = 0
        self.saved = []

    def train_step(self):
        self.train_calls += 1
        return 0.5

    def save(self, path, extra=None):
        self.saved.append(path)


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.rec = CalderaExperienceRecorder(MAX_STEPS)

    def test_each_link_in_a_batch_keeps_its_own_outcome(self):
        self.rec.record_batch([_link(MAPPED)] * 3, tick=2)
        self.rec.close_batch([('0', 1), ('1', 0), ('0', 0)], next_links=[], next_tick=3)
        rewards = [t.reward for t in self.rec.transitions]
        self.assertEqual(len(rewards), 3)
        self.assertAlmostEqual(rewards[0], STEP_COST + SUCCESS_BONUS + FACT_BONUS)
        self.assertAlmostEqual(rewards[1], STEP_COST + FAIL_PENALTY)
        self.assertAlmostEqual(rewards[2], STEP_COST + SUCCESS_BONUS)
        expected = caldera_features(cage_class(MAPPED), 2, MAX_STEPS)
        for t in self.rec.transitions:
            np.testing.assert_allclose(t.feature, expected)
            self.assertFalse(t.done)

    def test_objective_bonus_is_added_once_and_ends_the_run(self):
        self.rec.record_batch([_link(MAPPED)] * 3, tick=1)
        self.rec.close_batch([('0', 0), ('0', 0), ('0', 0)], next_links=[], next_tick=2, objective_reached=True)
        rewards = [t.reward for t in self.rec.transitions]
        self.assertAlmostEqual(sum(rewards), 3 * (STEP_COST + SUCCESS_BONUS) + OBJECTIVE_BONUS)
        self.assertTrue(all(t.done for t in self.rec.transitions))

    def test_unmapped_link_is_skipped_and_the_others_stay_aligned(self):
        self.rec.record_batch([_link(MAPPED), _link(UNMAPPED), _link(MAPPED)], tick=1)
        self.rec.close_batch([('1', 0), ('0', 0), ('0', 1)], next_links=[], next_tick=2)
        rewards = [t.reward for t in self.rec.transitions]
        self.assertEqual(len(rewards), 2)
        self.assertAlmostEqual(rewards[0], STEP_COST + FAIL_PENALTY)
        self.assertAlmostEqual(rewards[1], STEP_COST + SUCCESS_BONUS + FACT_BONUS)

    def test_outcome_count_must_match_the_batch(self):
        self.rec.record_batch([_link(MAPPED)] * 2, tick=1)
        with self.assertRaises(ValueError):
            self.rec.close_batch([('0', 0)], next_links=[], next_tick=2)

    def test_flush_seals_an_open_batch_once_as_terminal(self):
        self.rec.record_batch([_link(MAPPED)], tick=1)
        self.rec.flush()
        self.rec.flush()
        self.assertEqual(len(self.rec.transitions), 1)
        self.assertAlmostEqual(self.rec.transitions[0].reward, STEP_COST + FAIL_PENALTY)
        self.assertTrue(self.rec.transitions[0].done)

    def test_saved_transitions_load_back_with_the_same_rewards(self):
        self.rec.record_batch([_link(MAPPED)] * 3, tick=1)
        self.rec.close_batch([('0', 1), ('1', 0), ('0', 0)], next_links=[], next_tick=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'experience.npz'
            self.rec.save(path)
            loaded = CalderaExperienceRecorder.load(path)
        self.assertEqual([t.reward for t in loaded], [t.reward for t in self.rec.transitions])


class RetrainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        rec = CalderaExperienceRecorder(MAX_STEPS)
        rec.record_batch([_link(MAPPED)] * 3, tick=1)
        rec.close_batch([('0', 0), ('1', 0), ('0', 0)], next_links=[], next_tick=2)
        self.exp = self.tmp / 'experience.npz'
        rec.save(self.exp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_too_few_transitions_run_no_gradient_step(self):
        agent = FakeAgent(batch_size=32, warmup=64)
        stats = retrain_mod.retrain(agent, [self.exp], n_steps=5, log=_quiet)
        self.assertEqual(stats['transitions_loaded'], 3)
        self.assertEqual(stats['steps_run'], 0)
        self.assertEqual(agent.train_calls, 0)

    def test_enough_transitions_run_the_requested_gradient_steps(self):
        agent = FakeAgent(batch_size=2, warmup=2)
        stats = retrain_mod.retrain(agent, [self.exp], n_steps=4, log=_quiet)
        self.assertEqual(stats['steps_run'], 4)
        self.assertEqual(agent.train_calls, 4)

    def test_checkpoint_is_not_saved_when_no_gradient_step_ran(self):
        agent = FakeAgent(batch_size=32, warmup=64)
        out = self.tmp / 'out.pt'
        with mock.patch.object(retrain_mod.MultiHeadDDQNAgent, 'load', return_value=agent):
            result = retrain_mod.retrain_from_checkpoint(self.tmp / 'in.pt', [self.exp], out, n_steps=5, log=_quiet)
        self.assertIsNone(result)
        self.assertEqual(agent.saved, [])
        self.assertFalse(out.exists())

    def test_existing_output_is_never_overwritten(self):
        out = self.tmp / 'out.pt'
        out.write_bytes(b'x')
        with mock.patch.object(retrain_mod.MultiHeadDDQNAgent, 'load') as load:
            with self.assertRaises(FileExistsError):
                retrain_mod.retrain_from_checkpoint(self.tmp / 'in.pt', [self.exp], out, n_steps=5, log=_quiet)
        load.assert_not_called()
        self.assertEqual(out.read_bytes(), b'x')


class AfterRunTests(unittest.TestCase):
    """Step 10 trigger: fine-tunes on the experiences of all runs, into a candidate file."""

    def test_no_experience_files_means_no_training(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(retrain_mod, 'retrain_from_checkpoint') as retrain:
            out = retrain_mod.retrain_after_run(Path(tmp) / 'runs', 'in.pt', Path(tmp) / 'out.pt', log=_quiet)
        self.assertIsNone(out)
        retrain.assert_not_called()

    def test_every_runs_experience_file_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / 'runs'
            for name in ('run-b', 'run-a'):
                (runs / name).mkdir(parents=True)
                (runs / name / 'experience.npz').write_bytes(b'x')
            with mock.patch.object(retrain_mod, 'retrain_from_checkpoint', return_value='out.pt') as retrain:
                out = retrain_mod.retrain_after_run(runs, 'in.pt', Path(tmp) / 'out.pt', log=_quiet)
        self.assertEqual(out, 'out.pt')
        paths = retrain.call_args.args[1]
        self.assertEqual([p.parent.name for p in paths], ['run-a', 'run-b'])


class AgentLoopRecordingTests(unittest.TestCase):
    """The real loop (run_agentic) closes one record per link, also when the link fails."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run_batch(self, statuses):
        links = [plan_link('A', 'List users', technique=MAPPED), plan_link('B', 'Query B', technique=MAPPED)]
        client = FakeClient(links, statuses=statuses)
        rec = CalderaExperienceRecorder(MAX_STEPS)
        run_agentic(client, FakeLLM(responder(choice='ACTION: 0, 1', plan=NEUTRAL_PLAN)), 'Lakukan langkah uji',
                    'p1', ('lab-vm-1',), self.tmp / 'run', stop_file=self.tmp / 'STOP', max_steps=1,
                    max_failures=3, experience_recorder=rec, log=_quiet)
        return client, rec

    def test_each_link_of_a_batch_gets_its_own_transition(self):
        client, rec = self._run_batch({'A': '0', 'B': '1'})
        self.assertEqual(client.submitted, ['A', 'B'])
        rewards = sorted(round(t.reward, 6) for t in rec.transitions)
        self.assertEqual(rewards, [-0.15, 0.95])

    def test_a_batch_where_every_link_fails_is_still_recorded(self):
        _, rec = self._run_batch({'A': '1', 'B': '1'})
        self.assertEqual(len(rec.transitions), 2)
        for t in rec.transitions:
            self.assertAlmostEqual(t.reward, STEP_COST + FAIL_PENALTY)


if __name__ == '__main__':
    unittest.main()
