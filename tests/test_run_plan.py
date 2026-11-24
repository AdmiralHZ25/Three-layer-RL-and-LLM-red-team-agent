"""Offline tests for the limits of menu g, and for the rules that stop a run in the loop of run_operation.

No network: the Caldera client is a fake, and the menu function is called with input() patched.
Run from the RED_AGENT folder:  python -B -m unittest tests.test_run_plan -v
"""
import itertools
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import run_menu
from red import config as C
from red.caldera import run_plan
from red.caldera.memory import RunMemory
from red.caldera.models import PotentialLink
from red.caldera.runner import run_operation
from red.caldera.scope import Scope


def _link(ability, host='lab-vm-1'):
    return PotentialLink(paw='p1', ability_id=ability, technique_id='T1', target=host, name=ability,
                         tactic='discovery', command=f'echo {ability}')


class FakeClient:
    """Records what would have been sent. statuses and outputs are given per ability_id."""

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

    def get_candidates(self, op_id, paw):
        return list(self.links)

    def submit(self, op_id, pl, allowed=None):
        self.submitted.append(pl.ability_id)
        link_id = f'link-{len(self.submitted)}'
        self._ability_of[link_id] = pl.ability_id
        return link_id

    def wait(self, op_id, link_id):
        return self.statuses.get(self._ability_of[link_id], '0')

    def get_link_result(self, op_id, link_id):
        return self.outputs.get(self._ability_of[link_id], '')


class FirstChooser:
    """Always picks the first candidate that is still offered."""

    def choose(self, links, stage=None, tick=0, horizon=1, history=None):
        return 0, {'source': 'test'}


class _RunCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _run(self, client, ids, max_steps=5, hosts=('lab-vm-1',), name='run', **kwargs):
        scope = Scope(paw='p1', lab_hosts=hosts, allowed_abilities=frozenset(ids), max_steps=max_steps,
                      stop_file=self.root / 'STOP')
        return run_operation(client, scope, FirstChooser(), 'llm', 'recon', self.root / name,
                             log=lambda *_: None, memory=RunMemory(), **kwargs)


class LimitTests(unittest.TestCase):
    def test_valid_limits_pass(self):
        run_plan.check_run_settings(10, 1800.0, 3)
        run_plan.check_run_settings(1, 0.5, 1)
        run_plan.check_run_settings(20, 60, 5)

    def test_steps_must_be_1_to_20(self):
        for value in (0, 21, True, '5', None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    run_plan.check_run_settings(value, 1800.0, 3)

    def test_seconds_must_be_above_zero(self):
        for value in (0, -1, True, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    run_plan.check_run_settings(10, value, 3)

    def test_failures_must_be_at_least_one(self):
        with self.assertRaises(ValueError):
            run_plan.check_run_settings(10, 1800.0, 0)

    def test_config_defaults_are_valid(self):
        run_plan.check_run_settings(C.CALDERA_RUN_MAX_STEPS, C.CALDERA_RUN_MAX_SECONDS,
                                    C.CALDERA_MAX_CONSECUTIVE_FAILURES)


class FailureRuleTests(_RunCase):
    def test_one_failed_link_does_not_stop_the_run(self):
        client = FakeClient([_link('A'), _link('B'), _link('C')], statuses={'A': '1'})
        summary = self._run(client, 'ABC', max_steps=3)
        self.assertEqual(client.submitted, ['A', 'B', 'C'])
        self.assertEqual(summary['stop'], 'budget_used')
        self.assertEqual(summary['failures'], 1)

    def test_three_failures_in_a_row_stop_the_run(self):
        client = FakeClient([_link(x) for x in 'ABCDE'], statuses={'A': '1', 'B': '1', 'C': '1'})
        summary = self._run(client, 'ABCDE', max_steps=5)
        self.assertEqual(client.submitted, ['A', 'B', 'C'])
        self.assertTrue(summary['stop'].startswith('3 link gagal berturut-turut'), summary['stop'])
        self.assertEqual(summary['failures'], 3)

    def test_a_success_resets_the_count(self):
        # fail, success, fail, fail: with a limit of 2 in a row, the run stops only after the last two.
        client = FakeClient([_link(x) for x in 'ABCDE'], statuses={'A': '1', 'C': '1', 'D': '1'})
        summary = self._run(client, 'ABCDE', max_steps=5, max_failures=2)
        self.assertEqual(client.submitted, ['A', 'B', 'C', 'D'])
        self.assertTrue(summary['stop'].startswith('2 link gagal berturut-turut'), summary['stop'])

    def test_untrusted_link_stops_at_once(self):
        client = FakeClient([_link('A'), _link('B')], statuses={'A': 'untrusted'})
        summary = self._run(client, 'AB', max_steps=5)
        self.assertEqual(client.submitted, ['A'])
        self.assertEqual(summary['stop'], 'link status untrusted')


class ObjectiveStopTests(_RunCase):
    def test_own_host_is_not_a_new_host(self):
        client = FakeClient([_link('A', 'DESKTOP-OWN'), _link('B', 'DESKTOP-OWN')],
                            outputs={'A': 'Host Name: DESKTOP-OWN', 'B': 'Host Name: DESKTOP-OWN'})
        summary = self._run(client, 'AB', max_steps=2, hosts=('DESKTOP-OWN',), goal=('host_lain', 1))
        self.assertEqual(summary['stop'], 'budget_used')
        self.assertFalse(summary['goal_reached'])

    def test_another_host_reaches_the_goal(self):
        client = FakeClient([_link('A', 'DESKTOP-OWN'), _link('B', 'DESKTOP-OWN')],
                            outputs={'A': 'Host Name: DESKTOP-OWN', 'B': 'Host Name: HOST-2'})
        summary = self._run(client, 'AB', max_steps=5, hosts=('DESKTOP-OWN',), goal=('host_lain', 1))
        self.assertEqual(summary['stop'], 'goal_reached')
        self.assertTrue(summary['goal_reached'])
        self.assertEqual(summary['goal'], ['host_lain', 1])
        self.assertEqual(client.submitted, ['A', 'B'])

    def test_time_limit_stops_before_the_next_step(self):
        clock = itertools.chain([0.0, 0.0, 11.0], itertools.repeat(11.0))
        client = FakeClient([_link('A'), _link('B')])
        with mock.patch('red.caldera.runner.time.monotonic', side_effect=lambda: next(clock)):
            summary = self._run(client, 'AB', max_steps=5, max_seconds=10.0)
        self.assertEqual(client.submitted, ['A'])
        self.assertEqual(summary['stop'], 'time_limit')
        self.assertFalse(summary['goal_reached'])

    def test_bad_failure_limit_is_refused_before_anything_is_sent(self):
        client = FakeClient([_link('A')])
        with self.assertRaises(ValueError):
            self._run(client, 'A', max_steps=2, max_failures=0)
        self.assertEqual(client.submitted, [])


class StepRecordTests(_RunCase):
    def test_step_record_names_the_ability_executor_and_link(self):
        link = PotentialLink(paw='p1', ability_id='A', technique_id='T1119', target='lab-vm-1',
                             name='Automated Collection', tactic='collection', command='dir', executor='cmd')
        client = FakeClient([link])
        self._run(client, 'A', max_steps=1, name='rec')
        rec = json.loads((self.root / 'rec' / 'steps.jsonl').read_text(encoding='utf-8').splitlines()[0])
        self.assertEqual(rec['ability_id'], 'A')
        self.assertEqual(rec['ability_name'], 'Automated Collection')
        self.assertEqual(rec['executor'], 'cmd')
        self.assertEqual(rec['link_id'], 'link-1')

    def test_a_step_that_was_not_sent_has_no_link_id(self):
        client = FakeClient([_link('A')])
        scope = Scope(paw='p1', lab_hosts=('lab-vm-1',), allowed_abilities=frozenset({'Z'}), max_steps=1,
                      stop_file=self.root / 'STOP')
        run_operation(client, scope, FirstChooser(), 'llm', 'recon', self.root / 'ns',
                      log=lambda *_: None, memory=RunMemory())
        rec = json.loads((self.root / 'ns' / 'steps.jsonl').read_text(encoding='utf-8').splitlines()[0])
        self.assertEqual(rec['status'], 'not sent')
        self.assertIsNone(rec['link_id'])


class MenuTests(unittest.TestCase):
    def test_objective_is_taken_as_typed_and_empty_cancels(self):
        with mock.patch('builtins.input', return_value='  Temukan domain controller  '):
            self.assertEqual(run_menu._ask_objective(), 'Temukan domain controller')
        with mock.patch('builtins.input', return_value=''):
            self.assertIsNone(run_menu._ask_objective())


if __name__ == '__main__':
    unittest.main()
