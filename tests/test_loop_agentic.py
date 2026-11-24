"""Offline tests for the goal and the stage gate of the Caldera loop: fact extraction (IP, host, port, user), the goal that
stops the run, the stage gate, and the unchanged default loop.

No network and no Caldera: the client is a fake, and the LLM is FakeLLM. Nothing here imports the Caldera client.
Run from the RED_AGENT folder:  python -B -m unittest tests.test_loop_agentic -v
"""
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from red.caldera.choosers import LlmChooser
from red.caldera.facts import check_need, extract_facts, goal_reached
from red.caldera.memory import RunMemory
from red.caldera.models import PotentialLink
from red.caldera.runner import run_operation
from red.caldera.scope import Scope
from red.caldera.stages import STAGES, allowed_by_stage, check_stages, current_stage, stages_done
from red.llm import FakeLLM

SUMMARY_KEYS = {'mode', 'stage', 'paw', 'max_steps', 'steps_sent', 'failures', 'goal', 'goal_reached', 'stop', 'started',
                'operation_id', 'finished'}
STEP_KEYS = {'step', 'time', 'status', 'ability_id', 'ability_name', 'executor', 'technique', 'target', 'link_id', 'choice'}


def _link(ability, tactic='discovery', host='lab-vm-1'):
    return PotentialLink(paw='p1', ability_id=ability, technique_id='T1', target=host, name=ability,
                         tactic=tactic, command=f'echo {ability}')


def _ids(links):
    return [l.ability_id for l in links]


class FakeClient:
    """Records what would have been sent. get_candidates lists every link, as the server does, even after an ability
    was sent. outputs and statuses are given per ability_id."""

    def __init__(self, links, outputs=None, statuses=None):
        self.links = links
        self.outputs = outputs or {}
        self.statuses = statuses or {}
        self.submitted = []
        self.observed = []
        self._ability_of = {}

    @contextmanager
    def operation(self, name='x'):
        yield 'op-1'

    def is_done(self, op_id):
        return False

    def get_candidates(self, op_id, paw):
        return list(self.links)

    def submit(self, op_id, pl):
        self.submitted.append(pl.ability_id)
        link_id = f'link-{len(self.submitted)}'
        self._ability_of[link_id] = pl.ability_id
        return link_id

    def wait(self, op_id, link_id):
        return self.statuses.get(self._ability_of[link_id], '0')

    def get_link_result(self, op_id, link_id):
        self.observed.append(link_id)
        return self.outputs.get(self._ability_of[link_id], '')


class PickChooser:
    """Always picks index ``pick`` and records the candidates, the stage and the history of each call."""

    def __init__(self, pick=0):
        self.pick = pick
        self.shown = []
        self.stages = []
        self.histories = []

    def choose(self, links, stage=None, tick=0, horizon=1, history=None):
        self.shown.append(_ids(links))
        self.stages.append(stage)
        self.histories.append(history)
        return self.pick, {'source': 'test'}


class OldStyleChooser:
    """The chooser signature from before the history argument. A history keyword would raise TypeError here."""

    def __init__(self):
        self.shown = []

    def choose(self, links, stage=None, tick=0, horizon=1):
        self.shown.append(_ids(links))
        return 0, {'source': 'test'}


class ExtractFactsTests(unittest.TestCase):
    def test_ip_from_ipconfig_without_netmask_loopback_or_zero(self):
        text = ('   IPv4 Address. . . . . . . . . . . : 192.168.56.10(Preferred)\n'
                '   Subnet Mask . . . . . . . . . . . : 255.255.255.0\n'
                '   Default Gateway . . . . . . . . . : 192.168.56.1\n'
                'loopback 127.0.0.1, unspecified 0.0.0.0, not an address 999.1.1.1\n')
        ips = {v for k, v in extract_facts(text) if k == 'ip'}
        self.assertEqual(ips, {'192.168.56.10', '192.168.56.1'})

    def test_ip_is_found_in_the_full_text_however_long_it_is(self):
        self.assertIn(('ip', '10.1.2.3'), extract_facts('x' * 400 + ' 10.1.2.3 end'))

    def test_host_from_ipconfig_systeminfo_unc_and_environment(self):
        text = ('   Host Name . . . . . . . . . . . . : desktop-xyz\n'
                'Host Name:          DESKTOP-XYZ\n'
                '\\\\FILESRV01\\share\n'
                'COMPUTERNAME=WS-02\n')
        hosts = {v for k, v in extract_facts(text) if k == 'host'}
        self.assertEqual(hosts, {'DESKTOP-XYZ', 'FILESRV01', 'WS-02'})

    def test_host_name_does_not_run_into_the_next_line(self):
        self.assertEqual(extract_facts('Host Name . . . :\nDESKTOP-XYZ\n'), [])

    def test_host_name_longer_than_15_characters_is_refused(self):
        self.assertEqual(extract_facts('Host Name: ABCDEFGHIJKLMNOP\n'), [])

    def test_ports_from_netstat_and_nmap_open_only(self):
        text = ('  TCP    0.0.0.0:445    0.0.0.0:0    LISTENING    4\n'
                '  TCP    192.168.56.10:49667    10.0.0.5:51234    ESTABLISHED    812\n'
                '  UDP    0.0.0.0:5353   *:*                                    1234\n'
                '445/tcp open  microsoft-ds\n'
                '139/tcp closed netbios-ssn\n')
        ports = {v for k, v in extract_facts(text) if k == 'port'}
        self.assertEqual(ports, {'445', '5353', '49667'})  # 139 is closed; 51234 is a remote port, not counted

    def test_user_from_whoami_and_environment(self):
        text = 'DESKTOP-XYZ\\alice\nNT AUTHORITY\\SYSTEM\nUSERNAME=Bob\n'
        users = {v for k, v in extract_facts(text) if k == 'user'}
        self.assertEqual(users, {'alice', 'system', 'bob'})

    def test_same_output_gives_the_same_sorted_list_without_duplicates(self):
        text = 'Host Name: B-HOST\n192.168.56.10 ... 192.168.56.10\n445/tcp open smb\n'
        first = extract_facts(text)
        self.assertEqual(first, sorted(first))
        self.assertEqual(first, extract_facts(text))
        self.assertEqual(len(first), len(set(first)))

    def test_empty_and_missing_text_give_nothing(self):
        self.assertEqual(extract_facts(''), [])
        self.assertEqual(extract_facts(None), [])
        self.assertEqual(extract_facts('nothing useful: 0.0.0.0 127.0.0.1 255.255.255.255'), [])


class GoalTests(unittest.TestCase):
    def test_host_lain_counts_different_hosts(self):
        facts = {('host', 'DESKTOP-A'), ('host', 'FILESRV01')}
        self.assertTrue(goal_reached(facts, ('host_lain', 2)))
        self.assertFalse(goal_reached(facts, ('host_lain', 3)))

    def test_the_same_host_in_another_case_counts_once(self):
        facts = [('host', 'DESKTOP-A'), ('host', 'desktop-a')]
        self.assertFalse(goal_reached(facts, ('host_lain', 2)))
        self.assertTrue(goal_reached(facts, ('host_lain', 1)))

    def test_other_kinds_are_counted_separately(self):
        facts = {('ip', '192.168.56.10'), ('port', '445'), ('user', 'alice')}
        self.assertTrue(goal_reached(facts, ('ip', 1)))
        self.assertTrue(goal_reached(facts, ('user', 1)))
        self.assertFalse(goal_reached(facts, ('host_lain', 1)))  # a port or a user is not a host

    def test_unknown_or_badly_counted_goal_is_refused(self):
        bad_goals = (('host_lan', 1), ('host_lain', 0), ('host_lain', -1), ('host_lain', '2'),
                     ('host_lain', True), ('host_lain',), 'host_lain', None)
        for bad in bad_goals:
            with self.subTest(goal=bad):
                with self.assertRaises(ValueError):
                    check_need(bad)
                with self.assertRaises(ValueError):
                    goal_reached(set(), bad)


class _LoopCase(unittest.TestCase):
    """Shared set-up and helper for the loop tests."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _run(self, client, chooser, name='run', max_steps=3, **kwargs):
        scope = Scope(paw='p1', lab_hosts=('lab-vm-1',), allowed_abilities=frozenset({'A', 'B', 'C'}),
                      max_steps=max_steps, stop_file=self.root / 'STOP')
        run_dir = self.root / name
        summary = run_operation(client, scope, chooser, 'llm', 'recon', run_dir, log=lambda *_: None, **kwargs)
        steps_file = run_dir / 'steps.jsonl'
        steps = []
        if steps_file.exists():
            steps = [json.loads(line) for line in steps_file.read_text(encoding='utf-8').splitlines()]
        return summary, steps


class GoalAndStageLoopTests(_LoopCase):
    # goal
    def test_goal_stops_the_run_after_the_step_that_reaches_it(self):
        client = FakeClient([_link('A'), _link('B'), _link('C')],
                            outputs={'A': 'Host Name: HOST-1', 'B': 'Host Name: HOST-2'})
        summary, steps = self._run(client, PickChooser(0), memory=RunMemory(), max_steps=3, goal=('host_lain', 2))
        self.assertEqual(client.submitted, ['A', 'B'])
        self.assertEqual(summary['stop'], 'goal_reached')
        self.assertEqual(summary['steps_sent'], 2)
        self.assertEqual(len(steps), 2)

    def test_goal_not_reached_runs_to_the_budget(self):
        client = FakeClient([_link('A'), _link('B')], outputs={'A': 'Host Name: HOST-1'})
        summary, _ = self._run(client, PickChooser(0), memory=RunMemory(), max_steps=2, goal=('host_lain', 2))
        self.assertEqual(summary['stop'], 'budget_used')
        self.assertEqual(client.submitted, ['A', 'B'])

    def test_failed_link_does_not_stop_the_run_and_its_facts_are_kept(self):
        # Rule since 2026-11-19: one failed link is not the end of the run. Its facts are kept, but the goal is only
        # checked after a link that succeeded.
        mem = RunMemory()
        client = FakeClient([_link('A'), _link('B')],
                            outputs={'A': 'Host Name: HOST-1', 'B': 'Host Name: HOST-2'}, statuses={'B': '1'})
        summary, _ = self._run(client, PickChooser(0), memory=mem, max_steps=3, goal=('host_lain', 2))
        self.assertEqual(client.submitted, ['A', 'B'])
        self.assertEqual(summary['stop'], 'no_allowed_candidate')   # B was the last candidate
        self.assertEqual(summary['failures'], 1)
        self.assertFalse(summary['goal_reached'])
        self.assertIn(('host', 'HOST-2'), mem.facts)

    def test_facts_come_from_the_full_output_not_the_300_character_summary(self):
        mem = RunMemory()
        client = FakeClient([_link('A')], outputs={'A': 'x' * 400 + ' 10.1.2.3'})
        self._run(client, PickChooser(0), memory=mem, max_steps=1)
        self.assertIn(('ip', '10.1.2.3'), mem.facts)

    def test_goal_and_stages_need_memory_and_nothing_is_sent(self):
        for kwargs in ({'goal': ('host_lain', 1)}, {'stages': STAGES}):
            with self.subTest(kwargs=sorted(kwargs)):
                client = FakeClient([_link('A')])
                with self.assertRaises(ValueError):
                    self._run(client, PickChooser(0), name='no_memory', **kwargs)
                self.assertEqual(client.submitted, [])
                self.assertFalse((self.root / 'no_memory').exists())

    def test_bad_goal_is_refused_before_any_action(self):
        client = FakeClient([_link('A')])
        with self.assertRaises(ValueError):
            self._run(client, PickChooser(0), name='bad_goal', memory=RunMemory(), goal=('host_lan', 1))
        self.assertEqual(client.submitted, [])
        self.assertFalse((self.root / 'bad_goal').exists())

    # stage gate: data and functions
    def test_at_the_start_only_the_first_stage_is_open(self):
        links = [_link('A', 'discovery'), _link('B', 'execution')]
        self.assertEqual(_ids(allowed_by_stage(links, set())), ['A'])

    def test_next_stage_opens_when_the_previous_one_is_done(self):
        links = [_link('A', 'discovery'), _link('B', 'lateral-movement'), _link('C', 'impact')]
        self.assertEqual(_ids(allowed_by_stage(links, {'recon'})), ['A', 'B'])
        self.assertEqual(_ids(allowed_by_stage(links, {'recon', 'advance'})), ['A', 'B', 'C'])

    def test_a_later_stage_does_not_open_before_an_earlier_one_is_done(self):
        links = [_link('A', 'discovery'), _link('B', 'execution')]
        self.assertEqual(_ids(allowed_by_stage(links, {'advance'})), ['A'])

    def test_a_tactic_in_no_stage_is_refused(self):
        links = [_link('A', 'persistence'), _link('B', '')]
        for done in (set(), {'recon', 'advance', 'impact'}):
            with self.subTest(done=sorted(done)):
                self.assertEqual(allowed_by_stage(links, done), [])

    def test_stage_done_comes_from_the_facts(self):
        self.assertEqual(stages_done(STAGES, []), {'impact'})
        self.assertEqual(stages_done(STAGES, [('ip', '192.168.56.10')]), {'impact'})
        self.assertEqual(stages_done(STAGES, [('ip', '192.168.56.10'), ('host', 'DESKTOP-XYZ')]),
                         {'recon', 'impact'})
        self.assertEqual(stages_done(STAGES, [('ip', '192.168.56.10'), ('host', 'DESKTOP-XYZ'), ('user', 'alice')]),
                         {'recon', 'advance', 'impact'})

    def test_current_stage_is_the_first_one_not_done(self):
        self.assertEqual(current_stage(STAGES, set()), 'recon')
        self.assertEqual(current_stage(STAGES, {'recon'}), 'advance')
        self.assertEqual(current_stage(STAGES, {'recon', 'advance', 'impact'}), 'impact')

    def test_stage_data_is_checked(self):
        bad_stage_lists = [
            (),
            [],
            [{'name': 'recon', 'tactics': ('discovery',)}],
            [{'name': 'recon', 'tactics': (), 'done_when': ()}],
            [{'name': '', 'tactics': ('discovery',), 'done_when': ()}],
            [{'name': 'recon', 'tactics': ('discovery',), 'done_when': ()},
             {'name': 'recon', 'tactics': ('impact',), 'done_when': ()}],
            [{'name': 'recon', 'tactics': ('discovery',), 'done_when': ()},
             {'name': 'advance', 'tactics': ('discovery',), 'done_when': ()}],
            [{'name': 'recon', 'tactics': ('discovery',), 'done_when': (('ip', 0),)}],
            [{'name': 'recon', 'tactics': ('discovery',), 'done_when': (('ipp', 1),)}],
        ]
        for stages in bad_stage_lists:
            with self.subTest(stages=stages):
                with self.assertRaises(ValueError):
                    check_stages(stages)
        check_stages(STAGES)  # the real data is accepted

    # stage gate: in the loop
    def test_run_hides_the_next_stage_until_its_facts_are_found(self):
        client = FakeClient([_link('A', 'discovery'), _link('B', 'execution')], outputs={'A': ''})
        chooser = PickChooser(0)
        summary, _ = self._run(client, chooser, memory=RunMemory(), max_steps=3, stages=STAGES)
        self.assertEqual(client.submitted, ['A'])
        self.assertEqual(chooser.shown, [['A']])  # B (advance) is not offered before recon facts are known
        self.assertEqual(chooser.stages, ['recon'])
        self.assertEqual(summary['stop'], 'stage_blocked')

    def test_next_stage_opens_after_the_facts_are_found(self):
        client = FakeClient([_link('A', 'discovery'), _link('B', 'execution')],
                            outputs={'A': 'IPv4 Address . . . : 192.168.56.10\nHost Name . . . : DESKTOP-XYZ'})
        chooser = PickChooser(0)
        summary, _ = self._run(client, chooser, memory=RunMemory(), max_steps=2, stages=STAGES)
        self.assertEqual(client.submitted, ['A', 'B'])
        self.assertEqual(chooser.shown, [['A'], ['B']])
        self.assertEqual(chooser.stages, ['recon', 'advance'])  # the chooser is told the open stage
        self.assertEqual(summary['stop'], 'budget_used')

    def test_stage_gate_with_the_llm_chooser_and_fake_llm(self):
        prompts = []

        def responder(prompt):
            prompts.append(prompt)
            return 'ACTION: 0'

        client = FakeClient([_link('A', 'discovery'), _link('B', 'execution')],
                            outputs={'A': 'Host Name: HOST-1\nlink to 10.0.0.5'})
        summary, _ = self._run(client, LlmChooser(FakeLLM(responder=responder)), memory=RunMemory(),
                               max_steps=2, stages=STAGES)
        self.assertEqual(client.submitted, ['A', 'B'])
        self.assertIn('Current stage: recon', prompts[0])
        self.assertIn('Current stage: advance', prompts[1])
        self.assertIn('0. B', prompts[1])
        self.assertEqual(summary['stop'], 'budget_used')


class DefaultLoopTests(_LoopCase):
    """The loop without goal and stages behaves as before."""

    def test_without_goal_or_stages_facts_are_kept_but_nothing_stops_or_changes(self):
        client = FakeClient([_link('A'), _link('B')],
                            outputs={'A': 'Host Name: HOST-1\nlink 10.0.0.5', 'B': 'Host Name: HOST-2'})
        chooser = PickChooser(0)
        mem = RunMemory()
        summary, steps = self._run(client, chooser, memory=mem, max_steps=2)
        self.assertEqual(client.submitted, ['A', 'B'])
        self.assertEqual(summary['stop'], 'budget_used')  # two hosts are known, but there is no goal
        self.assertEqual(set(summary), SUMMARY_KEYS)
        self.assertEqual(set(steps[0]), STEP_KEYS | {'output'})
        self.assertEqual(chooser.stages, ['recon', 'recon'])  # the stage argument is the one given to run_operation
        self.assertEqual(mem.facts, {('host', 'HOST-1'), ('host', 'HOST-2'), ('ip', '10.0.0.5')})

    def test_without_memory_the_chooser_and_summary_are_as_before(self):
        client = FakeClient([_link('A'), _link('B')])
        summary, steps = self._run(client, OldStyleChooser(), max_steps=2)
        self.assertEqual(client.submitted, ['A', 'A'])
        self.assertEqual(client.observed, [])
        self.assertEqual(summary['stop'], 'budget_used')
        self.assertEqual(set(summary), SUMMARY_KEYS)
        self.assertEqual(set(steps[0]), STEP_KEYS)

    def test_run_memory_keeps_each_fact_once(self):
        mem = RunMemory()
        mem.add_facts(extract_facts('Host Name: HOST-1'))
        mem.add_facts(extract_facts('Host Name: host-1'))
        self.assertEqual(mem.facts, {('host', 'HOST-1')})


if __name__ == '__main__':
    unittest.main()
