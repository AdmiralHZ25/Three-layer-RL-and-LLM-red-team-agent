"""Offline tests for the Caldera menu helpers, the step logger, and the automatic run flow.

The client is a fake and the LLM is a fake: nothing reaches a server or Ollama.

Run from the RED_AGENT folder:  python -m unittest tests.test_caldera_manual -v
"""
import contextlib
import io
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from red import config
from red.caldera.manual import (CONFIRM_WORD, ManualSession, abilities_for_platform, describe_status,
                                parse_choice_numbers)
from red.caldera.models import PotentialLink
from red.caldera.scope import Scope


def _link(ability_id, target='lab-win'):
    return PotentialLink(paw='p1', ability_id=ability_id, name=f'name {ability_id}', technique_id='T1046',
                         tactic='discovery', target=target, command='echo x', executor='sh')


class FakeClient:
    def __init__(self, links=(), status='0'):
        self._links = list(links)
        self.status = status
        self.submitted = []

    def get_candidates(self, op_id, paw):
        return list(self._links)

    def submit(self, op_id, pl):
        self.submitted.append(pl.ability_id)
        return 'link-1'

    def wait(self, op_id, link_id):
        return self.status


class ManualSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.log = self.dir / 'steps.jsonl'
        self.stop = self.dir / 'STOP'

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, links, status='0', max_steps=3, allowed=('a1', 'a2'), hosts=('lab-win',)):
        scope = Scope(paw='p1', lab_hosts=hosts, allowed_abilities=frozenset(allowed),
                      max_steps=max_steps, stop_file=self.stop)
        client = FakeClient(links, status)
        session = ManualSession(client, scope, self.log)
        session.op_id = 'op-1'
        session.refresh()
        return session, client

    def log_lines(self):
        return [json.loads(line) for line in self.log.read_text(encoding='utf-8').splitlines()]

    def test_nothing_is_sent_without_the_typed_confirmation(self):
        session, client = self.session([_link('a1')])
        for confirm in ('', 'ya', 'no'):
            with self.subTest(confirm=confirm):
                label, _ = session.act(0, confirm)
                self.assertEqual(label, 'BATAL')
        self.assertEqual(client.submitted, [])
        self.assertEqual(session.sent, 0)

    def test_confirmed_step_is_sent_once_and_logged(self):
        session, client = self.session([_link('a1')])
        label, _ = session.act(0, CONFIRM_WORD)
        self.assertEqual(label, 'BERHASIL')
        self.assertEqual(client.submitted, ['a1'])
        self.assertEqual(session.sent, 1)
        lines = self.log_lines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]['result'], 'BERHASIL')
        self.assertEqual(lines[0]['link_status'], '0')
        self.assertIn('approval_id', lines[0])

    def test_ability_outside_the_allow_list_is_refused(self):
        session, client = self.session([_link('a1')], allowed=('a2',))
        label, _ = session.act(0, CONFIRM_WORD)
        self.assertEqual(label, 'DITOLAK SCOPE')
        self.assertEqual(client.submitted, [])

    def test_host_outside_the_lab_is_refused(self):
        session, client = self.session([_link('a1', target='other-host')], hosts=('lab-win',))
        label, _ = session.act(0, CONFIRM_WORD)
        self.assertEqual(label, 'DITOLAK SCOPE')
        self.assertEqual(client.submitted, [])

    def test_stop_file_blocks_every_action(self):
        session, client = self.session([_link('a1')])
        self.stop.write_text('stop', encoding='utf-8')
        label, _ = session.act(0, CONFIRM_WORD)
        self.assertEqual(label, 'DIHENTIKAN')
        self.assertEqual(client.submitted, [])

    def test_step_limit_per_session_is_enforced(self):
        session, client = self.session([_link('a1')], max_steps=1)
        self.assertEqual(session.act(0, CONFIRM_WORD)[0], 'BERHASIL')
        self.assertEqual(session.act(0, CONFIRM_WORD)[0], 'BATAS TERCAPAI')
        self.assertEqual(client.submitted, ['a1'])

    def test_wait_timeout_is_reported_not_hidden(self):
        session, _ = self.session([_link('a1')], status='timeout')
        label, text = session.act(0, CONFIRM_WORD)
        self.assertEqual(label, 'WAKTU HABIS')
        self.assertIn('GUI Caldera', text)

    def test_bad_index_is_refused(self):
        session, client = self.session([_link('a1')])
        with self.assertRaises(ValueError):
            session.act(5, CONFIRM_WORD)
        self.assertEqual(client.submitted, [])

    def test_status_labels(self):
        self.assertEqual(describe_status('0')[0], 'BERHASIL')
        self.assertEqual(describe_status('1')[0], 'GAGAL')
        self.assertEqual(describe_status('124')[0], 'WAKTU HABIS')
        self.assertEqual(describe_status('untrusted')[0], 'MENUNGGU')
        self.assertEqual(describe_status('999')[0], 'TIDAK DIKENAL')


class ChoiceHelpers(unittest.TestCase):
    def test_abilities_for_platform_keeps_only_matching_executors(self):
        abilities = [
            {'ability_id': 'a1', 'name': 'Find files', 'technique_id': 'T1083', 'tactic': 'discovery',
             'executors': [{'name': 'psh', 'platform': 'windows'}, {'name': 'sh', 'platform': 'linux'}]},
            {'ability_id': 'a2', 'name': 'Linux only', 'technique_id': 'T1046', 'tactic': 'discovery',
             'executors': [{'name': 'sh', 'platform': 'linux'}]},
        ]
        rows = abilities_for_platform(abilities, 'windows')
        self.assertEqual([r['ability_id'] for r in rows], ['a1'])
        self.assertEqual(rows[0]['executors'], 'psh')

    def test_parse_choice_numbers(self):
        self.assertEqual(parse_choice_numbers('1, 3', 3), [0, 2])
        self.assertEqual(parse_choice_numbers('2,2', 3), [1])
        for bad in ('', '0', '4', 'x', '1,9'):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_choice_numbers(bad, 3)


class FakeServer:
    """Stands in for CalderaClient for the whole menu flow. Nothing reaches a real server."""

    def __init__(self):
        self.submitted = []

    def list_agents(self):
        return [{'paw': 'p1', 'host': 'DESKTOP-TEST', 'platform': 'windows', 'status': 'alive', 'trusted': True}]

    def list_abilities(self):
        return [
            {'ability_id': 'a1', 'name': 'Find files', 'technique_id': 'T1083', 'tactic': 'discovery',
             'executors': [{'name': 'psh', 'platform': 'windows'}]},
            {'ability_id': 'a2', 'name': 'Linux only', 'technique_id': 'T1046', 'tactic': 'discovery',
             'executors': [{'name': 'sh', 'platform': 'linux'}]},
        ]

    def check_paw(self, paw):
        return paw

    @contextmanager
    def operation(self, name):
        yield 'op-1'

    def is_done(self, op_id):
        return False

    def get_candidates(self, op_id, paw):
        return [PotentialLink(paw='p1', ability_id='a1', name='Find files', technique_id='T1083',
                              tactic='discovery', target='DESKTOP-TEST', command='dir', executor='psh')]

    def submit(self, op_id, pl):
        self.submitted.append(pl.ability_id)
        return 'link-1'

    def wait(self, op_id, link_id):
        return '0'


class AutoFlow(unittest.TestCase):
    def test_auto_run_chooses_by_itself_end_to_end(self):
        import run_menu
        from red.llm import FakeLLM
        fake = FakeServer()
        # agent number, search (Enter = all), action number, mode, step budget, start
        answers = ['1', '', '1', 'llm', '1', 'JALANKAN']
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.multiple(config, CALDERA_URL='http://127.0.0.1:1', CALDERA_API_KEY='x',
                                    CALDERA_PAW='', CALDERA_LAB_HOSTS=(), CALDERA_ALLOWED_ABILITY_IDS=(),
                                    CALDERA_ADVERSARY_ID='', CALDERA_PLANNER_ID='', CALDERA_SOURCE_ID=''), \
                mock.patch.object(run_menu, 'ROOT', Path(tmp)), \
                mock.patch.object(run_menu, 'check_ollama', return_value=True), \
                mock.patch('red.llm.make_llm', return_value=FakeLLM()), \
                mock.patch('red.caldera.client.CalderaClient', return_value=fake), \
                mock.patch('builtins.input', side_effect=answers), \
                contextlib.redirect_stdout(io.StringIO()):
            run_menu.caldera_run()
            logs = list(Path(tmp, 'runs', 'caldera').glob('*/steps.jsonl'))
            self.assertEqual(len(logs), 1)
            record = json.loads(logs[0].read_text(encoding='utf-8').splitlines()[0])
            self.assertEqual(record['choice']['source'], 'LLM')
        self.assertEqual(fake.submitted, ['a1'])


class MenuText(unittest.TestCase):
    def test_menu_offers_automatic_run_and_no_manual_option(self):
        import run_menu
        self.assertIn('g  Serang otomatis', run_menu.MENU)
        self.assertIn('g', run_menu.ACTIONS)
        self.assertNotIn('m', run_menu.ACTIONS)


if __name__ == '__main__':
    unittest.main()
