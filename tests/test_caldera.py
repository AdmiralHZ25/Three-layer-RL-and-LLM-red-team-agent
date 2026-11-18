"""Offline tests for the Caldera bridge. No network: every request is faked or must never happen.

Run from the RED_AGENT folder:  python -m unittest tests.test_caldera -v
"""
import http.server
import json
import threading
import unittest
from unittest import mock

from red import config
from red.caldera.approval import ApprovalGate
from red.caldera.client import CalderaClient, CalderaError
from red.caldera.executor import ApprovalRequired, CalderaExecutor
from red.caldera.models import PotentialLink
from red.llm import FakeLLM


def _links():
    return [PotentialLink(paw='p1', ability_id='a1', name='Enumerate', technique_id='T1046',
                          tactic='discovery', target='h1', command='echo a1'),
            PotentialLink(paw='p1', ability_id='a2', name='Scan', technique_id='T1049',
                          tactic='discovery', target='h2', command='echo a2')]


class FakeClient:
    """Stands in for CalderaClient: records what would have been sent to a target."""

    def __init__(self):
        self.submitted = []

    def get_candidates(self, op_id, paw):
        return _links()

    def submit(self, op_id, pl):
        self.submitted.append(pl.ability_id)
        return 'link-1'

    def wait(self, op_id, link_id):
        return '0'


class ClientGuards(unittest.TestCase):
    def test_no_api_key_is_refused_before_any_request(self):
        with mock.patch.object(config, 'CALDERA_API_KEY', ''), \
                mock.patch('urllib.request.urlopen', side_effect=AssertionError('network used')):
            with self.assertRaises(CalderaError):
                CalderaClient(base_url='http://127.0.0.1:1', api_key=None)._req('GET', '/agents')

    def test_empty_allowlist_gives_no_candidates_and_no_request(self):
        with mock.patch.object(config, 'CALDERA_ALLOWED_ABILITY_IDS', ()), \
                mock.patch.object(CalderaClient, '_req', side_effect=AssertionError('request sent')):
            self.assertEqual(CalderaClient(api_key='x').get_candidates('op', 'p1'), [])

    def test_submit_refuses_an_ability_outside_the_allowlist(self):
        pl = PotentialLink(paw='p1', ability_id='a2', command='echo a2')
        with mock.patch.object(config, 'CALDERA_ALLOWED_ABILITY_IDS', ('a1',)), \
                mock.patch.object(CalderaClient, 'check_paw', return_value='p1'), \
                mock.patch.object(CalderaClient, '_req', side_effect=AssertionError('sent')):
            with self.assertRaises(CalderaError):
                CalderaClient(api_key='x').submit('op', pl)

    def test_check_paw_needs_the_paw_and_a_lab_host(self):
        with mock.patch.object(config, 'CALDERA_PAW', ''):
            with self.assertRaises(CalderaError):
                CalderaClient(api_key='x').check_paw('p1')
        with mock.patch.object(config, 'CALDERA_PAW', 'p1'), \
                mock.patch.object(config, 'CALDERA_LAB_HOSTS', ()):
            with self.assertRaises(CalderaError):
                CalderaClient(api_key='x').check_paw('p1')

    def test_check_paw_accepts_only_a_lab_host(self):
        agents = [{'paw': 'p1', 'host': 'lab-vm-1', 'host_ip_addrs': ['192.168.56.30']}]
        with mock.patch.object(config, 'CALDERA_PAW', 'p1'), \
                mock.patch.object(config, 'CALDERA_LAB_HOSTS', ('lab-vm-1',)), \
                mock.patch.object(CalderaClient, 'list_agents', return_value=agents):
            self.assertEqual(CalderaClient(api_key='x').check_paw('p1'), 'p1')
        with mock.patch.object(config, 'CALDERA_PAW', 'p1'), \
                mock.patch.object(config, 'CALDERA_LAB_HOSTS', ('other-host',)), \
                mock.patch.object(CalderaClient, 'list_agents', return_value=agents):
            with self.assertRaises(CalderaError):
                CalderaClient(api_key='x').check_paw('p1')


class _FakeCaldera(http.server.BaseHTTPRequestHandler):
    """A local stand-in for the Caldera REST API on 127.0.0.1 (not the lab)."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != '/api/v2/agents':
            self.send_response(404)
            self.end_headers()
            return
        if self.headers.get('KEY') != 'test-key':
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'unauthorized')
            return
        body = json.dumps([{'paw': 'p1', 'host': 'lab-vm-1', 'platform': 'linux'}]).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)


class LocalHttp(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _FakeCaldera)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_list_agents_with_the_right_key(self):
        agents = CalderaClient(base_url=self.url, api_key='test-key').list_agents()
        self.assertEqual(agents[0]['paw'], 'p1')

    def test_wrong_key_is_reported_as_an_error(self):
        with self.assertRaises(CalderaError):
            CalderaClient(base_url=self.url, api_key='wrong').list_agents()


class ExecutorGate(unittest.TestCase):
    def test_plan_only_selects_and_never_submits(self):
        fake = FakeClient()
        rec = CalderaExecutor(fake, FakeLLM(responder=lambda p: 'ACTION: 1')).step('op', 'p1', stage='recon')
        self.assertEqual((rec.ability_id, rec.source), ('a2', 'LLM'))
        self.assertEqual(fake.submitted, [])

    def test_llm_failure_falls_back_to_the_first_candidate(self):
        rec = CalderaExecutor(FakeClient(), FakeLLM(ok=False, error='down')).step('op', 'p1')
        self.assertEqual((rec.ability_id, rec.source), ('a1', 'fallback'))

    def test_execute_without_any_gate_is_refused_and_sends_nothing(self):
        fake = FakeClient()
        with self.assertRaises(ApprovalRequired):
            CalderaExecutor(fake, FakeLLM(responder=lambda p: 'ACTION: 0')).step('op', 'p1', execute=True)
        self.assertEqual(fake.submitted, [])

    def test_execute_needs_a_human_approval_and_each_token_works_once(self):
        fake, gate = FakeClient(), ApprovalGate()
        ex = CalderaExecutor(fake, FakeLLM(responder=lambda p: 'ACTION: 1'), approval=gate)
        with self.assertRaises(ApprovalRequired) as cm:
            ex.step('op', 'p1', execute=True)
        rid = cm.exception.approval_id
        self.assertEqual(fake.submitted, [])               # still nothing sent
        gate.approve(rid)
        ex.step('op', 'p1', execute=True, approval_id=rid)
        self.assertEqual(fake.submitted, ['a2'])
        with self.assertRaises(ApprovalRequired):          # the same token cannot be used again
            ex.step('op', 'p1', execute=True, approval_id=rid)
        self.assertEqual(fake.submitted, ['a2'])

    def test_a_different_choice_from_the_approved_one_is_refused(self):
        fake, gate = FakeClient(), ApprovalGate()
        answers = iter(['ACTION: 1', 'ACTION: 0'])
        ex = CalderaExecutor(fake, FakeLLM(responder=lambda p: next(answers)), approval=gate)
        with self.assertRaises(ApprovalRequired) as cm:
            ex.step('op', 'p1', execute=True)
        gate.approve(cm.exception.approval_id)
        with self.assertRaises(ApprovalRequired):          # now the LLM picks a1, not the approved a2
            ex.step('op', 'p1', execute=True, approval_id=cm.exception.approval_id)
        self.assertEqual(fake.submitted, [])


if __name__ == '__main__':
    unittest.main()
