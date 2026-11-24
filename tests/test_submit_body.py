"""Offline tests: the link body sent to Caldera must include the executor platform.

Without "platform", Caldera fails with HTTP 500 (platform.lower() on None). No network: _req is patched.

Run from the RED_AGENT folder:  python -m unittest tests.test_submit_body -v
"""
import unittest
from unittest import mock

from red import config
from red.caldera.client import CalderaClient, CalderaError
from red.caldera.models import PotentialLink


class SubmitBody(unittest.TestCase):
    def test_submit_sends_the_executor_platform(self):
        pl = PotentialLink(paw='p1', ability_id='a1', executor='psh', command='dir', platform='windows')
        with mock.patch.object(config, 'CALDERA_ALLOWED_ABILITY_IDS', ('a1',)), \
                mock.patch.object(CalderaClient, 'check_paw', return_value='p1'), \
                mock.patch.object(CalderaClient, '_req', return_value={'id': 'link-1'}) as req:
            link_id = CalderaClient(api_key='x').submit('op-1', pl)
        self.assertEqual(link_id, 'link-1')
        method, path, body = req.call_args.args
        self.assertEqual((method, path), ('POST', '/operations/op-1/potential-links'))
        self.assertEqual(body['executor'], {'name': 'psh', 'command': 'dir', 'platform': 'windows'})

    def test_submit_refuses_a_link_without_platform(self):
        pl = PotentialLink(paw='p1', ability_id='a1', executor='psh', command='dir')
        with mock.patch.object(config, 'CALDERA_ALLOWED_ABILITY_IDS', ('a1',)), \
                mock.patch.object(CalderaClient, 'check_paw', return_value='p1'), \
                mock.patch.object(CalderaClient, '_req', side_effect=AssertionError('sent')):
            with self.assertRaises(CalderaError):
                CalderaClient(api_key='x').submit('op-1', pl)

    def test_candidates_keep_the_platform_from_caldera(self):
        links = [{'paw': 'p1', 'host': 'h1',
                  'ability': {'ability_id': 'a1', 'name': 'n', 'tactic': 't', 'technique_id': 'T1'},
                  'executor': {'name': 'psh', 'platform': 'windows', 'command': 'dir'}}]
        with mock.patch.object(config, 'CALDERA_ALLOWED_ABILITY_IDS', ('a1',)), \
                mock.patch.object(CalderaClient, '_req', return_value=links):
            out = CalderaClient(api_key='x').get_candidates('op-1', 'p1')
        self.assertEqual(out[0].platform, 'windows')


class ServerNoAnswer(unittest.TestCase):
    def test_a_timeout_becomes_a_caldera_error_not_a_crash(self):
        with mock.patch('urllib.request.urlopen', side_effect=TimeoutError('timed out')):
            with self.assertRaises(CalderaError):
                CalderaClient(base_url='http://127.0.0.1:1', api_key='x')._req('POST', '/operations/op-1/potential-links', {})


class OperationBody(unittest.TestCase):
    def test_create_operation_keeps_the_operation_open(self):
        with mock.patch.multiple(config, CALDERA_ADVERSARY_ID='', CALDERA_PLANNER_ID='', CALDERA_SOURCE_ID=''), \
                mock.patch.object(CalderaClient, '_req', return_value={'id': 'op-1'}) as req:
            CalderaClient(api_key='x').create_operation()
        method, path, body = req.call_args.args
        self.assertEqual((method, path), ('POST', '/operations'))
        self.assertIs(body['auto_close'], False)


if __name__ == '__main__':
    unittest.main()
