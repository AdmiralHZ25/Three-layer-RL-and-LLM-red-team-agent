"""Offline tests for the agentic Caldera run (menu g): the planner's reply parsing, the short list, the filters, and the loop.

No network and no LLM server: the LLM is a FakeLLM with a responder, and the Caldera client is a fake.
Run from the RED_AGENT folder:  python -B -m unittest tests.test_agentic_planner -v
"""
import base64
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from red.caldera.agent_loop import offered_links, run_agentic
from red.caldera.models import PotentialLink
from red.caldera.planner import (DATA_LABEL, SHORTLIST_FALLBACK, SHORTLIST_SIZE, build_choice_prompt, build_plan_prompt, phrases_shown, required_phrases,
                                 evidence_in_output, notes_text, objective_terms, parse_batch, parse_done, parse_plan,
                                 shortlist)
from red.llm import FakeLLM

DEFAULT_PLAN = 'RENCANA:\n1. Cari user lokal\n2. Tulis hasilnya ke file'
NEUTRAL_PLAN = 'RENCANA:\n1. Langkah satu\n2. Langkah dua'   # its words match no ability name in these tests


def _link(ability, name=None, tactic='discovery', host='lab-vm-1', command=None, technique='T1087'):
    return PotentialLink(paw='p1', ability_id=ability, name=name or ability, tactic=tactic, technique_id=technique,
                         target=host, command=command if command is not None else f'echo {ability}', executor='psh')


class FakeClient:
    def __init__(self, links, outputs=None, statuses=None):
        self.links = links
        self.outputs = outputs or {}
        self.statuses = statuses or {}
        self.submitted = []
        self.allowed_seen = []
        self._ability_of = {}

    @contextmanager
    def operation(self, name='x'):
        yield 'op-1'

    def is_done(self, op_id):
        return False

    def get_all_candidates(self, op_id, paw):
        return list(self.links)

    def submit(self, op_id, pl, allowed=None):
        self.allowed_seen.append(allowed)
        if allowed is not None and pl.ability_id not in allowed:
            raise AssertionError(f'{pl.ability_id} was sent but is not in the allowed list')
        self.submitted.append(pl.ability_id)
        link_id = f'link-{len(self.submitted)}'
        self._ability_of[link_id] = pl.ability_id
        return link_id

    def wait(self, op_id, link_id):
        return self.statuses.get(self._ability_of[link_id], '0')

    def get_link_result(self, op_id, link_id):
        return self.outputs.get(self._ability_of[link_id], '')


def responder(choice='ACTION: 0', done='DONE: no\nEVIDENCE:', plan=DEFAULT_PLAN, seen=None):
    """Answers each kind of prompt. ``seen`` collects every prompt, so a test can check what the planner was shown."""
    def respond(prompt):
        if seen is not None:
            seen.append(prompt)
        if 'Buat rencana singkat' in prompt:
            return plan
        if 'Pilih sampai' in prompt:
            return choice
        return done
    return respond


class PlannerParsingTests(unittest.TestCase):
    def test_one_or_several_numbers_are_read(self):
        self.assertEqual(parse_batch('ACTION: 2', 5), ('indices', [2]))
        self.assertEqual(parse_batch('ACTION: 0, 3', 5), ('indices', [0, 3]))
        self.assertEqual(parse_batch('ACTION: 1 2 4', 5), ('indices', [1, 2, 4]))
        self.assertEqual(parse_batch('<think>ACTION: 9</think>\nACTION: 1', 5), ('indices', [1]))

    def test_unclear_or_unsafe_replies_are_refused(self):
        refused = ('none', None)
        self.assertEqual(parse_batch('ACTION: 0, 1, 2, 3', 5), refused)      # more than three
        self.assertEqual(parse_batch('ACTION: 0, 0', 5), refused)            # the same number twice
        self.assertEqual(parse_batch('ACTION: 9', 5), refused)               # out of range
        self.assertEqual(parse_batch('ACTION: 1\nACTION: 2', 5), refused)    # two ACTION lines
        self.assertEqual(parse_batch('ACTION: 1\nACTION: STOP', 5), refused)
        self.assertEqual(parse_batch('saya pilih yang kedua', 5), refused)
        self.assertEqual(parse_batch('', 5), refused)
        self.assertEqual(parse_batch('ACTION: 0', 0), refused)

    def test_stop_carries_the_reason(self):
        self.assertEqual(parse_batch('ACTION: STOP\nALASAN: habis', 5), ('stop', 'habis'))
        self.assertEqual(parse_batch('ACTION: STOP', 5), ('stop', 'tidak ada alasan'))

    def test_plan_keeps_the_alternatives_and_is_at_most_five_steps(self):
        text = '<think>...</think>\nRENCANA:\n1. Cari user | alternatif: lihat grup\n2. Lapor\n3. A\n4. B\n5. C\n6. D'
        self.assertEqual(parse_plan(text), ['Cari user | alternatif: lihat grup', 'Lapor', 'A', 'B', 'C'])
        self.assertEqual(parse_plan('tidak ada format'), [])
        self.assertIn('alternatif', build_plan_prompt('x'))

    def test_done_reply_needs_both_lines(self):
        self.assertEqual(parse_done('DONE: yes\nEVIDENCE: Host Name: LAB-2'), (True, 'Host Name: LAB-2'))
        self.assertEqual(parse_done('DONE: no\nEVIDENCE:'), (False, ''))
        self.assertEqual(parse_done('kayaknya sudah'), (False, ''))

    def test_evidence_can_be_checked_word_by_word(self):
        output = 'Stopped  Spooler   Print Spooler'
        self.assertTrue(evidence_in_output('Print Spooler stopped', output))
        self.assertFalse(evidence_in_output('Print Spooler failed', output))
        self.assertFalse(evidence_in_output('Print stopped error', 'Stopped Spooler'))  # 'print' and 'error' are missing

    def test_evidence_must_be_in_the_output(self):
        output = 'Host Name:   LAB-2\nIPv4 Address. . . : 192.168.56.20'
        self.assertTrue(evidence_in_output('host name: lab-2', output))
        self.assertFalse(evidence_in_output('Host Name: LAB-9', output))
        self.assertFalse(evidence_in_output('', output))
        self.assertFalse(evidence_in_output('ab', output))   # too short to count

    def test_the_words_the_objective_asks_for_are_read_from_memuat(self):
        self.assertEqual(required_phrases('Hapus X. Tujuan tercapai jika output memuat "No items found".'),
                         ('all', ['No items found']))
        self.assertEqual(required_phrases('Hentikan Y. Tujuan tercapai jika output memuat "stopped" atau "dihentikan".'),
                         ('any', ['stopped', 'dihentikan']))
        self.assertEqual(required_phrases('Lakukan sesuatu.'), ('all', []))

    def test_required_words_must_all_be_shown_unless_any_is_allowed(self):
        self.assertTrue(phrases_shown('all', [], 'apa saja'))
        self.assertFalse(phrases_shown('all', ['No items found'], 'The backup catalog has been deleted'))
        self.assertTrue(phrases_shown('all', ['No items found'], 'vssadmin: No items found that satisfy the query'))
        self.assertTrue(phrases_shown('any', ['stopped', 'dihentikan'], 'Service Stopped'))

    def test_notes_are_labelled_as_data(self):
        prompt = build_choice_prompt('Temukan user', [_link('A', 'List users')], notes=notes_text({('user', 'alice')}))
        self.assertIn('user=alice', prompt)
        self.assertIn(DATA_LABEL, prompt)


class ShortlistTests(unittest.TestCase):
    def test_indonesian_objective_words_mean_the_english_ability_words(self):
        terms = objective_terms('Hentikan layanan Spooler')
        self.assertIn('stop', terms)
        self.assertIn('service', terms)
        self.assertIn('spooler', terms)

    def test_the_matching_ability_comes_first(self):
        links = [_link('A', 'List processes', tactic='discovery'),
                 _link('B', 'Windows - Stop service using net.exe', tactic='stealth'),
                 _link('C', 'Create A New Local User Account', tactic='persistence')]
        self.assertEqual(shortlist(links, 'Hentikan layanan Spooler')[0].ability_id, 'B')

    def test_when_nothing_matches_the_list_mixes_tactics(self):
        links = [_link(f'd{i}', f'x{i}', tactic='discovery') for i in range(20)]
        links += [_link(f'e{i}', f'y{i}', tactic='execution') for i in range(10)]
        first = shortlist(links, 'Lakukan sesuatu', [])
        self.assertEqual({link.tactic for link in first[:2]}, {'discovery', 'execution'})

    def test_when_nothing_matches_the_list_is_short(self):
        links = [_link(f'a{i}', f'x{i}') for i in range(100)]
        self.assertEqual(len(shortlist(links, 'Lakukan sesuatu', [])), SHORTLIST_FALLBACK)

    def test_only_matching_abilities_are_shown(self):
        # Unrelated abilities are not offered just to fill the list (no persistence or stealth by accident).
        matching = [_link(f'm{i}', f'Stop service {i}', tactic='stealth') for i in range(3)]
        unrelated = [_link(f'u{i}', f'Create file {i}', tactic='persistence') for i in range(10)]
        shown = shortlist(unrelated + matching, 'Hentikan layanan Spooler', [])
        self.assertEqual([link.ability_id for link in shown], ['m0', 'm1', 'm2'])

    def test_the_list_never_exceeds_its_size(self):
        links = [_link(f'm{i}', f'Stop service {i}') for i in range(60)]
        self.assertEqual(len(shortlist(links, 'Hentikan layanan', [])), SHORTLIST_SIZE)


class FilterTests(unittest.TestCase):
    def test_an_ability_with_an_unfilled_fact_variable_is_not_offered(self):
        with_var = _link('A', 'Set wallpaper', command='Invoke-WebRequest #{url_of_wallpaper}')
        plain = _link('B', 'List users', command='net user')
        self.assertEqual([l.ability_id for l in offered_links([with_var, plain])], ['B'])

    def test_an_encoded_command_with_a_fact_variable_is_not_offered(self):
        encoded = base64.b64encode(b'Copy file #{file_path}').decode('ascii')
        hidden = _link('H', 'Copy file', command=encoded)
        self.assertEqual(offered_links([hidden]), [])

    def test_abilities_that_would_cut_the_agent_control_are_not_offered(self):
        links = [_link('A', 'Restart System - Windows', command='shutdown /r /t 1'),
                 _link('B', 'Logoff System - Windows', command='garbage'),
                 _link('C', 'Disrupt WIFI', command='.\\wifi.ps1 -Off'),
                 _link('D', 'Reboot box', command='shutdown /r /t 1'),
                 _link('E', 'List users', command='net user')]
        self.assertEqual([l.ability_id for l in offered_links(links)], ['E'])

    def test_once_the_atomic_module_is_missing_its_abilities_are_not_offered(self):
        needs = 'PathToAtomicsFolder\\..\\ExternalPayloads\\Sdelete\\sdelete.exe'
        links = [_link('A', 'Overwrite file', command=needs), _link('N', 'Create user', command='net user T1 /add')]
        self.assertEqual([l.ability_id for l in offered_links(links, frozenset({'atomic'}))], ['N'])
        self.assertEqual(len(offered_links(links)), 2)


class AgenticLoopTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _run(self, client, llm, max_steps=1, max_failures=3, stop_file=None, objective='Lakukan langkah uji'):
        return run_agentic(client, llm, objective, 'p1', ('lab-vm-1',), self.root / 'run',
                           stop_file=stop_file or self.root / 'STOP', max_steps=max_steps,
                           max_failures=max_failures, log=lambda *_: None)

    def _steps(self):
        return [json.loads(line) for line in (self.root / 'run' / 'steps.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_the_plan_is_kept_and_shown_to_the_choice(self):
        seen = []
        client = FakeClient([_link('A', 'List users')])
        summary = self._run(client, FakeLLM(responder(seen=seen)))
        self.assertEqual(summary['plan'], ['Cari user lokal', 'Tulis hasilnya ke file'])
        choices = [p for p in seen if 'Pilih sampai' in p]
        self.assertTrue(choices)
        self.assertTrue(all('Cari user lokal' in p for p in choices))

    def test_without_a_plan_nothing_is_sent(self):
        client = FakeClient([_link('A', 'List users')])
        summary = self._run(client, FakeLLM(responder(plan='saya belum tahu')), max_steps=2)
        self.assertEqual(client.submitted, [])
        self.assertTrue(summary['stop'].startswith('planner_no_valid_choice (rencana'), summary['stop'])
        self.assertFalse((self.root / 'run' / 'steps.jsonl').exists())

    def test_a_failed_llm_call_for_the_plan_stops_before_anything_is_sent(self):
        client = FakeClient([_link('A', 'List users')])
        summary = self._run(client, FakeLLM(ok=False, error='timeout'))
        self.assertEqual(client.submitted, [])
        self.assertEqual(summary['stop'], 'planner_no_valid_choice (rencana: timeout)')

    def test_a_batch_of_two_abilities_is_sent_together(self):
        client = FakeClient([_link('A', 'Query A'), _link('B', 'Query B'), _link('C', 'Query C')])
        summary = self._run(client, FakeLLM(responder(choice='ACTION: 0, 1', plan=NEUTRAL_PLAN)), max_steps=1)
        self.assertEqual(client.submitted, ['A', 'B'])
        self.assertEqual(summary['steps_sent'], 2)
        self.assertEqual(summary['batches'], 1)
        self.assertEqual(len(self._steps()), 2)

    def test_a_batch_is_refused_when_one_ability_is_outside_the_lab(self):
        client = FakeClient([_link('A', 'Query A'), _link('B', 'Query B', host='office-pc')])
        summary = self._run(client, FakeLLM(responder(choice='ACTION: 0, 1', plan=NEUTRAL_PLAN)), max_steps=2)
        self.assertEqual(client.submitted, [])
        self.assertTrue(summary['stop'].startswith('scope:'), summary['stop'])
        self.assertEqual([rec['status'] for rec in self._steps()], ['not sent', 'not sent'])

    def test_an_ambiguous_choice_sends_nothing(self):
        client = FakeClient([_link('A', 'Query A'), _link('B', 'Query B')])
        summary = self._run(client, FakeLLM(responder(choice='ACTION: 0\nACTION: 1', plan=NEUTRAL_PLAN)))
        self.assertEqual(client.submitted, [])
        self.assertTrue(summary['stop'].startswith('planner_no_valid_choice (aksi'), summary['stop'])

    def test_the_planner_can_stop_with_a_reason(self):
        client = FakeClient([_link('A', 'Query A')])
        reply = 'ACTION: STOP\nALASAN: tidak ada aksi yang menuju domain controller'
        summary = self._run(client, FakeLLM(responder(choice=reply)), max_steps=3)
        self.assertEqual(client.submitted, [])
        self.assertEqual(summary['stop'], 'planner_stopped: tidak ada aksi yang menuju domain controller')

    def test_objective_is_reached_only_with_a_quote_from_the_output(self):
        client = FakeClient([_link('A', 'Query A'), _link('B', 'Query B')],
                            outputs={'B': 'Host Name: LAB-2 joined the domain'})
        done = 'DONE: yes\nEVIDENCE: Host Name: LAB-2'
        summary = self._run(client, FakeLLM(responder(choice='ACTION: 0, 1', done=done, plan=NEUTRAL_PLAN)),
                            max_steps=3)
        self.assertTrue(summary['objective_reached'])
        self.assertEqual(summary['stop'], 'objective_reached')
        self.assertEqual(summary['evidence'], 'Host Name: LAB-2')

    def test_a_yes_without_evidence_in_the_output_is_not_accepted(self):
        client = FakeClient([_link('A', 'Query A')], outputs={'A': 'nothing useful here'})
        done = 'DONE: yes\nEVIDENCE: Host Name: LAB-9'
        summary = self._run(client, FakeLLM(responder(done=done, plan=NEUTRAL_PLAN)), max_steps=1)
        self.assertFalse(summary['objective_reached'])
        self.assertEqual(summary['stop'], 'budget_used')
        self.assertTrue(self._steps()[0]['objective_check']['claimed_without_evidence'])

    def test_an_unproven_claim_makes_the_next_step_check_only(self):
        # Step 1 succeeds, but its output does not show the claimed quote. Step 2 may only use discovery abilities.
        seen = []
        links = [_link('A', 'Stop service', tactic='discovery'), _link('D', 'Check service', tactic='discovery'),
                 _link('E', 'Delete everything', tactic='impact')]
        client = FakeClient(links, outputs={'A': 'Service stopped'})
        done = 'DONE: yes\nEVIDENCE: Print Spooler stopped with an error'
        summary = self._run(client, FakeLLM(responder(done=done, plan=NEUTRAL_PLAN, seen=seen)), max_steps=3)
        self.assertEqual(client.submitted, ['A', 'D'])
        self.assertEqual(summary['stop'], 'objective_unverified')
        self.assertEqual(summary['unverified_claims'], 1)
        self.assertTrue(any('HANYA aksi pengecekan' in p for p in seen))

    def test_the_objective_words_must_be_in_the_output_even_if_the_quote_is_real(self):
        objective = 'Hapus salinan shadow copy. Tujuan tercapai jika output memuat "No items found".'
        done = 'DONE: yes\nEVIDENCE: The backup catalog has been successfully deleted'
        client = FakeClient([_link('W', 'Delete catalog')],
                            outputs={'W': 'The backup catalog has been successfully deleted'})
        summary = self._run(client, FakeLLM(responder(done=done, plan=NEUTRAL_PLAN)), max_steps=1,
                            objective=objective)
        self.assertFalse(summary['objective_reached'])
        self.assertEqual(summary['stop'], 'budget_used')
        self.assertEqual(summary['required'], ['all', ['No items found']])

    def test_the_objective_is_reached_when_its_words_are_in_the_output(self):
        objective = 'Hapus salinan shadow copy. Tujuan tercapai jika output memuat "No items found".'
        done = 'DONE: yes\nEVIDENCE: No items found that satisfy the query'
        client = FakeClient([_link('V', 'List shadows')],
                            outputs={'V': 'No items found that satisfy the query.'})
        summary = self._run(client, FakeLLM(responder(done=done, plan=NEUTRAL_PLAN)), max_steps=3,
                            objective=objective)
        self.assertTrue(summary['objective_reached'])

    def test_a_one_step_plan_is_asked_again(self):
        queue = ['RENCANA:\n1. Satu', 'RENCANA:\n1. Satu\n2. Dua']

        def respond(prompt):
            if 'Buat rencana singkat' in prompt:
                return queue.pop(0) if len(queue) > 1 else queue[0]
            if 'Pilih sampai' in prompt:
                return 'ACTION: 0'
            return 'DONE: no\nEVIDENCE:'
        client = FakeClient([_link('A', 'Query A')])
        summary = self._run(client, FakeLLM(respond), max_steps=1)
        self.assertEqual(summary['plan'], ['Satu', 'Dua'])

    def test_a_batch_with_one_success_is_not_a_failed_step(self):
        client = FakeClient([_link('A', 'Query A'), _link('B', 'Query B')], statuses={'A': '1'})
        summary = self._run(client, FakeLLM(responder(choice='ACTION: 0, 1', plan=NEUTRAL_PLAN)), max_steps=3,
                            max_failures=1)
        self.assertEqual(summary['stop'], 'no_allowed_candidate')
        self.assertEqual(summary['failures'], 1)

    def test_batches_that_all_fail_in_a_row_stop_the_run(self):
        links = [_link(x, f'Query {x}') for x in 'ABCD']
        client = FakeClient(links, statuses={x: '1' for x in 'ABCD'})
        summary = self._run(client, FakeLLM(responder(plan=NEUTRAL_PLAN)), max_steps=4, max_failures=2)
        self.assertEqual(client.submitted, ['A', 'B'])
        self.assertTrue(summary['stop'].startswith('2 batch gagal berturut-turut'), summary['stop'])

    def test_once_the_atomic_module_is_missing_its_abilities_are_not_offered_again(self):
        needs = 'PathToAtomicsFolder\\..\\ExternalPayloads\\Sdelete\\sdelete.exe -accepteula'
        a = _link('A', 'Query services', command=needs)
        c = _link('C', 'Dump users', command=needs)
        client = FakeClient([a, c, _link('B', 'List users')],
                            outputs={'A': "& : The module 'PathToAtomicsFolder' could not be loaded."},
                            statuses={'A': '1'})
        summary = self._run(client, FakeLLM(responder(plan=NEUTRAL_PLAN)), max_steps=2, max_failures=3)
        self.assertEqual(client.submitted, ['A', 'B'])
        self.assertEqual(summary['blocked'], ['atomic'])

    def test_a_link_that_needs_a_person_stops_the_run(self):
        client = FakeClient([_link('A', 'Query A')], statuses={'A': 'untrusted'})
        summary = self._run(client, FakeLLM(responder(plan=NEUTRAL_PLAN)), max_steps=3)
        self.assertEqual(summary['stop'], 'link status untrusted')

    def test_the_stop_file_prevents_any_step(self):
        client = FakeClient([_link('A', 'Query A')])
        stop = self.root / 'STOP'
        stop.write_text('stop', encoding='utf-8')
        summary = self._run(client, FakeLLM(responder(plan=NEUTRAL_PLAN)), stop_file=stop)
        self.assertEqual(client.submitted, [])
        self.assertEqual(summary['stop'], 'stop_file')

    def test_an_empty_objective_is_refused_before_anything_is_sent(self):
        client = FakeClient([_link('A', 'List users')])
        with self.assertRaises(ValueError):
            self._run(client, FakeLLM(responder()), objective='   ')
        self.assertEqual(client.submitted, [])


if __name__ == '__main__':
    unittest.main()
