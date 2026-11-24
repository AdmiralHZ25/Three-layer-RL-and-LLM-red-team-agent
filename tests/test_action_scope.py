"""Tests for red.caldera.action_scope. Offline: reads only the files under tests/fixtures/action_scope.

Run from the RED_AGENT folder:

    python -m unittest tests.test_action_scope -v

Policy (owner decision, 2026-11-19): every Caldera tactic is allowed. Only the name words, the per-profile allow-list,
and the platform and index checks still refuse an ability.
"""
import unittest
from pathlib import Path
from unittest import mock

from red.caldera import action_scope as scope

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'action_scope'
SAMPLE_ADVERSARY = FIXTURES / 'adversary_sample.yml'  # byte-identical copy of one stockpile profile
MIXED_ADVERSARY = FIXTURES / 'adversary_mixed.yml'    # synthetic profile for the allow-list tests

ALL_CALDERA_TACTICS = (
    'discovery', 'execution', 'privilege-escalation', 'persistence', 'defense-evasion', 'credential-access',
    'lateral-movement', 'collection', 'command-and-control', 'exfiltration', 'impact',
    'stealth', 'multiple', 'defense-impairment',
)


class ForbiddenReason(unittest.TestCase):
    def test_no_tactic_is_refused_by_itself(self):
        # Owner's decision: every tactic may be used, impact included.
        for tactic in ALL_CALDERA_TACTICS:
            with self.subTest(tactic=tactic):
                self.assertIsNone(scope.forbidden_reason({'name': 'Harmless name', 'tactic': tactic}))

    def test_no_tactic_is_configured_as_forbidden(self):
        self.assertEqual(scope.FORBIDDEN_TACTICS, frozenset())

    def test_defense_evasion_tactic_is_allowed(self):
        self.assertIsNone(scope.forbidden_reason({'name': 'Harmless name', 'tactic': 'defense-evasion'}))

    def test_log_removal_under_stealth_tactic_is_allowed(self):
        # Changed on 2026-11-19: the stealth tactic is no longer a block. Neither name contains a name word.
        for name in ('Prevent Powershell History Logging', 'Indicator Removal using FSUtil'):
            with self.subTest(name=name):
                self.assertIsNone(scope.forbidden_reason({'name': name, 'tactic': 'stealth'}))

    def test_a_tactic_can_still_be_blocked_by_configuration(self):
        # The mechanism stays in place, so a tactic can be blocked again by adding it to FORBIDDEN_TACTICS.
        with mock.patch.object(scope, 'FORBIDDEN_TACTICS', frozenset({'impact'})):
            reason = scope.forbidden_reason({'name': 'Harmless name', 'tactic': 'impact'})
            self.assertIsNotNone(reason)
            self.assertIn('impact', reason)
            self.assertIsNone(scope.forbidden_reason({'name': 'Harmless name', 'tactic': 'discovery'}))

    def test_tactic_spelling_does_not_matter_for_name_words(self):
        for tactic in ('Defense Evasion', 'lateral_movement', 'discovery'):
            with self.subTest(tactic=tactic):
                self.assertIsNotNone(scope.forbidden_reason({'name': 'Query Defender status', 'tactic': tactic}))

    def test_name_containing_defender_is_refused(self):
        reason = scope.forbidden_reason({'name': 'Query Defender status', 'tactic': 'discovery'})
        self.assertIsNotNone(reason)
        self.assertIn('Defender', reason)

    def test_name_containing_clear_is_refused_in_any_case(self):
        self.assertIsNotNone(scope.forbidden_reason({'name': 'clear event logs', 'tactic': 'discovery'}))
        self.assertIsNotNone(scope.forbidden_reason({'name': 'CLEAR EVENT LOGS', 'tactic': 'discovery'}))

    def test_every_forbidden_name_word_is_refused(self):
        for word in scope.FORBIDDEN_NAME_WORDS:
            with self.subTest(word=word):
                self.assertIsNotNone(scope.forbidden_reason({'name': f'Test {word.lower()} action', 'tactic': 'discovery'}))

    def test_name_word_is_refused_under_every_tactic(self):
        for tactic in ALL_CALDERA_TACTICS:
            with self.subTest(tactic=tactic):
                self.assertIsNotNone(scope.forbidden_reason({'name': 'Disable Firewall rule', 'tactic': tactic}))

    def test_ordinary_discovery_is_allowed(self):
        self.assertIsNone(scope.forbidden_reason({'name': 'List running processes', 'tactic': 'discovery'}))


class LoadAdversary(unittest.TestCase):
    def test_reads_ability_order_from_sample_profile(self):
        name, ids = scope.load_adversary(SAMPLE_ADVERSARY)
        self.assertEqual(name, 'Advanced Thief via DropBox')
        self.assertEqual(ids, [
            '10fad81e-3f68-47be-83b6-fbee7711c6a9',
            '30a8cf10-73dc-497c-8261-a64cc9e91505',
            '3ce95a28-25fc-4a7e-a0cd-0fdb190e2081',
        ])

    def test_keeps_repeated_ids_in_order(self):
        _, ids = scope.load_adversary(MIXED_ADVERSARY)
        self.assertEqual(ids[0], 'fx-win-discovery')
        self.assertEqual(ids[-1], 'fx-missing-from-index')
        self.assertEqual(len(ids), 8)


class LoadAbilityIndex(unittest.TestCase):
    def setUp(self):
        self.index = scope.load_ability_index(FIXTURES)

    def test_indexes_only_files_under_data_abilities(self):
        self.assertEqual(len(self.index), 7)
        self.assertNotIn('01c96671-afd3-47d4-8d31-8c116cc0221a', self.index)
        self.assertNotIn('fixture-mixed-profile', self.index)

    def test_entry_has_name_tactic_technique_and_platforms(self):
        self.assertEqual(self.index['fx-win-discovery'], {
            'name': 'Fixture list directory',
            'tactic': 'discovery',
            'technique_id': 'T1083',
            'platforms': ['windows'],
        })

    def test_reads_technique_id_and_executors_shape(self):
        entry = self.index['fx-executors-windows']
        self.assertEqual(entry['technique_id'], 'T1057')
        self.assertEqual(entry['platforms'], ['windows'])

    def test_splits_comma_separated_platforms(self):
        self.assertEqual(self.index['fx-mixed-platforms']['platforms'], ['darwin', 'linux'])

    def test_tactic_is_normalised(self):
        self.assertEqual(self.index['fx-win-evasion']['tactic'], 'defense-evasion')

    def test_unknown_id_is_absent(self):
        self.assertNotIn('fx-missing-from-index', self.index)


class AllowedActions(unittest.TestCase):
    def setUp(self):
        self.index = scope.load_ability_index(FIXTURES)

    def test_keeps_only_the_requested_platform(self):
        allowed = scope.allowed_actions(MIXED_ADVERSARY, 'windows', {'discovery'}, self.index)
        self.assertEqual(allowed, ['fx-win-discovery', 'fx-executors-windows'])

    def test_linux_and_darwin_select_their_own_abilities(self):
        linux = scope.allowed_actions(MIXED_ADVERSARY, 'linux', {'discovery'}, self.index)
        darwin = scope.allowed_actions(MIXED_ADVERSARY, 'darwin', {'discovery'}, self.index)
        self.assertEqual(linux, ['fx-linux-discovery', 'fx-mixed-platforms'])
        self.assertEqual(darwin, ['fx-mixed-platforms'])

    def test_defense_evasion_ability_is_allowed_when_its_tactic_is_listed(self):
        allowed = scope.allowed_actions(MIXED_ADVERSARY, 'windows', {'discovery', 'defense-evasion'}, self.index)
        self.assertIn('fx-win-evasion', allowed)

    def test_name_blocked_abilities_stay_refused_whatever_the_tactic(self):
        allowed = scope.allowed_actions(MIXED_ADVERSARY, 'windows', {'discovery', 'defense-evasion'}, self.index)
        self.assertNotIn('fx-win-defender', allowed)
        self.assertNotIn('fx-clear-logs', allowed)

    def test_tactic_outside_allowed_tactics_is_refused(self):
        self.assertEqual(scope.allowed_actions(MIXED_ADVERSARY, 'windows', {'collection'}, self.index), [])
        self.assertEqual(scope.allowed_actions(MIXED_ADVERSARY, 'windows', set(), self.index), [])

    def test_id_missing_from_index_is_refused(self):
        allowed = scope.allowed_actions(MIXED_ADVERSARY, 'windows', {'discovery', 'defense-evasion'}, self.index)
        self.assertNotIn('fx-missing-from-index', allowed)


if __name__ == '__main__':
    unittest.main()
