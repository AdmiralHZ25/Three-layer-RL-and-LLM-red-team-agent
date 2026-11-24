"""Tests for saved Caldera profiles (red/caldera/profiles.py). Nothing here contacts Caldera or Ollama.

Each test works in a temporary folder and patches ROOT, so the real runs/ folder is never touched.
"""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from red.caldera import profiles

CREATED = '2026-11-19T10:30:00'


def _data(**changes):
    """Raw profile data as it would be written to a file."""
    data = {'name': 'recon-kali', 'paw': 'abc123', 'host': 'kali-lab', 'platform': 'linux',
            'ability_ids': ['ab-1', 'ab-2'], 'mode': 'rl+llm', 'budget': 5, 'created': CREATED}
    data.update(changes)
    return data


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        patcher = mock.patch.object(profiles, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folder = self.root / 'runs' / 'caldera_profiles'

    def _build(self, **changes):
        args = dict(name='recon-kali', paw='abc123', host='kali-lab', platform='linux',
                    ability_ids=['ab-1', 'ab-2'], mode='rl+llm', budget=5, created=CREATED)
        args.update(changes)
        return profiles.build_profile(**args)

    def test_save_then_load_roundtrip_writes_only_profile_fields(self):
        profile = self._build()
        path = profiles.save_profile(profile)
        self.assertEqual(path, self.folder / 'recon-kali.json')
        self.assertEqual(profiles.load_profile('recon-kali'), profile)
        saved = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(saved, profile)
        self.assertEqual(sorted(saved), sorted(profiles.PROFILE_FIELDS))

    def test_profile_folder_follows_root(self):
        self.assertEqual(profiles.profiles_dir(), self.folder)

    def test_explicit_root_is_used_when_given(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        path = profiles.save_profile(self._build(), root=other.name)
        self.assertEqual(path, Path(other.name) / 'runs' / 'caldera_profiles' / 'recon-kali.json')
        self.assertFalse(self.folder.exists())

    def test_default_created_is_now_in_iso_format(self):
        before = datetime.now().replace(microsecond=0)
        profile = self._build(created=None)
        stamp = datetime.fromisoformat(profile['created'])
        self.assertTrue(before <= stamp <= datetime.now())

    def test_duplicate_ability_ids_are_dropped_in_order(self):
        profile = self._build(ability_ids=['ab-1', 'ab-2', 'ab-1'])
        self.assertEqual(profile['ability_ids'], ['ab-1', 'ab-2'])

    def test_budget_limits_are_1_and_20(self):
        self.assertEqual(self._build(budget=1)['budget'], 1)
        self.assertEqual(self._build(budget=20)['budget'], 20)

    def test_save_never_overwrites_an_existing_profile(self):
        profiles.save_profile(self._build())
        with self.assertRaises(profiles.ProfileError):
            profiles.save_profile(self._build(budget=7))
        self.assertEqual(profiles.load_profile('recon-kali')['budget'], 5)
        with self.assertRaises(profiles.ProfileError):
            profiles.validate_new_name('recon-kali')

    def test_save_rejects_an_invalid_profile_and_writes_nothing(self):
        with self.assertRaises(profiles.ProfileError):
            profiles.save_profile({'name': 'recon-kali'})
        self.assertFalse((self.folder / 'recon-kali.json').exists())

    def test_load_missing_profile_raises(self):
        with self.assertRaises(profiles.ProfileError):
            profiles.load_profile('tidak-ada')

    def test_list_is_empty_when_folder_does_not_exist(self):
        self.assertEqual(profiles.list_profiles(), ([], []))

    def test_name_rules(self):
        for bad in ('', '../evil', 'a/b', 'nama spasi', 'x' * 41, 'api.key', '-awal', None):
            with self.subTest(name=bad):
                with self.assertRaises(profiles.ProfileError):
                    profiles.validate_name(bad)
        for good in ('recon-kali_1', 'x' * 40, 'A'):
            with self.subTest(name=good):
                self.assertEqual(profiles.validate_name(good), good)

    def test_build_rejects_bad_values(self):
        bad_changes = [
            {'mode': 'auto'}, {'mode': None},
            {'budget': 0}, {'budget': 21}, {'budget': True}, {'budget': '5'},
            {'ability_ids': []}, {'ability_ids': ['']},
            {'paw': '  '}, {'host': ''}, {'platform': None},
            {'created': 'kemarin'}, {'created': ''},
            {'name': '../evil'},
        ]
        for changes in bad_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(profiles.ProfileError):
                    self._build(**changes)

    def test_broken_profile_files_are_refused_and_good_ones_kept(self):
        profiles.save_profile(self._build())
        self.folder.mkdir(parents=True, exist_ok=True)
        without_budget = _data(name='missing-budget')
        del without_budget['budget']
        cases = {
            'missing-budget': json.dumps(without_budget),
            'extra-field': json.dumps(_data(name='extra-field', api_key='dummy')),
            'bad-mode': json.dumps(_data(name='bad-mode', mode='auto')),
            'budget-zero': json.dumps(_data(name='budget-zero', budget=0)),
            'budget-too-big': json.dumps(_data(name='budget-too-big', budget=21)),
            'budget-bool': json.dumps(_data(name='budget-bool', budget=True)),
            'no-abilities': json.dumps(_data(name='no-abilities', ability_ids=[])),
            'bad-created': json.dumps(_data(name='bad-created', created='kemarin')),
            'name-mismatch': json.dumps(_data(name='other-name')),
            'bad-json': '{"name": ',
            'not-object': '["recon-kali"]',
        }
        for stem, body in cases.items():
            (self.folder / f'{stem}.json').write_text(body, encoding='utf-8')
        (self.folder / 'binary.json').write_bytes(b'\xff\xfe\x00')
        stems = list(cases) + ['binary']

        good, rejected = profiles.list_profiles()
        self.assertEqual([p['name'] for p in good], ['recon-kali'])
        self.assertEqual({name for name, _ in rejected}, {f'{stem}.json' for stem in stems})
        for stem in stems:
            with self.subTest(file=stem):
                with self.assertRaises(profiles.ProfileError):
                    profiles.load_profile(stem)

    def test_refusal_reason_never_echoes_a_value(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / 'extra-field.json').write_text(
            json.dumps(_data(name='extra-field', api_key='dummy-secret')), encoding='utf-8')
        _, rejected = profiles.list_profiles()
        self.assertEqual(len(rejected), 1)
        reason = rejected[0][1]
        self.assertIn('tidak dikenal', reason)
        self.assertNotIn('dummy-secret', reason)

    def test_summary_lines_are_in_order_and_complete(self):
        profile = self._build()
        self.assertEqual(profiles.format_summary(profile), [
            '  profil    : recon-kali',
            '  agent     : kali-lab  (paw abc123)',
            '  platform  : linux',
            '  aksi      : 2 id',
            '  mode      : rl+llm',
            '  langkah   : maksimal 5',
            '  dibuat    : 2026-11-19 10:30',
        ])


if __name__ == '__main__':
    unittest.main()
