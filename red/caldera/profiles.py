"""Saved profiles for the automatic Caldera run (menu option g).

A profile keeps only the choices the operator made: the agent (paw, host, platform), the allowed ability ids,
the mode and the step budget. It never holds the API key or any environment value.
Profiles are JSON files in runs/caldera_profiles/<name>.json. The runs/ folder is ignored by git.
"""
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROFILE_FIELDS = ('name', 'paw', 'host', 'platform', 'ability_ids', 'mode', 'budget', 'created')
PROFILE_MODES = ('rl', 'llm', 'rl+llm')   # same names as red.caldera.choosers.MODES
BUDGET_MIN, BUDGET_MAX = 1, 20
_NAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}')


class ProfileError(ValueError):
    """A profile name or profile file is not acceptable. The message is safe to show to the operator."""


def _base(root):
    return ROOT if root is None else Path(root)


def profiles_dir(root=None):
    return _base(root) / 'runs' / 'caldera_profiles'


def validate_name(name):
    """Return the name if it is safe as a file name: letters, digits, - and _, 1 to 40 characters."""
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ProfileError('nama profil hanya boleh huruf, angka, - atau _ (1 sampai 40 karakter, tanpa spasi)')
    return name


def profile_path(name, root=None):
    return profiles_dir(root) / f'{validate_name(name)}.json'


def validate_new_name(name, root=None):
    """Like validate_name, and the name must not be used by a saved profile yet."""
    validate_name(name)
    if profile_path(name, root).exists():
        raise ProfileError(f'profil dengan nama {name} sudah ada. Pilih nama lain.')
    return name


def _is_text(value):
    return isinstance(value, str) and value.strip() != ''


def validate_profile(data, expected_name=None):
    """Return data if it is a complete, well-formed profile. Raises ProfileError otherwise.

    Only the fields in PROFILE_FIELDS are allowed, so a profile file cannot carry extra values (such as a key).
    """
    if not isinstance(data, dict):
        raise ProfileError('isi profil harus berupa objek JSON')
    missing = [f for f in PROFILE_FIELDS if f not in data]
    if missing:
        raise ProfileError('field kurang: ' + ', '.join(missing))
    extra = len(set(data) - set(PROFILE_FIELDS))
    if extra:
        raise ProfileError(f'ada {extra} field yang tidak dikenal; profil hanya boleh berisi field resmi')
    validate_name(data['name'])
    if expected_name is not None and data['name'] != expected_name:
        raise ProfileError('nama di dalam file tidak sama dengan nama file')
    for field in ('paw', 'host', 'platform'):
        if not _is_text(data[field]):
            raise ProfileError(f'field {field} harus teks yang tidak kosong')
    ids = data['ability_ids']
    if not isinstance(ids, list) or not ids or not all(_is_text(i) for i in ids):
        raise ProfileError('ability_ids harus daftar berisi minimal satu id aksi')
    if data['mode'] not in PROFILE_MODES:
        raise ProfileError('mode harus salah satu dari: ' + ', '.join(PROFILE_MODES))
    budget = data['budget']
    if isinstance(budget, bool) or not isinstance(budget, int) or not BUDGET_MIN <= budget <= BUDGET_MAX:
        raise ProfileError(f'budget harus angka {BUDGET_MIN} sampai {BUDGET_MAX}')
    if not _is_text(data['created']):
        raise ProfileError('created harus berisi waktu')
    try:
        datetime.fromisoformat(data['created'])
    except ValueError:
        raise ProfileError('created harus waktu dalam format ISO')
    return data


def build_profile(name, paw, host, platform, ability_ids, mode, budget, created=None):
    """A new profile dict, checked. Duplicate ability ids are dropped. created defaults to now."""
    profile = {
        'name': name,
        'paw': paw,
        'host': host,
        'platform': platform,
        'ability_ids': list(dict.fromkeys(ability_ids)),
        'mode': mode,
        'budget': budget,
        'created': created if created is not None else datetime.now().isoformat(timespec='seconds'),
    }
    return validate_profile(profile)


def _read(path, expected_name):
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise ProfileError('profil tidak ditemukan')
    except (OSError, UnicodeDecodeError):
        raise ProfileError('file profil tidak bisa dibaca')
    try:
        data = json.loads(text)
    except ValueError:
        raise ProfileError('file profil bukan JSON yang benar')
    return validate_profile(data, expected_name=expected_name)


def load_profile(name, root=None):
    """One saved profile by name. Raises ProfileError if it is missing or broken."""
    return _read(profile_path(name, root), expected_name=name)


def list_profiles(root=None):
    """Valid saved profiles sorted by name, and the refused files as (file name, reason)."""
    good, rejected = [], []
    folder = profiles_dir(root)
    if not folder.is_dir():
        return good, rejected
    for path in sorted(folder.glob('*.json')):
        try:
            good.append(_read(path, expected_name=path.stem))
        except ProfileError as exc:
            rejected.append((path.name, str(exc)))
    return good, rejected


def save_profile(profile, root=None):
    """Write a new profile to runs/caldera_profiles/<name>.json. Never overwrites. Returns the path."""
    validate_profile(profile)
    path = profile_path(profile['name'], root)
    validate_new_name(profile['name'], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'x', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
            f.write('\n')
    except FileExistsError:
        raise ProfileError(f'profil dengan nama {profile["name"]} sudah ada. Pilih nama lain.')
    return path


def format_summary(profile):
    """Short lines shown before a run is confirmed. Contains no secret values."""
    created = datetime.fromisoformat(profile['created']).strftime('%Y-%m-%d %H:%M')
    return [
        f"  profil    : {profile['name']}",
        f"  agent     : {profile['host']}  (paw {profile['paw']})",
        f"  platform  : {profile['platform']}",
        f"  aksi      : {len(profile['ability_ids'])} id",
        f"  mode      : {profile['mode']}",
        f"  langkah   : maksimal {profile['budget']}",
        f"  dibuat    : {created}",
    ]
