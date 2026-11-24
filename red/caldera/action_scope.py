"""Action scope: which Caldera abilities the RED agent may propose.

Built offline from the adversary profiles and ability YAML files shipped under caldera/plugins.
Nothing here contacts the Caldera server. An ability the local files cannot describe is treated as not allowed.
"""
from pathlib import Path

import yaml

try:
    _YamlLoader = yaml.CSafeLoader
except AttributeError:  # PyYAML built without libyaml
    _YamlLoader = yaml.SafeLoader

# Tactics that are never allowed, in Caldera's normalised form (lower case, hyphens).
# Empty on purpose: the owner decided that every Caldera tactic may be used, impact included. The blocks that remain are
# the name words below, and the lab-host check in scope.py. To block a tactic, add its name to this set.
FORBIDDEN_TACTICS = frozenset()

# Words that forbid an ability when they appear in its name (case-insensitive substring match).
FORBIDDEN_NAME_WORDS = (
    'Defender',
    'Clear',
    'Sysmon',
    'Ransom',
    'Masquerad',
    'LSASS',
    'Firewall',
    'Hidden',
    'Bypass',
    'Grant',
    'Disable',
    'Recycle',
    'Ryuk',
)


def _norm_tactic(value):
    return str(value or '').strip().lower().replace('_', '-').replace(' ', '-')


def _is_ability_file(path):
    parts = path.parts
    return any(parts[i:i + 2] == ('data', 'abilities') for i in range(len(parts) - 1))


def _technique_id(ability):
    technique = ability.get('technique')
    value = technique.get('attack_id') if isinstance(technique, dict) else ability.get('technique_id')
    return str(value).strip() if value else None


def _platform_names(ability):
    """Platforms of an ability, lower case. Handles both the 'platforms' dict and the 'executors' list.

    Some files list several platforms under one key, e.g. 'darwin,linux', so keys are split on commas.
    """
    keys = []
    platforms = ability.get('platforms')
    if isinstance(platforms, dict):
        keys.extend(platforms.keys())
    for executor in ability.get('executors') or []:
        if isinstance(executor, dict):
            keys.append(executor.get('platform'))
    names = set()
    for key in keys:
        for part in str(key or '').split(','):
            part = part.strip().lower()
            if part:
                names.add(part)
    return names


def load_adversary(path):
    """Read one adversary profile. Returns (profile name, ability ids in atomic order)."""
    path = Path(path)
    data = yaml.load(path.read_text(encoding='utf-8'), Loader=_YamlLoader)
    if not isinstance(data, dict):
        raise ValueError(f'{path}: not an adversary profile')
    ordering = data.get('atomic_ordering') or []
    if not isinstance(ordering, list):
        raise ValueError(f'{path}: atomic_ordering is not a list')
    name = str(data.get('name') or path.stem).strip()
    return name, [str(item).strip() for item in ordering]


def load_ability_index(root):
    """Index every ability under the given folder: ability_id -> {name, tactic, technique_id, platforms}.

    Only YAML files below a 'data/abilities' folder are read. The tactic is stored normalised.
    If an id appears in more than one file, the first file in sorted path order wins.
    """
    root = Path(root)
    index = {}
    for path in sorted(root.rglob('*.yml')):
        if not _is_ability_file(path):
            continue
        try:
            data = yaml.load(path.read_text(encoding='utf-8'), Loader=_YamlLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f'{path}: cannot parse ability file: {exc}') from exc
        entries = data if isinstance(data, list) else [data]
        for ability in entries:
            if not isinstance(ability, dict) or not ability.get('id'):
                continue
            ability_id = str(ability['id']).strip()
            if ability_id in index:
                continue
            index[ability_id] = {
                'name': str(ability.get('name') or '').strip(),
                'tactic': _norm_tactic(ability.get('tactic')),
                'technique_id': _technique_id(ability),
                'platforms': sorted(_platform_names(ability)),
            }
    return index


def forbidden_reason(ability):
    """Return why an ability is forbidden, or None if it is not. Takes one entry of the ability index."""
    tactic = _norm_tactic(ability.get('tactic'))
    if tactic in FORBIDDEN_TACTICS:
        return f'tactic {tactic}'
    name = str(ability.get('name') or '').lower()
    for word in FORBIDDEN_NAME_WORDS:
        if word.lower() in name:
            return f'name contains "{word}"'
    return None


def allowed_actions(profile_path, platform, allowed_tactics, index):
    """Ability ids of one profile that may be used, in profile order without repeats.

    An ability is allowed when it is in the index, its platform matches, its tactic is in allowed_tactics,
    and forbidden_reason() is None. Abilities missing from the index are refused, because their platform
    and tactic cannot be checked.
    """
    wanted_platform = str(platform).strip().lower()
    tactics = {_norm_tactic(tactic) for tactic in allowed_tactics}
    _, ability_ids = load_adversary(profile_path)
    allowed = []
    for ability_id in ability_ids:
        if ability_id in allowed:
            continue
        ability = index.get(ability_id)
        if ability is None:
            continue
        if wanted_platform not in ability['platforms']:
            continue
        if ability['tactic'] not in tactics:
            continue
        if forbidden_reason(ability) is not None:
            continue
        allowed.append(ability_id)
    return allowed
