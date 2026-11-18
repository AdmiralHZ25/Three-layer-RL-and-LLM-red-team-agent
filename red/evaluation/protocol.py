"""Experiment protocol (configs/protocol_*.json): load, check seed ranges, build per-run CLI flags.

Pure data handling; no simulator, no torch.
"""
import json
from pathlib import Path

EVAL_RANGES = ('validation_eval', 'legacy_test', 'new_test')


def load(path):
    """Read a protocol JSON file into a plain dict."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate(protocol):
    """Raise ValueError unless every training scenario block is disjoint from all eval/test ranges
    (and from the other training blocks). Returns plain dicts describing the checked ranges."""
    train = protocol['training']
    seeds, offsets, guard = train['seeds'], train['scenario_seed_offsets'], train['max_episodes']
    errors = []
    if len(seeds) != len(offsets) or not seeds:
        errors.append('training.seeds and training.scenario_seed_offsets must be non-empty and equally long')
    if guard < 1:
        errors.append('training.max_episodes must be >= 1')
    blocks = {f'train_{k}': (int(o), int(o) + int(guard) - 1) for k, o in enumerate(offsets)}
    ranges = {name: (int(protocol[name]['first']), int(protocol[name]['last'])) for name in EVAL_RANGES}
    for name, (lo, hi) in ranges.items():
        if lo > hi:
            errors.append(f'{name}: first > last')
    named = {**blocks, **ranges}
    names = list(named)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (alo, ahi), (blo, bhi) = named[a], named[b]
            if alo <= bhi and blo <= ahi:
                errors.append(f'{a} {named[a]} overlaps {b} {named[b]}')
    if errors:
        raise ValueError('Invalid protocol: ' + '; '.join(errors))
    return {'ok': True, 'training_blocks': {k: list(v) for k, v in blocks.items()},
            'eval_ranges': {k: list(v) for k, v in ranges.items()}}


def run_flags(protocol, k):
    """CLI args (list of str) for training run index k: --seed and --scenario-seed-offset."""
    train = protocol['training']
    return ['--seed', str(train['seeds'][k]), '--scenario-seed-offset', str(train['scenario_seed_offsets'][k])]



def check_overlap(base, completed, eval_seed, eval_episodes):
    """Raise if training seeds base..base+completed-1 intersect held-out seeds eval_seed..+eval_episodes-1."""
    if base < eval_seed + eval_episodes and eval_seed < base + completed:
        raise ValueError(f'Training scenario seeds {base}..{base + completed - 1} overlap the held-out '
                         f'evaluation seeds {eval_seed}..{eval_seed + eval_episodes - 1}.')
