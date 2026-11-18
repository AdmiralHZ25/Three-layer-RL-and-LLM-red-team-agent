"""Shared evaluation helpers: episode-row finalizing, JSON and CSV writers, run summaries."""
import csv
import json
import math
from pathlib import Path

import numpy as np

POLICY_LABEL = 'ddqn'  # kept from the original summaries; evaluate() sets its own 'policy' name


def finalize_row(row):
    """Add latency median/p95 and illegal_action_rate to an episode row (existing columns kept).

    illegal_action_rate = (invalid_actions + filtered_actions) / decisions: fraction of selected
    candidates that the simulator turned into InvalidAction or removed before execution. Candidates
    come from the true action mask, so this is expected to be 0; it is measured, not assumed.
    """
    lat = json.loads(row['rl_decision_latencies_ms']) if row.get('rl_decision_latencies_ms') else []
    row['latency_median_ms'] = float(np.median(lat)) if lat else ''
    row['latency_p95_ms'] = float(np.percentile(lat, 95)) if lat else ''
    dec = row.get('decisions', 0)
    row['illegal_action_rate'] = (row['invalid_actions'] + row['filtered_actions']) / dec if dec else 0.0
    return row


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(_clean(obj), f, indent=2, ensure_ascii=False)
        f.write('\n')


class _CsvAppender:
    def __init__(self, stream):
        self.stream, self.writer = stream, None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()


def summarize(rows):
    n = len(rows)
    lat = [x for r in rows for x in json.loads(r['rl_decision_latencies_ms'] or '[]')]
    dec = sum(r['decisions'] for r in rows)

    def mean(key):
        return float(np.mean([float(r[key]) for r in rows])) if n else None
    return {'policy': POLICY_LABEL, 'episodes': n,
            'impact_success': int(sum(r['impact_success'] for r in rows)),
            'mean_return': mean('episode_return'), 'mean_length_ticks': mean('length_ticks'),
            'restricted': int(sum(r['restricted'] for r in rows)),
            'foothold': int(sum(r['foothold'] for r in rows)),
            'privilege': int(sum(r['privilege'] for r in rows)),
            'invalid_actions': int(sum(r['invalid_actions'] for r in rows)),
            'filtered_actions': int(sum(r['filtered_actions'] for r in rows)),
            'illegal_action_rate': (sum(r['invalid_actions'] + r['filtered_actions'] for r in rows) / dec
                                    if dec else 0.0),
            'latency_median_ms': float(np.median(lat)) if lat else None,
            'latency_p95_ms': float(np.percentile(lat, 95)) if lat else None,
            'latency_mean_ms': float(np.mean(lat)) if lat else None,
            'decisions': int(dec)}
