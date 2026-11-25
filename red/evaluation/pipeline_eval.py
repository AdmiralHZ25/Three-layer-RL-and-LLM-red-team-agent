r"""Offline-built evaluator for the full decision order: RL specialist (Router) > Confidence Gate > LLM > Acceptance Gate.

Wires the existing `build_rl_chooser` (StagedHybridChooser) into the instrumented `episode()` loop against
Blue = cc4BlueRandomAgent, one episode per seed, and logs EVERY decision. Gate values are read from the lock
file (default configs/gate_lock_2026-11-18.json); they are never hard-coded here.

Seeds inside the protocol's new_test range are refused unless --open-new-test is passed (open it once only).
Without --llm the LLM fallback is off and a failed gate falls back to the best RL pick (source 'cadangan').

Run from the RED_AGENT folder:
  python -B -m red.evaluation.pipeline_eval --checkpoint general=checkpoints/general_seed42_final.pt --seeds 30000-30199 --output-dir runs/pipeline_eval/<new_name> [--llm ollama] [--llm-only]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from red.agents.stages import StageSpecialists, max_q_stage_provider
from red.llm.prompt import GOAL_TEXT
from red.pipeline.wiring import build_rl_chooser, load_agents
from red.simulator import cage4 as v3
from red.simulator.episode import episode
from red.evaluation import protocol as protocol_io
from red.evaluation.metrics import _CsvAppender, finalize_row, summarize, write_json
from red.router.snapshot import router_stage_provider
from red.router.stage_router import StageRouter

ROOT = Path(__file__).resolve().parents[2]
POLICY = 'pipeline'
_SOURCE = {'rl': 'rl', 'llm': 'llm', 'api': 'llm', 'fallback': 'cadangan', 'sleep': 'sleep'}


def parse_seeds(spec):
    """'30000-30002,30010' -> [30000, 30001, 30002, 30010]."""
    if not spec.strip():
        raise ValueError('no seeds given')
    seeds = []
    for part in spec.split(','):
        lo, _, hi = part.strip().partition('-')
        seeds.extend(range(int(lo), int(hi or lo) + 1))
    return seeds


def check_seeds(seeds, protocol, open_new_test=False):
    """Refuse seeds in protocol['new_test'] unless ``open_new_test``."""
    lo, hi = protocol['new_test']['first'], protocol['new_test']['last']
    if not open_new_test and any(lo <= s <= hi for s in seeds):
        raise ValueError(f'seeds in the new_test range {lo}..{hi} are refused; '
                         'the test split is opened once, with --open-new-test only')


def load_gate(path):
    gate = json.loads(Path(path).read_text(encoding='utf-8'))['gate']
    return float(gate['theta']), float(gate['delta'])


class _Recorder:
    """Wraps a chooser; writes one decision record per choose() call."""

    def __init__(self, inner, write, episode_index, seed):
        self.inner, self.write, self.episode_index, self.seed, self.n = inner, write, episode_index, seed, 0

    def choose(self, candidates, knowledge=None, tick=None):
        choice = self.inner.choose(candidates, knowledge, tick)
        m = choice.meta
        self.n += 1
        self.write({'episode': self.episode_index, 'seed': self.seed, 'decision': self.n, 'tick': tick,
                    'stage': m.get('routed_stage'), 'index': choice.index, 'source': _SOURCE[m['layer']],
                    'tau': m['tau'], 'sigma': m['sigma'], 'passed': m['passed'],
                    'llm_calls': m.get('llm_calls', []), 'rejections': m.get('rejections', [])})
        return choice


def evaluate(make_chooser, seeds, runtime, out_dir, log=print, extra=None):
    """Run one episode per seed. ``make_chooser()`` returns a fresh chooser. ``out_dir`` must not exist.
    Writes decisions.jsonl, evaluation_metrics.csv, evaluation_summary.json. Returns the summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows, sources = [], Counter()
    with (out / 'decisions.jsonl').open('x', encoding='utf-8') as dec, \
            (out / 'evaluation_metrics.csv').open('x', newline='', encoding='utf-8') as mf:
        metrics = _CsvAppender(mf)

        def write(rec):
            sources[rec['source']] += 1
            dec.write(json.dumps(rec, allow_nan=False) + '\n')

        for i, seed in enumerate(seeds, 1):
            chooser = _Recorder(make_chooser(), write, i, seed)
            row = episode(seed, runtime, None, 0.0, None, True, 1.0, POLICY, lambda event: None, chooser=chooser)
            finalize_row(row)
            row['episode'] = i
            metrics.write(row)
            rows.append(row)
            log(f"[eval {POLICY}] {i}/{len(seeds)} seed={seed} return={row['episode_return']:.2f} "
                f"impact={row['impact_success']} ticks={row['length_ticks']} reason={row['termination_reason']}")
    summary = summarize(rows)
    summary.update(policy=POLICY, decisions_by_source=dict(sources), seeds=[seeds[0], seeds[-1], len(seeds)], **(extra or {}))
    write_json(out / 'evaluation_summary.json', summary)
    return summary


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True, metavar='STAGE=PATH',
                   help='repeatable; STAGE in recon|advance|impact|general; general is required (empty-mask fallback)')
    p.add_argument('--seeds', required=True, help="e.g. '30000-30199' or '30000,30005-30007'")
    p.add_argument('--output-dir', type=Path, required=True, help='must not exist')
    p.add_argument('--llm', metavar='BACKEND', help='LLM backend for make_llm (stub|ollama|...); omit = LLM off')
    p.add_argument('--llm-only', action='store_true',
                   help='LLM-only framework: the LLM chooses every step (needs --llm); no RL pick is used')
    p.add_argument('--llm-goal', action='store_true',
                   help='put the mission goal (red/llm/prompt.py GOAL_TEXT) in the LLM prompt; omit = original prompt')
    p.add_argument('--stage', help='fixed stage instead of the Stage Router (diagnostic use)')
    p.add_argument('--select', choices=('stage', 'maxq'), default='stage',
                   help="'stage' = Stage Router (or --stage); 'maxq' = specialist with the highest main-head Q, no stage prediction")
    p.add_argument('--gate-lock', type=Path, default=ROOT / 'configs' / 'gate_lock_2026-11-18.json')
    p.add_argument('--protocol', type=Path, default=ROOT / 'configs' / 'protocol_2026-11-12.json')
    p.add_argument('--open-new-test', action='store_true', help='allow seeds in the new_test range (once only)')
    p.add_argument('--threads', type=int, default=1)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    seeds = parse_seeds(a.seeds)
    try:
        check_seeds(seeds, protocol_io.load(a.protocol), a.open_new_test)
    except ValueError as exc:
        raise SystemExit(str(exc))
    import torch
    from red.evaluation.protocol import check_overlap
    torch.set_num_threads(a.threads)
    theta, delta = load_gate(a.gate_lock)
    paths = dict(item.split('=', 1) for item in a.checkpoint)
    if 'general' not in paths:
        raise SystemExit('--checkpoint general=<path> is required')
    agents = load_agents(paths)
    for stage, agent in agents.items():
        extra = agent.checkpoint_extra
        try:
            check_overlap(extra.get('scenario_base', agent.cfg.scenario_base), extra.get('completed_episodes', 0),
                          seeds[0], seeds[-1] - seeds[0] + 1)
        except ValueError as exc:
            raise SystemExit(f'{stage}: {exc}')
    base = agents['general']
    # shortcut: runtime (horizon, reward version) comes from the general checkpoint; upgrade if specialists differ
    runtime = SimpleNamespace(steps=base.checkpoint_extra.get('max_steps', base.cfg.max_steps), weights=v3.Weights(),
                              reward_version=base.cfg.reward_version, mask_last_withdraw=base.cfg.mask_last_withdraw)
    if a.llm_only and not a.llm:
        raise SystemExit('--llm-only needs --llm <backend>')
    llm = None
    if a.llm:
        from red.llm.factory import make_llm
        llm = make_llm(a.llm)
    if a.select == 'maxq':
        if a.stage:
            raise SystemExit('--select maxq cannot be combined with --stage')
        stage = max_q_stage_provider(StageSpecialists(agents))
    else:
        stage = a.stage or router_stage_provider(StageRouter(), runtime.steps)
    goal = GOAL_TEXT if a.llm_goal else None
    summary = evaluate(lambda: build_rl_chooser(agents, stage, theta, delta, llm_local=llm, llm_only=a.llm_only,
                                                prompt_goal=goal),
                       seeds, runtime, a.output_dir,
                       extra={'theta': theta, 'delta': delta, 'llm_only': a.llm_only, 'select': a.select,
                              'llm_goal': a.llm_goal})
    print('EVAL SUMMARY', summary, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
