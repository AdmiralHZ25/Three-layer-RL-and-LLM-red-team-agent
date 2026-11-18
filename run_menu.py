"""RED_AGENT evaluation menu: RL > Confidence Gate > LLM, validation seeds only.

Run from this folder:  python run_menu.py
No dashboard and no training. Caldera is not used. Test-split seeds (40000-40499) are refused.
"""
import csv
import json
import statistics
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / 'checkpoints'
GENERAL_CKPT = CKPT_DIR / 'general_seed42_final.pt'
SPECIALIST_CKPTS = {stage: CKPT_DIR / f'{stage}.pt' for stage in ('recon', 'advance', 'impact')}
GATE_LOCK = ROOT / 'configs' / 'gate_lock_2026-11-18.json'
PROTOCOL = ROOT / 'configs' / 'protocol_2026-11-12.json'
RUNS = ROOT / 'runs' / 'pipeline_eval'
OLLAMA = 'http://127.0.0.1:11434'
MODEL = 'qwen3-red'
QUICK_SEEDS = '30000-30009'   # quick validation subset; kept out of the menu text on purpose

MENU = """
==================================================
   RED AGENT - EVALUATION
   RL > Confidence Gate > LLM   (validation seeds only)
==================================================
Decision frameworks
  a. RL-only                                   [ready]
  b. RL + on-prem LLM                          [ready]
  c. LLM-only                                  [ready]
On-prem vs API LLM
  d. API LLM only                              [needs API client]
  e. On-prem to API assistance                 [needs API client]
  (On-prem only is option b.)
Single-RL vs Multi-RL
  f. Multi-RL + LLM (Recon, Advance, Impact)   [needs trained specialists]
  (Single-RL is option b.)
Results and tools
  r. Results summary
  s. System check
  o. Ollama check
  x. Custom run
  q. Quit
"""

NOT_READY = {
    'd': 'API LLM is not wired yet. Needed: an API provider, a client in red/llm, and your '
         'approval before any briefing is sent to an external service.',
    'e': 'On-prem to API assistance is not wired yet. Same requirements as option d.',
}


def check_install():
    ok = True
    print(f'Python {sys.version.split()[0]}')
    if sys.version_info < (3, 10):
        ok = False
        print('  [FAIL] Python 3.10 or newer is required')
    files = (('general checkpoint', GENERAL_CKPT), ('gate lock', GATE_LOCK), ('seed protocol', PROTOCOL),
             ('CybORG', ROOT / 'third_party' / 'cyborg' / 'CybORG' / '__init__.py'))
    for label, path in files:
        found = path.exists()
        ok &= found
        print(f"  [{'OK' if found else 'MISSING'}] {label}: {path.relative_to(ROOT)}")
    for mod in ('numpy', 'torch'):
        try:
            __import__(mod)
            print(f'  [OK] {mod}')
        except ImportError:
            ok = False
            print(f'  [MISSING] {mod} (run: pip install -r requirements.txt)')
    try:
        import red.evaluation.pipeline_eval  # noqa: F401
        print('  [OK] evaluation module imports')
    except Exception as exc:  # show the real reason to the user
        ok = False
        print(f'  [FAIL] evaluation module: {type(exc).__name__}: {exc}')
    print('Ready to run.' if ok else 'Not ready yet. Fix the items above first.')
    return ok


def check_ollama():
    try:
        with urllib.request.urlopen(OLLAMA + '/api/tags', timeout=5) as r:
            names = [m['name'] for m in json.load(r).get('models', [])]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f'Ollama is not reachable at {OLLAMA}: {exc}')
        return False
    has_model = any(n.split(':')[0] == MODEL for n in names)
    print(f"Ollama OK. Model {MODEL}: {'found' if has_model else 'NOT FOUND'}")
    return has_model


def run_eval(seeds, use_llm, tag, specialists=False, llm_only=False):
    if use_llm and not check_ollama():
        print('LLM run cancelled: Ollama or the model is not ready.')
        return
    checkpoints = [f'general={GENERAL_CKPT}']
    if specialists:
        missing = [str(p.relative_to(ROOT)) for p in SPECIALIST_CKPTS.values() if not p.exists()]
        if missing:
            print('Multi-RL needs these trained checkpoints, which are missing: ' + ', '.join(missing))
            return
        checkpoints += [f'{stage}={path}' for stage, path in SPECIALIST_CKPTS.items()]
    from red.evaluation import pipeline_eval as pe
    out = RUNS / f'{tag}_{datetime.now():%Y%m%d-%H%M%S}'
    argv = []
    for item in checkpoints:
        argv += ['--checkpoint', item]
    argv += ['--seeds', seeds, '--output-dir', str(out), '--gate-lock', str(GATE_LOCK), '--protocol', str(PROTOCOL)]
    if use_llm:
        argv += ['--llm', 'ollama']
    if llm_only:
        argv += ['--llm-only']
    print(f'Output folder: {out}')
    try:
        pe.main(argv)
    except SystemExit as exc:  # pipeline_eval stops with a message, e.g. a test-split seed was refused
        print(f'Stopped: {exc.code}')
        return
    print('Done. Open option r for the results summary.')


def custom():
    seeds = input('Seeds (e.g. 30000-30009 or 30000,30005; Enter = quick subset): ').strip() or QUICK_SEEDS
    use_llm = input('Use the on-prem LLM? (y/n, Enter = n): ').strip().lower() == 'y'
    tag = ''.join(c for c in (input('Run name (Enter = custom): ').strip() or 'custom') if c.isalnum() or c in '-_')
    run_eval(seeds, use_llm, tag or 'custom')


def mean_time_to_impact(csv_path):
    values = []
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['impact_success'] == '1' and row.get('time_to_impact'):
                values.append(float(row['time_to_impact']))
    return statistics.mean(values) if values else None


def median_llm_latency(decisions_path):
    latencies = []
    if decisions_path.exists():
        with open(decisions_path, encoding='utf-8') as f:
            for line in f:
                for call in json.loads(line).get('llm_calls', []):
                    latencies.append(call['latency_s'])
    return statistics.median(latencies) if latencies else None


def fmt(value, spec):
    return '-' if value is None else format(value, spec)


def summary():
    files = sorted(RUNS.glob('*/evaluation_summary.json'))
    if not files:
        print('No results yet in runs/pipeline_eval.')
        return
    print(f"{'run':<42}{'eps':>5}{'impact%':>9}{'TTI':>7}{'LLM%':>7}{'LLM s':>8}  restr/foot/priv")
    for f in files:
        s = json.loads(f.read_text(encoding='utf-8'))
        run = f.parent
        episodes = s['episodes'] or 1
        by_source = s.get('decisions_by_source', {})
        total = sum(by_source.values()) or 1
        tti = mean_time_to_impact(run / 'evaluation_metrics.csv')
        latency = median_llm_latency(run / 'decisions.jsonl')
        print(f"{run.name:<42}{s['episodes']:>5}{100 * s['impact_success'] / episodes:>9.1f}"
              f"{fmt(tti, '.1f'):>7}{100 * by_source.get('llm', 0) / total:>7.1f}{fmt(latency, '.2f'):>8}"
              f"  {s['restricted']}/{s['foothold']}/{s['privilege']}")
    print('impact% = impact success rate. TTI = mean time-to-impact in ticks, over episodes with impact.')
    print('LLM% = share of decisions made by the LLM. LLM s = median LLM latency in seconds.')
    print('restr/foot/priv = number of episodes that reached each stage.')


ACTIONS = {
    'a': lambda: run_eval(QUICK_SEEDS, False, 'rl_only'),
    'b': lambda: run_eval(QUICK_SEEDS, True, 'rl_llm_onprem'),
    'c': lambda: run_eval(QUICK_SEEDS, True, 'llm_only', llm_only=True),
    'f': lambda: run_eval(QUICK_SEEDS, True, 'multi_rl_llm', specialists=True),
    'r': summary,
    's': check_install,
    'o': check_ollama,
    'x': custom,
}


def main():
    while True:
        print(MENU)
        try:
            choice = input('Choice: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice == 'q':
            return
        if choice in NOT_READY:
            print(NOT_READY[choice])
        elif choice in ACTIONS:
            ACTIONS[choice]()
        else:
            print('Unknown choice.')
            continue
        try:
            input('\nPress Enter to return to the menu...')
        except (EOFError, KeyboardInterrupt):
            print()
            return


if __name__ == '__main__':
    main()
