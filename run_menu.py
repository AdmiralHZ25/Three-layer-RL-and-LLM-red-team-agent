"""RED_AGENT evaluation menu: RL > Confidence Gate > LLM, validation seeds only.

Run from this folder:  python run_menu.py
No dashboard and no training. Caldera options: k (read-only check), p (plan one step, nothing sent),
g (the RED agent chooses each step itself, inside the scope you set).
Test-split seeds (40000-40499) are refused.
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
QUICK_SEEDS = '30000-30009'   # quick validation subset; kept out of the menu text on purpose

MENU = """
==================================================
  RED AGENT
  Ketik satu huruf, lalu Enter.
==================================================

  SIMULASI  (di simulator, tidak menyentuh VM)
    a  RL saja
    b  RL + LLM lokal
    c  LLM saja
    x  Run kustom

  EMULASI CALDERA  (VM lab saja, server Caldera harus jalan)
    g  Serang otomatis: agen RED memilih aksi sendiri

  CEK
    k  Cek server dan daftar agent (hanya baca)
    r  Ringkasan hasil
    s  Cek instalasi
    o  Cek Ollama (LLM lokal)
    q  Keluar
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


def ollama_endpoint():
    """The same address the LLM client uses (red/llm/factory.py): RED_LLM_ENDPOINT, else this PC."""
    from red import config as C
    return (C.LLM_ENDPOINT or 'http://127.0.0.1:11434').split('/v1')[0].rstrip('/')


def check_ollama():
    from red import config as C
    endpoint = ollama_endpoint()
    model = C.LLM_MODEL
    try:
        with urllib.request.urlopen(endpoint + '/api/tags', timeout=5) as r:
            names = [m['name'] for m in json.load(r).get('models', [])]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f'Ollama is not reachable at {endpoint}: {exc}')
        return False
    has_model = any(n.split(':')[0] == model for n in names)
    print(f"Ollama OK at {endpoint}. Model {model}: {'found' if has_model else 'NOT FOUND'}")
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
    'k': lambda: caldera_check(),
    'g': lambda: caldera_run(),
}


CALDERA_REQUIRED = ('CALDERA_URL', 'CALDERA_API_KEY', 'CALDERA_PAW', 'CALDERA_LAB_HOSTS')


def caldera_settings_status():
    """Print the Caldera settings (the API key value is never printed). Returns the names that are not set."""
    from red import config as C
    print(f'  CALDERA_URL                    {C.CALDERA_URL}')
    print(f"  CALDERA_API_KEY                {'set' if C.CALDERA_API_KEY else 'NOT SET'} (value is never shown)")
    print(f"  CALDERA_PAW                    {C.CALDERA_PAW or 'NOT SET'}")
    print(f"  CALDERA_LAB_HOSTS              {', '.join(C.CALDERA_LAB_HOSTS) or 'NOT SET'}")
    ids = (C.CALDERA_ADVERSARY_ID, C.CALDERA_PLANNER_ID, C.CALDERA_SOURCE_ID)
    print(f"  adversary / planner / source   {'all set' if all(ids) else 'some empty (ad-hoc)'}")
    values = {'CALDERA_URL': C.CALDERA_URL, 'CALDERA_API_KEY': C.CALDERA_API_KEY, 'CALDERA_PAW': C.CALDERA_PAW,
              'CALDERA_LAB_HOSTS': C.CALDERA_LAB_HOSTS}
    return [name for name in CALDERA_REQUIRED if not values[name]]


def caldera_check():
    """Read-only: settings, the agents Caldera lists, and whether CALDERA_PAW passes the lab-host check."""
    from red import config as C
    from red.caldera.client import CalderaClient, CalderaError
    print('Caldera settings:')
    caldera_settings_status()
    if not C.CALDERA_API_KEY:
        print('Set CALDERA_API_KEY first. Nothing was contacted.')
        return
    client = CalderaClient()
    try:
        agents = client.list_agents()
    except CalderaError as exc:
        print(f'Caldera did not answer: {exc}')
        return
    print(f'Agents on the server: {len(agents)}')
    for a in agents:
        mark = '   <- CALDERA_PAW' if a.get('paw') == C.CALDERA_PAW else ''
        print(f"  paw={a.get('paw')}  host={a.get('host')}  platform={a.get('platform')}{mark}")
    if C.CALDERA_PAW:
        try:
            client.check_paw(C.CALDERA_PAW)
            print('CALDERA_PAW passes the lab-host check.')
        except CalderaError as exc:
            print(f'CALDERA_PAW check failed: {exc}')


def _choose_agent(client):
    """Step 1: list the agents on the server and let the operator pick one. Returns the agent or None."""
    print('\nLangkah 1 dari 4: pilih agent.')
    agents = client.list_agents()
    if not agents:
        print('Server belum punya agent. Pasang agent di VM dulu (perintah deploy dari GUI Caldera).')
        return None
    for i, a in enumerate(agents, 1):
        print(f"  {i:>2}. {a.get('host')}   platform={a.get('platform')}   status={a.get('status')}   "
              f"trusted={a.get('trusted')}   privilege={a.get('privilege')}")
    text = input('Nomor agent (Enter = batal): ').strip()
    if not text:
        return None
    if not (text.isdigit() and 1 <= int(text) <= len(agents)):
        print('Nomor tidak ada di daftar.')
        return None
    agent = agents[int(text) - 1]
    if not agent.get('paw'):
        print('Agent ini tidak punya paw. Pilih agent lain.')
        return None
    if agent.get('status') != 'alive':
        print('Peringatan: agent ini belum alive. Operasi akan menunggu sampai agent hidup.')
    return agent


def _ask_objective():
    """Ask for the objective in words. Enter = cancel. Returns the text or None."""
    print('\nTujuan operasi, dalam bahasa biasa.')
    print('Contoh: Temukan domain controller dan satu user yang bisa dipakai di jaringan lab.')
    text = input('Tujuan (Enter = batal): ').strip()
    return text or None


def _load_rl_chooser():
    """Load the RL agent and return an RlChooser, or None if unavailable."""
    if not GENERAL_CKPT.exists():
        print(f'Checkpoint tidak ditemukan: {GENERAL_CKPT.relative_to(ROOT)}. Pakai mode LLM saja.')
        return None
    if not GATE_LOCK.exists():
        print(f'Gate lock tidak ditemukan: {GATE_LOCK.relative_to(ROOT)}. Pakai mode LLM saja.')
        return None
    try:
        from red.agents.rl import MultiHeadAgent
        from red.caldera.choosers import RlChooser, read_gate_lock
        agent = MultiHeadAgent.load(GENERAL_CKPT)
        agent.online.eval()
        theta, delta = read_gate_lock(GATE_LOCK)
        print(f'RL agent dimuat dari {GENERAL_CKPT.name}. Gate: tau≥{theta}, sigma≤{delta}.')
        return RlChooser(agent, theta, delta)
    except Exception as exc:
        print(f'RL agent gagal dimuat: {exc}. Pakai mode LLM saja.')
        return None


def caldera_run():
    """Agentic operation. The person chooses the agent and writes the objective. The LLM plans and chooses each step
    from what the server offers, inside the scope. One confirmation (1) starts the run. Nothing is asked during the run."""
    from red import config as C
    from red.caldera.action_scope import FORBIDDEN_NAME_WORDS, FORBIDDEN_TACTICS
    from red.caldera.agent_loop import run_agentic
    from red.caldera.client import CalderaClient, CalderaError
    from red.caldera.run_plan import check_run_settings
    from red.caldera.scope import ScopeViolation
    from red.llm import make_llm
    print('\n=== OPERASI AGENTIK: agen RED memilih aksi sendiri ===')
    print('Server Caldera harus sudah jalan.')
    if not C.CALDERA_URL or not C.CALDERA_API_KEY:
        print('Set dulu CALDERA_URL dan CALDERA_API_KEY di terminal ini. Tidak ada yang dihubungi.')
        return
    try:
        check_run_settings(C.CALDERA_RUN_MAX_STEPS, C.CALDERA_RUN_MAX_SECONDS, C.CALDERA_MAX_CONSECUTIVE_FAILURES)
    except ValueError as exc:
        print(f'Batas di config tidak sah: {exc}. Tidak ada yang dihubungi.')
        return
    if not check_ollama():
        print('Dibatalkan: Ollama atau model belum siap. Cek dengan pilihan o.')
        return

    print('\nMode pemilihan aksi:')
    print('  1. LLM saja  — Ollama merencanakan dan memilih setiap langkah')
    print('  2. RL + LLM  — RL memilih satu aksi bila yakin (gate lulus), LLM melengkapi sisa batch')
    mode_input = input('Mode (Enter = 1): ').strip() or '1'
    use_rl = mode_input == '2'
    rl_chooser = _load_rl_chooser() if use_rl else None
    if use_rl and rl_chooser is None:
        use_rl = False   # fell back to LLM-only due to load error

    client = CalderaClient()
    try:
        agent = _choose_agent(client)
        if agent is None:
            print('Dibatalkan. Tidak ada yang dikirim.')
            return
        objective = _ask_objective()
        if objective is None:
            print('Dibatalkan. Tidak ada yang dikirim.')
            return
        run_root = ROOT / 'runs' / 'caldera'
        mode_label = 'RL + LLM (gate: RL memilih dulu bila yakin)' if use_rl else f'LLM saja ({C.LLM_MODEL})'
        print('\nRingkasan')
        print(f"  agent     : {agent['host']}   (paw {agent['paw']})")
        print(f'  tujuan    : {objective}')
        print(f'  mode      : {mode_label}')
        print(f"  batas     : {C.CALDERA_RUN_MAX_STEPS} langkah, {int(C.CALDERA_RUN_MAX_SECONDS)} detik, "
              f'berhenti setelah {C.CALDERA_MAX_CONSECUTIVE_FAILURES} gagal berturut-turut   (dari config)')
        blocked_tactics = ', '.join(sorted(FORBIDDEN_TACTICS)) or 'tidak ada'
        print(f"  blokir    : taktik {blocked_tactics}; nama yang memuat {', '.join(FORBIDDEN_NAME_WORDS)}")
        print(f"  berhenti  : buat file {run_root / 'STOP'}")
        print('Agen memilih dan mengirim sendiri. Tidak ada konfirmasi per langkah.')
        if input('Ketik 1 untuk jalankan, Enter untuk batal: ').strip() != '1':
            print('Dibatalkan. Tidak ada yang dikirim.')
            return
        # Used only for this run: the check in client.check_paw reads these settings.
        C.CALDERA_PAW = agent['paw']
        C.CALDERA_LAB_HOSTS = (agent['host'],)
        client.check_paw(C.CALDERA_PAW)
        llm = make_llm()
        run_dir = run_root / datetime.now().strftime('%Y%m%d-%H%M%S')
        print('\nMulai. Setiap langkah dicetak di bawah.')
        summary = run_agentic(client, llm, objective, C.CALDERA_PAW, C.CALDERA_LAB_HOSTS, run_dir,
                              stop_file=run_root / 'STOP', max_steps=C.CALDERA_RUN_MAX_STEPS,
                              max_seconds=C.CALDERA_RUN_MAX_SECONDS,
                              max_failures=C.CALDERA_MAX_CONSECUTIVE_FAILURES,
                              rl_chooser=rl_chooser)
        if summary['objective_reached']:
            print(f"\nSelesai. Tujuan tercapai. Bukti dari output: {summary['evidence']}")
        else:
            print(f"\nSelesai. Tujuan belum tercapai. Langkah terkirim: {summary['steps_sent']}. "
                  f"Alasan berhenti: {summary['stop']}.")
        print(f'Log: {run_dir}')
        print('Operasi di Caldera sudah ditutup. Caldera bisa menjalankan cleanup untuk aksi yang sudah dijalankan.')
    except (CalderaError, ScopeViolation, ValueError) as exc:
        print(f'Berhenti: {exc}')
    except KeyboardInterrupt:
        print('\nDihentikan.')


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
