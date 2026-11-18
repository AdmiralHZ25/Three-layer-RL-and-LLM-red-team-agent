# RED_AGENT

Red agent for CAGE-4 (CybORG): **RL (multi-head DDQN) > Confidence Gate > LLM > Acceptance Gate**.
Runs from a terminal menu. There is no dashboard and no training code.
Caldera (Phase 2, lab only) is used only for read-only checks and for planning one step. Nothing is sent to a target from the menu.

## Folder layout

```
RED_AGENT/
├── run_menu.py            terminal menu (start here)
├── README.md
├── requirements.txt
├── configs/               gate lock, seed protocol, Modelfile for the qwen3-red LLM
├── checkpoints/           general_seed42_final.pt (general RL agent)
├── runs/                  evaluation results (created by the menu)
├── tests/                 test_smoke.py, test_caldera.py (offline)
├── third_party/cyborg/    CAGE-4 simulator (CybORG)
└── red/                   the agent package
    ├── config.py          LLM and Caldera settings (environment variables)
    ├── simulator/         CAGE-4 environment, episode loop, choice contract
    ├── agents/            RL agent (multi-head DDQN), networks, replay buffers, stage specialists, Confidence Gate
    ├── router/            stage router and game-state snapshot
    ├── llm/               prompt, LLM clients (Ollama, stub), Acceptance Gate
    ├── pipeline/          RL > gate > LLM chooser and its wiring
    ├── evaluation/        evaluator, seed protocol, metrics
    └── caldera/           Caldera REST client, link selection, human approval gate (Phase 2)
```

## Setup (Linux, for example Kali)

```bash
python3.10 -m venv .venv                  # Python 3.10 (3.11 should also work)
source .venv/bin/activate
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu   # CPU build; drop the index for a GPU build
pip install -r requirements.txt
python -m unittest tests.test_smoke tests.test_caldera -v
python run_menu.py
```

Windows: `python -m venv .venv`, then `.venv\Scripts\activate`, then the same `pip` lines.

## LLM (optional, on-prem)

1. Install the Ollama app.
2. Download `Qwen3-14B-UD-Q2_K_XL.gguf` from `unsloth/Qwen3-14B-GGUF` (about 6.07 GB) into `configs/`.
   Digest of the model blob used on this PC: `sha256 6abefcfb88de4157acc6109a711af65fdb5a6da2a77a1ac91bdada2ded22fd49`.
3. In `configs/`: `ollama create qwen3-red -f qwen3-red.Modelfile`
4. In the menu, option `o` checks that the model is found.

## Caldera (Phase 2, lab only)

Caldera runs on its own machine (for example the Kali VM) and the agent talks to it over REST.
Settings come only from environment variables. The API key is never written to a file.

| Variable | What it is | Where to get it |
|---|---|---|
| `CALDERA_URL` | Caldera server address, for example `http://192.168.56.20:8888` | the host-only address of the Caldera server |
| `CALDERA_API_KEY` | Caldera API key (the `red` key) | `api_key_red` in Caldera's `conf/local.yml`. Do not paste it into files or commits |
| `CALDERA_PAW` | Agent id (paw) of the lab VM | GET `/api/v2/agents`, or option `k` |
| `CALDERA_LAB_HOSTS` | Comma-separated lab host names allowed for the paw | your lab VM names |
| `CALDERA_ALLOWED_ABILITY_IDS` | Comma-separated ability ids allowed | the abilities you choose in Caldera. Empty = nothing may run |
| `CALDERA_ADVERSARY_ID`, `CALDERA_PLANNER_ID`, `CALDERA_SOURCE_ID` | Optional. Empty = ad-hoc (the planner adds no actions) | Caldera's API |

Menu options:
- `k` **read-only check**: prints the settings (never the key), lists the agents on the server, and checks `CALDERA_PAW` against the lab hosts.
- `p` **plan one step, no execution**: creates a temporary operation, lists the allowed abilities for the agent, lets the LLM choose one, and finishes the operation again. The chosen ability is printed. Nothing is sent to the target.

Execution (`execute=True` in `red/caldera/executor.py`) needs a live client and a human approval for each action. It is not available from the menu yet.

## Menu

| Option | What it runs | Status |
|---|---|---|
| a | RL-only (no LLM), quick validation subset | ready |
| b | RL + on-prem LLM, quick validation subset (also the on-prem-only and single-RL run) | ready |
| c | LLM-only: the LLM chooses every step among the same legal options; no RL pick. If the answer is rejected, the agent takes `Sleep`. Needs Ollama | ready |
| d | API LLM only | needs an API client |
| e | On-prem to API assistance | needs an API client |
| f | Multi-RL + LLM (Recon, Advance, Impact specialists) | needs trained specialists |
| k | Caldera read-only check | ready (needs Caldera settings) |
| p | Caldera plan one step, no execution | ready (needs Caldera settings) |
| r | Results summary: impact rate, time-to-impact, LLM share, LLM latency, stage counts | ready |
| s | System check | ready |
| o | Ollama check | ready |
| x | Custom run: choose seeds and whether to use the LLM | ready |
| q | Quit | |

The quick validation subset is 10 seeds from the validation range. Seeds 40000-40499 (test split) are refused.
Results are saved under `runs/pipeline_eval/<name>_<time>/`. Existing results are never overwritten.

## Limits

- Multi-RL (option f) needs recon and impact specialists that were actually trained. The ones in the training runs have 0 updates, so they are not used.
- API options (d, e) need an API client and your approval before any briefing is sent to an external service.
- Caldera execution from the menu is not wired. The approval gate exists in code and is tested offline.
- The planning step (option p) still has a fallback: if the LLM gives no usable answer, the first allowed ability is taken. Check the LLM first (option o).
- LLM outputs are not always identical between runs, even with the same seed and code. Example: seed 30000 gave return 126.45 in one validation run and 117.45 in another. The cause is not confirmed.
- Gate values (theta 0.85, delta 0.8) are in `configs/gate_lock_2026-11-18.json`.
