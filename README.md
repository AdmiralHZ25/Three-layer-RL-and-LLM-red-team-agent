# RED_AGENT

Red agent for CAGE-4 (CybORG). The decision pipeline is **RL (multi-head DDQN) > Confidence Gate > LLM > Acceptance Gate**.

The agent runs from a terminal menu (`run_menu.py`). There is no dashboard and no training code in this repository.

- **Simulation** (options `a`, `b`, `c`, `x`): runs inside the CAGE-4 simulator only. It never touches a VM.
- **Caldera** (options `k`, `g`): Phase 2, lab only. It talks to a Caldera server over REST. Option `k` only reads. Option `g` sends commands to the lab VM, inside the limits described below.

## Folder layout

```
RED_AGENT/
├── run_menu.py            terminal menu (start here)
├── README.md
├── requirements.txt
├── configs/               gate lock, seed protocol, Modelfile for the qwen3-red LLM
├── checkpoints/           general_seed42_final.pt (general RL agent)
├── tests/                 offline tests (unittest); fixtures/ holds small YAML inputs for them
├── third_party/cyborg/    CAGE-4 simulator (CybORG)
└── red/                   the agent package
    ├── config.py          LLM and Caldera settings (environment variables)
    ├── simulator/         CAGE-4 environment, episode loop, choice contract
    ├── agents/            RL agent (multi-head DDQN), networks, replay buffers, stage specialists, Confidence Gate
    ├── router/            stage router and game-state snapshot
    ├── llm/               prompt, LLM clients (Ollama, stub), Acceptance Gate
    ├── pipeline/          RL > gate > LLM chooser and its wiring
    ├── evaluation/        evaluator, seed protocol, metrics
    └── caldera/           Caldera REST client, action scope, planner, agentic loop, memory, profiles, manual approval
```

Folders created when you run the menu (`runs/`) and the local Caldera server copy (`caldera/`) are not in the repository (see `.gitignore`).

## Setup

Linux (for example Kali):

```bash
python3.10 -m venv .venv                  # Python 3.10 (3.11 should also work)
source .venv/bin/activate
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu   # CPU build; drop the index for a GPU build
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
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

Caldera runs on its own machine (for example a Kali VM). The agent talks to it over REST.
Settings come only from environment variables. The API key is never written to a file.

| Variable | What it is | Where to get it |
|---|---|---|
| `CALDERA_URL` | Caldera server address, for example `http://192.168.56.20:8888` | the host-only address of the Caldera server |
| `CALDERA_API_KEY` | Caldera API key (the `red` key) | `api_key_red` in Caldera's `conf/local.yml`. Do not paste it into files or commits |
| `CALDERA_PAW` | Agent id (paw) of the lab VM | GET `/api/v2/agents`, or option `k` |
| `CALDERA_LAB_HOSTS` | Comma-separated lab host names allowed for the paw | your lab VM names |
| `CALDERA_ALLOWED_ABILITY_IDS` | Comma-separated ability ids allowed by the allow-list (used by the older single-step path) | the abilities you choose in Caldera. Empty = nothing may run |
| `CALDERA_ADVERSARY_ID`, `CALDERA_PLANNER_ID`, `CALDERA_SOURCE_ID` | Optional. Empty = ad-hoc. If any of them is set, the operation is refused, because an adversary or planner could run actions outside the allow-list | Caldera's API |
| `CALDERA_RUN_MAX_STEPS` | Step ceiling for option `g` (default 10, range 1 to 20) | your choice |
| `CALDERA_RUN_MAX_SECONDS` | Time ceiling for the whole `g` run (default 1800) | your choice |
| `CALDERA_MAX_CONSECUTIVE_FAILURES` | Stop after this many steps in a row without a successful link (default 3) | your choice |

Menu options:
- `k` **read-only check**: prints the settings (never the key), lists the agents on the server, and checks `CALDERA_PAW` against `CALDERA_LAB_HOSTS`.
- `g` **agentic operation**: you choose the agent and write the objective in words. The LLM (Ollama) writes a short plan, then chooses each step from the abilities the server offers. You type `1` once to start. Nothing is asked during the run. RL is not used on this path.
  - Blocked: abilities whose name contains a blocked word (see `red/caldera/action_scope.py`), abilities with unfilled Caldera fact variables, and abilities that would cut the agent's own control (restart, shutdown, logoff, Wi-Fi off).
  - Every link must be on the lab host and in the list the server offered. If one link is not, nothing in that step is sent.
  - The run stops at the objective (only if the quote is really in the output), at the step or time ceiling, at the STOP file, or when the operation ends.
  - Logs go to `runs/caldera/<time>/`: `steps.jsonl` and `summary.json`.

Stopping early: create an empty file named `STOP` in `runs/caldera/`.

## Menu

| Option | What it runs | Status |
|---|---|---|
| a | RL only, quick validation subset (simulator) | ready |
| b | RL + on-prem LLM, quick validation subset (simulator) | ready |
| c | LLM only: the LLM chooses every step among the same legal options. If the answer is rejected, the agent takes `Sleep`. Needs Ollama | ready |
| x | Custom run: choose seeds and whether to use the LLM | ready |
| g | Caldera agentic operation (LLM plans, inside the limits above). Needs Ollama and Caldera settings | ready |
| k | Caldera read-only check | ready (needs Caldera settings) |
| r | Results summary: impact rate, time-to-impact, LLM share, LLM latency, stage counts | ready |
| s | System check | ready |
| o | Ollama check | ready |
| q | Quit | |

The quick validation subset is 10 seeds from the validation range. Seeds 40000-40499 (test split) are refused.
Results are saved under `runs/pipeline_eval/<name>_<time>/`. Existing results are never overwritten.

## Limits

- Option `g` acts on the lab VM without asking for each step. The only approval is the single `1` at the start. Keep the scope to discovery abilities until you have reviewed the logs.
- Output text from the target is treated as data in every prompt, but the LLM can still be wrong about what it means. The objective check only accepts a quote that appears in the real output.
- The mapping from Caldera techniques to CAGE-4 classes (`red/caldera/mapping.py`) and the 22-feature vector for Caldera (`red/caldera/features.py`) are draft approximations.
- The gate values for Caldera are the locked CAGE-4 values, not calibrated for Caldera.
- LLM outputs are not always identical between runs, even with the same seed and code. Example: seed 30000 gave return 126.45 in one validation run and 117.45 in another. The cause is not confirmed.
- Gate values (theta 0.85, delta 0.8) are in `configs/gate_lock_2026-11-18.json`.
- Options d and e (API LLM) and f (multi-RL with specialists) are not in the menu: the API client is not built, and the trained specialists were not used.
