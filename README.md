# RED_AGENT

Red agent for CAGE-4 (CybORG): **RL (multi-head DDQN) > Confidence Gate > LLM > Acceptance Gate**.
Runs from a terminal menu. There is no dashboard, no training code, and no connection to Caldera.

## Folder layout

```
RED_AGENT/
├── run_menu.py            terminal menu (start here)
├── README.md
├── requirements.txt
├── configs/               gate lock, seed protocol, Modelfile for the qwen3-red LLM
├── checkpoints/           general_seed42_final.pt (general RL agent)
├── runs/                  evaluation results (created by the menu)
├── tests/test_smoke.py    smoke test
├── third_party/cyborg/    CAGE-4 simulator (CybORG)
└── red/                   the agent package
    ├── config.py          LLM settings (environment variables)
    ├── simulator/         CAGE-4 environment, episode loop, choice contract
    ├── agents/            RL agent (multi-head DDQN), networks, replay buffers, stage specialists, Confidence Gate
    ├── router/            stage router and game-state snapshot
    ├── llm/               prompt, LLM clients (Ollama, stub), Acceptance Gate
    ├── pipeline/          RL > gate > LLM chooser and its wiring
    └── evaluation/        evaluator, seed protocol, metrics
```

## Setup (Linux, for example Kali)

```bash
python3.10 -m venv .venv                  # Python 3.10 (3.11 should also work)
source .venv/bin/activate
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu   # CPU build; drop the index for a GPU build
pip install -r requirements.txt
python -m unittest tests.test_smoke -v    # quick check that the package works on this machine
python run_menu.py
```

Windows: `python -m venv .venv`, then `.venv\Scripts\activate`, then the same `pip` lines.

## LLM (optional, on-prem)

1. Install the Ollama app.
2. Download `Qwen3-14B-UD-Q2_K_XL.gguf` from `unsloth/Qwen3-14B-GGUF` (about 6.07 GB) into `configs/`.
   Digest of the model blob used on this PC: `sha256 6abefcfb88de4157acc6109a711af65fdb5a6da2a77a1ac91bdada2ded22fd49`.
3. In `configs/`: `ollama create qwen3-red -f qwen3-red.Modelfile`
4. In the menu, option `o` checks that the model is found.

## Menu

| Option | What it runs | Status |
|---|---|---|
| a | RL-only (no LLM), quick validation subset | ready |
| b | RL + on-prem LLM, quick validation subset (also the on-prem-only and single-RL run) | ready |
| c | LLM-only: the LLM chooses every step among the same legal options; no RL pick. If the answer is rejected, the agent takes `Sleep`. Needs Ollama | ready |
| d | API LLM only | needs an API client |
| e | On-prem to API assistance | needs an API client |
| f | Multi-RL + LLM (Recon, Advance, Impact specialists) | needs trained specialists |
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
- LLM outputs are not always identical between runs, even with the same seed and code. Example: seed 30000 gave return 126.45 in one validation run and 117.45 in another. The cause is not confirmed.
- Gate values (theta 0.85, delta 0.8) are in `configs/gate_lock_2026-11-18.json`.
