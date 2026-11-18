"""A1 — wire the REAL RL pipeline into the agent Worker.

`Worker.execute(intent, candidates)` duck-types a chooser; here we hand it the real
`StagedHybridChooser` (multi-head DDQN specialist + Confidence Gate + optional LLM
fallback) instead of the demo chooser. So the agent shell (R2–R4) now drives the
actual RL instead of a placeholder.

Candidate contract (from `env.cage4_v3.candidates`): each candidate is
`(agent, action, feature_22dim, target)`. The chooser reads `c[2]` for the RL Q-matrix
and `type(c[1]).__name__` for the per-stage mask.

The wiring (`build_rl_chooser`/`build_rl_worker`) needs only numpy. Building a *trained*
agent needs torch + a checkpoint (`load_agents`); that import stays lazy so this module
loads on Kali without torch.
"""
from red.pipeline.chooser import StagedHybridChooser
from red.router.stage_router import fixed_stage
from red.agents.stages import StageSpecialists
from red.router.snapshot import GENERAL


def build_rl_chooser(agents, stage, theta=0.6, delta=0.4, default_stage=GENERAL,
                     llm_local=None, llm_api=None, acceptance=None, main_head=None, llm_only=False):
    """`agents`: dict stage -> MultiHeadDDQNAgent, or a StageSpecialists. An unknown/UNKNOWN stage routes to
    the 'general' agent (`default_stage`); it must be registered, else choose() raises ValueError.
    `stage`: a stage string (constant provider) or a callable stage_provider."""
    provider = fixed_stage(stage) if isinstance(stage, str) else stage
    specialists = agents if isinstance(agents, StageSpecialists) else StageSpecialists(agents)
    return StagedHybridChooser(specialists, provider, theta, delta, default_stage=default_stage,
                               llm_local=llm_local, llm_api=llm_api, acceptance=acceptance,
                               main_head=main_head, llm_only=llm_only)


def load_agents(paths):
    """`paths`: dict stage -> checkpoint path -> dict stage -> MultiHeadDDQNAgent (needs torch)."""
    from red.agents.rl import MultiHeadDDQNAgent
    return {stage: MultiHeadDDQNAgent.load(path) for stage, path in paths.items()}
