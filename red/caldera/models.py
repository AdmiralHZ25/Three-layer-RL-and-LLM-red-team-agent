"""Small data structures shared by the Caldera modules."""
from dataclasses import dataclass


@dataclass
class PotentialLink:
    """One candidate action = one ability that can run on one agent (paw)."""
    paw: str
    ability_id: str
    executor: str = "sh"
    command: str = ""              # real command (from executor.command in Caldera); required when posting a link
    tactic: str = ""
    technique_id: str = ""
    name: str = ""
    detectability: str = "medium"   # low/medium/high (from the technique schema)
    target: str = ""                # host of the target; "" for global actions (e.g. recon)
    id: str = ""  # filled with the link id after it is queued


@dataclass
class StepRecord:
    """One decision: which ability was chosen, by which source, and how long it took."""
    t: int
    stage: str
    paw: str
    ability_id: str
    technique_id: str
    source: str        # LLM / fallback
    tau: float         # confidence
    sigma: float       # uncertainty
    decide_s: float    # time spent deciding
    latency_s: float   # time spent executing the link (0 when not executed)
    link_status: str
    reward: float
