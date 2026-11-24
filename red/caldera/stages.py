"""Stage gate for the Caldera loop. The stages, and the facts that open each one, are data (STAGES), not code.

Stages come in order. A stage is done when every requirement in its 'done_when' is met (facts.goal_reached). The actions
of the first stage that is not done are open, and so are the actions of the stages before it. So the actions of the
next stage open only after the facts that close the stage before are known. A candidate whose tactic belongs to no stage
is never open (fail closed).

The tactics are the same as executor.STAGE_TACTICS. That list is only a hint for the prompt; here it decides which
actions are open, so the two must be kept in step. The requirements in STAGES are a first guess for review, not a
validated result.
"""
from .facts import check_need, goal_reached

STAGES = (
    {'name': 'recon', 'tactics': ('discovery',),
     'done_when': (('ip', 1), ('host_lain', 1))},
    {'name': 'advance', 'tactics': ('execution', 'lateral-movement', 'credential-access', 'privilege-escalation'),
     'done_when': (('user', 1),)},
    {'name': 'impact', 'tactics': ('impact', 'exfiltration'),
     'done_when': ()},
)

_REQUIRED_KEYS = ('name', 'tactics', 'done_when')


def check_stages(stages):
    """Raise ValueError when the stage data cannot be used. run_operation calls this before anything is sent."""
    if not isinstance(stages, (tuple, list)) or not stages:
        raise ValueError('stages must be a non-empty list of stages')
    names, tactics = set(), set()
    for s in stages:
        if not isinstance(s, dict) or any(k not in s for k in _REQUIRED_KEYS):
            raise ValueError(f'a stage needs the keys {_REQUIRED_KEYS}, got {s!r}')
        name = s['name']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f'a stage name must be a new non-empty text, got {name!r}')
        names.add(name)
        ts = s['tactics']
        if not isinstance(ts, (tuple, list)) or not ts or any(not isinstance(t, str) or not t for t in ts):
            raise ValueError(f'stage {name!r} needs a non-empty list of tactic names')
        for t in ts:
            if t in tactics:
                raise ValueError(f'tactic {t!r} belongs to more than one stage')
        tactics.update(ts)
        if not isinstance(s['done_when'], (tuple, list)):
            raise ValueError(f'stage {name!r}: done_when must be a list of (name, n) pairs')
        for need in s['done_when']:
            check_need(need)


def stages_done(stages, facts):
    """Names of the stages whose done_when is met by facts (an iterable of (kind, value))."""
    facts = list(facts)
    return {s['name'] for s in stages if all(goal_reached(facts, need) for need in s['done_when'])}


def current_stage(stages, stage_done):
    """Name of the stage that is open now: the first one that is not done, or the last one when all are done."""
    for s in stages:
        if s['name'] not in stage_done:
            return s['name']
    return stages[-1]['name']


def allowed_by_stage(candidates, stage_done, stages=STAGES):
    """The candidates whose stage is open. A stage is open when it is the first one not done, or comes before it.
    stage_done holds stage names (see stages_done). A candidate whose tactic is in no stage is refused."""
    open_names = set()
    for s in stages:
        open_names.add(s['name'])
        if s['name'] not in stage_done:
            break
    stage_of = {t: s['name'] for s in stages for t in s['tactics']}
    return [l for l in candidates if stage_of.get(l.tactic) in open_names]
