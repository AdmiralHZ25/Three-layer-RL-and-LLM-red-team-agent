"""Turn a legal-candidate set into a numbered prompt for the LLM.

The agent's candidates are 22-dim numeric features, which a language model cannot read, so the prompt is
built from each candidate's action class name and target. The model is asked to reply with one integer
into this same list (the parser also accepts ``ACTION: <index>``). The model picks an index
only: no free text, no commands. Any other output is a failure that falls through to the RL choice.
"""


def candidate_label(candidate):
    """Human-readable label for one (agent, action, feature, target) candidate tuple."""
    action = candidate[1]
    name = type(action).__name__ if action is not None else 'Action'
    target = candidate[3]
    return name if target in (None, '') else f'{name} target={target}'


def llm_options(candidates, main_q, mask=None):
    """Candidate indices the LLM may pick: legal ones, one per distinct label. Among duplicates the highest
    main-head Q wins, so the RL still decides which duplicate; the LLM decides the action kind and target."""
    best = {}
    for i, c in enumerate(candidates):
        if mask is not None and not mask[i]:
            continue
        label = candidate_label(c)
        if label not in best or main_q[i] > main_q[best[label]]:
            best[label] = i
    return list(best.values())


# Tujuan misi untuk LLM (uji 25 Nov, flag --llm-goal). Hipotesis, belum tervalidasi: LLM tidak pernah memilih Impact.
GOAL_TEXT = ('Mission goal: reach Impact on a target host. Typical progression: discover hosts, gain foothold, '
             'escalate privilege, then Impact. Prefer actions that move toward Impact. Avoid Sleep and Withdraw '
             'unless no other action helps.')


def build_prompt(candidates, knowledge=None, tick=None, options=None, goal=None):
    """``options``: candidate indices to list (default: all). The reply is an index into this list.
    ``goal``: optional one-line mission text placed above the state; None keeps the original prompt."""
    idx = list(range(len(candidates))) if options is None else list(options)
    n = len(idx)
    head = [goal, ''] if goal else []
    lines = head + [f'State: tick {tick if tick is not None else "?"}, {n} legal actions.',
                    f'Choose exactly one action from the numbered candidates below (0 to {n - 1}).',
                    'Reply with one integer only, nothing else.', '']
    for k, i in enumerate(idx):
        lines.append(f'{k}. {candidate_label(candidates[i])}')
    return '\n'.join(lines)
