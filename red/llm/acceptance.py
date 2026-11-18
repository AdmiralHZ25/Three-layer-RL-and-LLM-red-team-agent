"""Acceptance Gate for an LLM-proposed action (decision D7).

Default (D7 = (1)+(2)): accept iff the output parses to an index AND that index is a legal candidate.
Optional strict mode (3): the proposed action's main-head Q must be within the top ``m`` candidates, with
``m`` chosen on validation seeds. The index safety means the LLM can never invent an out-of-set action, so
the 22-feature contract and native action indices are never touched.
"""
import re

import numpy as np

# The whole output must be one non-negative integer, optionally prefixed 'ACTION:'. Anything else (prose,
# several numbers, a reasoning block, a refusal) is a failure, never a guess.
_ACTION = re.compile(r'\s*(?:ACTION:\s*)?([0-9]+)\s*', re.IGNORECASE)


def parse_action_index(text):
    """The index if the entire output is one integer (optionally 'ACTION: <n>'); else None."""
    m = _ACTION.fullmatch(text or '')
    return int(m.group(1)) if m else None


class AcceptanceGate:
    def __init__(self, top_m=None):
        self.top_m = top_m                      # None -> rule (3) off

    def accept(self, text, n_candidates, main_head_q=None):
        """Return {accepted, index, reason}. ``main_head_q`` ([N] Q of the main head) is only needed when
        ``top_m`` is set."""
        idx = parse_action_index(text)
        if idx is None:
            return {'accepted': False, 'index': None, 'reason': 'unparseable'}
        if not 0 <= idx < n_candidates:
            return {'accepted': False, 'index': idx, 'reason': 'out_of_range'}
        if self.top_m is not None:
            if main_head_q is None:
                return {'accepted': False, 'index': idx, 'reason': 'no_q_for_top_m'}
            order = list(np.argsort(np.asarray(main_head_q, dtype=np.float64))[::-1])
            if idx not in order[:self.top_m]:
                return {'accepted': False, 'index': idx, 'reason': 'not_in_top_m'}
        return {'accepted': True, 'index': idx, 'reason': 'ok'}
