"""Bridge from the agent's choice to one live Caldera ability (Phase 2).

The RL agent was trained in the CAGE-4 simulator, and its 22-feature action space does not map one-to-one onto
Caldera's abilities. So the LLM picks one ability from the live candidate list (CalderaClient.get_candidates),
and the Acceptance Gate checks that the answer is a legal index. The LLM can only choose an index into that
list, so it cannot invent an action.

SAFETY: step() with execute=False only reads and selects. With execute=True it sends a command to a real target,
and only when a live client, an ApprovalGate, and a matching human approval are all present.
The STAGE -> tactic mapping below is a design choice (REVIEW_REQUIRED), not a validated result.
"""
import time

from red.llm import AcceptanceGate

from .models import StepRecord

# Which ATT&CK tactics belong to each stage. This is a hint for the prompt only; it does not restrict the
# legal set, and it must be reviewed before real use.
STAGE_TACTICS = {
    'recon':   ('discovery',),
    'advance': ('execution', 'lateral-movement', 'credential-access', 'privilege-escalation'),
    'impact':  ('impact', 'exfiltration'),
}


def build_link_prompt(links, stage=None):
    """Numbered prompt over the live Caldera candidates (ability name, technique, tactic, target)."""
    lines = ['You are the red agent executing on a real target via Caldera.',
             'Choose exactly one ability from the numbered candidates.',
             'Reply with exactly one line: ACTION: <index>']
    if stage:
        tactics = ', '.join(STAGE_TACTICS.get(stage, ())) or 'any'
        lines.append(f'Current stage: {stage} (prefer tactics: {tactics}).')
    lines.append('')
    for i, l in enumerate(links):
        label = l.name or l.ability_id
        lines.append(f'{i}. {label} [{l.technique_id} {l.tactic}] target={l.target or "-"}')
    return '\n'.join(lines)


def select_link(links, llm, acceptance=None, stage=None, tick=0):
    """Pick one candidate with the LLM and the Acceptance Gate. Falls back to the first legal candidate when the
    answer is unparseable or out of range, or when the LLM call fails. No network except the LLM, no execution."""
    if not links:
        raise ValueError('no candidate links to select from')
    acceptance = acceptance or AcceptanceGate()
    t0 = time.perf_counter()
    res = llm.complete(build_link_prompt(links, stage))
    rec = {'stage': stage, 'tick': tick, 'decide_s': time.perf_counter() - t0,
           'llm': {'ok': res.ok, 'model': res.model, 'latency_s': res.latency_s,
                   'prompt_tokens': res.prompt_tokens, 'completion_tokens': res.completion_tokens,
                   'error': res.error}}
    if res.ok:
        verdict = acceptance.accept(res.text, len(links))
        if verdict['accepted']:
            idx = verdict['index']
            return {**rec, 'index': idx, 'source': 'LLM', 'chosen': links[idx]}
        rec['reject_reason'] = verdict['reason']
    return {**rec, 'index': 0, 'source': 'fallback', 'chosen': links[0]}


class ApprovalRequired(RuntimeError):
    """execute=True without a valid human approval. ``approval_id`` is the token waiting for approval
    (None when there is no gate at all). Nothing is sent to Caldera in this case."""

    def __init__(self, message, approval_id=None):
        super().__init__(message)
        self.approval_id = approval_id


class CalderaExecutor:
    """One Phase-2 decision against a Caldera operation. ``client`` is any object with get_candidates, submit and
    wait (tests pass a fake). Selection is always offline; sending to the target needs execute=True, a live client,
    and an approved ApprovalGate token.

    Flow: step(execute=True) with no token selects an ability, asks the gate, and raises ApprovalRequired(id).
    A person calls gate.approve(id). step(execute=True, approval_id=id) then sends only if the new choice is the
    same as the approved one. Each token can be used once."""

    def __init__(self, client, llm, acceptance=None, approval=None):
        self.client = client
        self.llm = llm
        self.acceptance = acceptance or AcceptanceGate()
        self.approval = approval          # ApprovalGate; None means execute is refused
        self._used = set()

    @staticmethod
    def _decision(op_id, chosen):
        return {'op_id': op_id, 'paw': chosen.paw, 'ability_id': chosen.ability_id,
                'executor': chosen.executor, 'command': chosen.command, 'target': chosen.target}

    def _authorize(self, op_id, chosen, approval_id):
        if self.approval is None:
            raise ApprovalRequired('execute=True needs an ApprovalGate; none is attached')
        want = self._decision(op_id, chosen)
        if approval_id is None:
            rid = self.approval.request(want)
            raise ApprovalRequired(f'waiting for human approval of {chosen.ability_id} ({rid})', rid)
        if approval_id in self._used:
            raise ApprovalRequired(f'token {approval_id} was already used', approval_id)
        if not self.approval.is_approved(approval_id):
            raise ApprovalRequired(f'token {approval_id} is not approved', approval_id)
        if self.approval.decision(approval_id) != want:
            raise ApprovalRequired(f'the chosen action differs from the approved one ({approval_id})', approval_id)
        self._used.add(approval_id)

    def step(self, op_id, paw, stage=None, tick=0, execute=False, tau=0.0, sigma=0.0, approval_id=None):
        # Refuse before the GET of potential links, because that GET changes the operation on the server.
        if execute and self.approval is None:
            raise ApprovalRequired('execute=True needs an ApprovalGate; none is attached')
        links = self.client.get_candidates(op_id, paw)
        if not links:
            return None
        sel = select_link(links, self.llm, self.acceptance, stage, tick)
        chosen = sel['chosen']
        link_status, exec_s = '', 0.0
        if execute:
            self._authorize(op_id, chosen, approval_id)
            t1 = time.perf_counter()
            link_id = self.client.submit(op_id, chosen)
            link_status = self.client.wait(op_id, link_id)
            exec_s = time.perf_counter() - t1
        return StepRecord(t=tick, stage=stage or '', paw=paw, ability_id=chosen.ability_id,
                          technique_id=chosen.technique_id, source=sel['source'], tau=tau, sigma=sigma,
                          decide_s=sel['decide_s'], latency_s=exec_s, link_status=link_status, reward=0.0)
