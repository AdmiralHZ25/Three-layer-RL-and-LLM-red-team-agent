"""ApprovalGate: a human must approve each decision before it can run on a target (stdlib only).

Deny by default: every decision waits for an explicit approval. auto_allow=True is only for offline tests.
"""


class ApprovalGate:
    def __init__(self, auto_allow=False):
        self._auto = bool(auto_allow)
        self._decision = {}     # rid -> decision
        self._status = {}       # rid -> 'pending' | 'approved' | 'denied'
        self._n = 0

    def request(self, decision):
        """Submit a decision for approval. Returns the request id."""
        self._n += 1
        rid = "appr-%d" % self._n
        self._decision[rid] = decision
        self._status[rid] = "approved" if self._auto else "pending"
        return rid

    def approve(self, rid):
        self._set(rid, "approved")

    def deny(self, rid):
        self._set(rid, "denied")

    def _set(self, rid, status):
        if rid not in self._status:
            raise KeyError(rid)
        self._status[rid] = status

    def status(self, rid):
        return self._status.get(rid)

    def is_approved(self, rid):
        return self._status.get(rid) == "approved"

    def decision(self, rid):
        return self._decision.get(rid)

    def pending(self):
        return [rid for rid, s in self._status.items() if s == "pending"]
