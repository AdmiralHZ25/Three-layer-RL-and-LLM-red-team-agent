"""Caldera REST v2 client (Phase 2, real backend). Use only against a Caldera server you are authorized to operate.

Read-only: list_agents, check_paw, get_state, is_done.
Side effects on the server: create_operation / finish (an operation is created and then finished),
get_candidates (changes the operation's potential links), submit (sends a command to an agent).
Endpoints and fields were checked against the Caldera 5.3 source; [VERIFY] marks what must be checked once
the server is running.
"""
import json
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

from red import config
from .models import PotentialLink


class CalderaError(RuntimeError):
    pass


# Link states (Caldera c_link.py): HIGH_VIZ=-5 UNTRUSTED=-4 EXECUTE=-3 DISCARD=-2 PAUSE=-1 SUCCESS=0 ERROR=1 TIMEOUT=124.
# Terminal states end the wait. UNTRUSTED and HIGH_VIZ wait for a human on the server, so they are returned at once.
_TERMINAL_LINK_STATES = {-2, 0, 1, 124}
_SPECIAL_LINK_STATES = {-4: "untrusted", -5: "high_viz"}

# Operation states that count as finished (Caldera c_operation.py).
_FINISHED_OP_STATES = {"finished", "out_of_time", "cleanup"}


class CalderaClient:
    def __init__(self, base_url=None, api_key=None, timeout=None):
        self.base = (base_url or config.CALDERA_URL).rstrip("/")
        self.key = api_key or config.CALDERA_API_KEY
        self.timeout = timeout or config.CALDERA_HTTP_TIMEOUT

    def _req(self, method, path, body=None):
        if not (self.key or "").strip():
            raise CalderaError("CALDERA_API_KEY is not set (set it as an environment variable)")
        url = f"{self.base}/api/v2{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"KEY": self.key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raise CalderaError(f"{method} {path} -> {e.code} {e.read().decode(errors='ignore')}")
        except urllib.error.URLError as e:
            raise CalderaError(f"{method} {path} -> {e}")

    # --- read-only ---
    def list_agents(self):
        """GET /agents: every agent Caldera knows (paw, host, platform, ...)."""
        return self._req("GET", "/agents") or []

    def check_paw(self, paw):
        """The paw must equal CALDERA_PAW, exist in GET /agents, and run on a host listed in CALDERA_LAB_HOSTS
        (by host name or IP). Otherwise raises CalderaError. Returns the paw."""
        if not config.CALDERA_PAW:
            raise CalderaError("CALDERA_PAW is not set")
        if paw != config.CALDERA_PAW:
            raise CalderaError("paw is not the same as CALDERA_PAW")
        if not config.CALDERA_LAB_HOSTS:
            raise CalderaError("CALDERA_LAB_HOSTS is empty: no lab host is allowed")
        agent = next((a for a in self.list_agents() if a.get("paw") == paw), None)
        if agent is None:
            raise CalderaError(f"paw {paw} is not in the agent list of Caldera")
        lab = {h.lower() for h in config.CALDERA_LAB_HOSTS}
        seen = [agent.get("host") or ""] + list(agent.get("host_ip_addrs") or [])
        if not any(h.lower() in lab for h in seen if h):
            raise CalderaError(f"the host of agent {paw} is not in CALDERA_LAB_HOSTS")
        return paw

    def get_state(self, op_id):
        """GET /operations/{id} and the report (facts). [VERIFY] map facts to host counts for your scenario."""
        op = self._req("GET", f"/operations/{op_id}")
        rep = self._req("POST", f"/operations/{op_id}/report", {"enable_agent_output": False})
        facts = rep.get("facts", []) if isinstance(rep, dict) else []
        return {"op_state": op.get("state"), "facts": facts,
                "discovered": 0, "users": 0, "roots": 0, "impact": 0, "mitre_phase": 1, "hosts": {}}

    def is_done(self, op_id):
        """True when the operation state is finished, out_of_time or cleanup."""
        op = self._req("GET", f"/operations/{op_id}")
        return op.get("state") in _FINISHED_OP_STATES

    def get_candidates(self, op_id, paw):
        """GET potential links for this paw, keeping only abilities in CALDERA_ALLOWED_ABILITY_IDS.
        An empty allow-list returns [] without any request, because the GET itself changes the operation."""
        allowed = set(config.CALDERA_ALLOWED_ABILITY_IDS)
        if not allowed:
            return []
        links = self._req("GET", f"/operations/{op_id}/potential-links/{paw}")
        out = []
        for l in links or []:
            ab = l.get("ability", {}) or {}
            if ab.get("ability_id", "") not in allowed:
                continue
            ex = l.get("executor", {}) or {}
            out.append(PotentialLink(
                paw=l.get("paw", paw),
                ability_id=ab.get("ability_id", ""),
                executor=ex.get("name", "sh"),
                command=ex.get("command", ""),
                tactic=ab.get("tactic", ""),
                technique_id=ab.get("technique_id", ""),
                name=ab.get("name", ""),
                target=l.get("host", "")))
        return out

    # --- operation lifecycle (creates state on the server) ---
    def create_operation(self, name="red-episode"):
        """POST /operations. autonomous=1 is required, so posted links get status EXECUTE and the agent runs them.
        Empty adversary/planner/source ids mean ad-hoc: the built-in planner adds no actions of its own."""
        body = {
            "name": name,
            "adversary": {"adversary_id": config.CALDERA_ADVERSARY_ID},
            "planner": {"id": config.CALDERA_PLANNER_ID},
            "source": {"id": config.CALDERA_SOURCE_ID},
            "autonomous": 1,
            "state": "running",
        }
        op = self._req("POST", "/operations", body)
        return op.get("id")

    def finish(self, op_id):
        self._req("PATCH", f"/operations/{op_id}", {"state": "finished"})

    @contextmanager
    def operation(self, name="red-episode"):
        """Create an operation and always finish it. If the block failed, a finish error does not hide the first error."""
        op_id = self.create_operation(name)
        ok = False
        try:
            yield op_id
            ok = True
        finally:
            try:
                self.finish(op_id)
            except CalderaError:
                if ok:
                    raise

    # --- action (sends a command to an agent) ---
    def submit(self, op_id, pl):
        """POST a potential link. Refuses any ability not in CALDERA_ALLOWED_ABILITY_IDS, and any paw that
        fails check_paw. Required fields: paw, executor name, executor command."""
        self.check_paw(pl.paw)
        if pl.ability_id not in config.CALDERA_ALLOWED_ABILITY_IDS:
            raise CalderaError(f"ability {pl.ability_id!r} is not in CALDERA_ALLOWED_ABILITY_IDS")
        body = {
            "paw": pl.paw,
            "executor": {"name": pl.executor, "command": pl.command},
            "ability": {"ability_id": pl.ability_id},
        }
        link = self._req("POST", f"/operations/{op_id}/potential-links", body)
        pl.id = link.get("id", "")
        return pl.id

    def wait(self, op_id, link_id, timeout=None):
        """Poll the link until it reaches a terminal state, or until the timeout passes (then returns 'timeout')."""
        timeout = timeout or config.CALDERA_LINK_TIMEOUT
        deadline = time.time() + timeout
        while time.time() < deadline:
            link = self._req("GET", f"/operations/{op_id}/links/{link_id}")
            status = link.get("status")
            try:
                if status is not None and int(status) in _SPECIAL_LINK_STATES:
                    return _SPECIAL_LINK_STATES[int(status)]
                if status is not None and int(status) in _TERMINAL_LINK_STATES:
                    return str(status)
            except (TypeError, ValueError):
                pass
            time.sleep(config.CALDERA_POLL_INTERVAL)
        return "timeout"
