"""Manual control of one Caldera agent (Phase 2, lab only).

A person picks an action from the allow-list and types YA to confirm it. Nothing is sent without that confirmation.
Each action must pass the scope (allowed ability, lab host, step limit, STOP file), is recorded in the approval gate,
and is written to a log file (steps.jsonl). The typed YA is the human approval for that one decision.
"""
import json
from datetime import datetime

from .approval import ApprovalGate
from .scope import ScopeViolation

CONFIRM_WORD = 'YA'

_STATUS = {
    '0': ('BERHASIL', 'Perintah selesai di agent.'),
    '1': ('GAGAL', 'Perintah berakhir dengan error di agent.'),
    '-2': ('DIBUANG', 'Caldera membuang perintah ini.'),
    '124': ('WAKTU HABIS', 'Agent belum selesai. Cek hasilnya di GUI Caldera.'),
    'timeout': ('WAKTU HABIS', 'Agent belum selesai. Cek hasilnya di GUI Caldera.'),
    'untrusted': ('MENUNGGU', 'Agent belum dipercaya. Setujui di GUI Caldera.'),
    'high_viz': ('MENUNGGU', 'Aksi ini butuh persetujuan di GUI Caldera.'),
}


def describe_status(status):
    """(label, text) for a link status as returned by CalderaClient.wait."""
    return _STATUS.get(str(status), ('TIDAK DIKENAL', f'Status {status} tidak dikenal.'))


def describe_action(link):
    """Short human-readable name for one allowed action."""
    return f"{link.name or link.ability_id} (teknik {link.technique_id or '-'}, {link.tactic or '-'})"


class ManualSession:
    """One manual session on one agent. ``client`` is a CalderaClient (or a fake in tests)."""

    def __init__(self, client, scope, log_path, approval=None):
        self.client = client
        self.scope = scope
        self.log_path = log_path
        self.approval = approval or ApprovalGate()
        self.op_id = None
        self.links = []
        self.sent = 0

    def refresh(self):
        """Reload the allowed actions. This GET also changes the operation's potential links on the server."""
        self.links = self.client.get_candidates(self.op_id, self.scope.paw)
        return self.links

    def remaining(self):
        return self.scope.max_steps - self.sent

    def act(self, index, confirm):
        """Send self.links[index] only when confirm == CONFIRM_WORD. Returns (label, text) and writes one log line."""
        if not 0 <= index < len(self.links):
            raise ValueError('nomor aksi tidak ada di daftar')
        link = self.links[index]
        entry = {'time': datetime.now().isoformat(timespec='seconds'), 'paw': link.paw,
                 'ability_id': link.ability_id, 'name': link.name,
                 'technique_id': link.technique_id, 'target': link.target}
        if confirm != CONFIRM_WORD:
            return self._finish(entry, 'BATAL', 'Tidak ada perintah yang dikirim.')
        if self.scope.stop_requested():
            return self._finish(entry, 'DIHENTIKAN', 'File STOP ada. Hapus file STOP untuk mengizinkan lagi.')
        if self.remaining() <= 0:
            return self._finish(entry, 'BATAS TERCAPAI', f'Sudah {self.sent} langkah. Batas per sesi tercapai.')
        try:
            self.scope.check_link(link)
        except ScopeViolation as exc:
            return self._finish(entry, 'DITOLAK SCOPE', str(exc))
        decision = {'op_id': self.op_id, 'paw': link.paw, 'ability_id': link.ability_id,
                    'executor': link.executor, 'command': link.command, 'target': link.target}
        rid = self.approval.request(decision)
        self.approval.approve(rid)   # the typed YA is the approval for this one decision
        entry['approval_id'] = rid
        link_id = self.client.submit(self.op_id, link)
        self.sent += 1
        status = self.client.wait(self.op_id, link_id)
        entry['link_status'] = str(status)
        label, text = describe_status(status)
        return self._finish(entry, label, text)

    def _finish(self, entry, label, text):
        entry['result'] = label
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        return label, text


def abilities_for_platform(abilities, platform):
    """Rows for the menu: abilities that have an executor for this platform, sorted by tactic then name."""
    rows = []
    for ab in abilities:
        names = sorted({e.get('name') for e in ab.get('executors', []) if e.get('platform') == platform and e.get('name')})
        if names:
            rows.append({'ability_id': ab.get('ability_id', ''), 'name': ab.get('name') or ab.get('ability_id', ''),
                         'technique_id': ab.get('technique_id') or '-', 'tactic': ab.get('tactic') or '-',
                         'executors': ','.join(names)})
    return sorted(rows, key=lambda r: (r['tactic'], r['name']))


def parse_choice_numbers(text, count):
    """'1, 3' -> [0, 2] for a list of ``count`` items. Raises ValueError for anything not in the list."""
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if not parts:
        raise ValueError('pilih minimal satu nomor')
    picked = []
    for p in parts:
        if not p.isdigit() or not 1 <= int(p) <= count:
            raise ValueError(f'nomor {p} tidak ada di daftar')
        if int(p) - 1 not in picked:
            picked.append(int(p) - 1)
    return picked
