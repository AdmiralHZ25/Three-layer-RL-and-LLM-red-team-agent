"""Scope of an unattended Caldera run.

The scope is the approval, given in advance: only abilities in the allow-list and hosts in the lab list may be sent.
The run also stops at its step budget, or as soon as the STOP file exists.
"""
from dataclasses import dataclass
from pathlib import Path

MAX_STEPS_LIMIT = 20


class ScopeViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class Scope:
    paw: str
    lab_hosts: tuple
    allowed_abilities: frozenset
    max_steps: int
    stop_file: Path

    def __post_init__(self):
        if not 1 <= self.max_steps <= MAX_STEPS_LIMIT:
            raise ValueError(f'max_steps must be between 1 and {MAX_STEPS_LIMIT}')
        if not self.allowed_abilities:
            raise ValueError('the ability allow-list is empty')
        if not self.lab_hosts:
            raise ValueError('the lab host list is empty')

    def check_link(self, link):
        """Refuse a link whose ability or target host is outside the scope. Raises ScopeViolation."""
        if link.ability_id not in self.allowed_abilities:
            raise ScopeViolation(f'ability {link.ability_id} is not in the allow-list')
        lab = {h.lower() for h in self.lab_hosts}
        if (link.target or '').lower() not in lab:
            raise ScopeViolation(f'host {link.target or "-"} is not a lab host')

    def stop_requested(self):
        return self.stop_file.exists()
