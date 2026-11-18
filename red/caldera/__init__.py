"""Caldera bridge (Phase 2, lab only): REST client, link selection, and the human approval gate.

Nothing here sends a command to a target unless the caller asks for it with execute=True and an approved token.
Planning (execute=False) only reads from Caldera and selects one ability.
"""
