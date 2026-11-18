"""Unattended operation loop for Caldera. The operation is always finished at the end, even after an error.

The run stops, and the reason is written to summary.json, when:
  - the step budget is used up (stop = budget_used),
  - the STOP file exists (stop = stop_file),
  - the operation is finished on the server (stop = operation_finished),
  - no allowed candidate is left (stop = no_allowed_candidate),
  - the chooser gives no valid choice; fail closed, nothing is sent (stop = no_valid_choice),
  - the scope refuses a link; nothing is sent (stop = scope: ...),
  - a link does not end with success '0' (stop = link status ...).
"""
import json
from datetime import datetime
from pathlib import Path

from .scope import ScopeViolation

OK_STATUS = '0'


def _log(steps, step, link, info, status):
    rec = {'step': step, 'time': datetime.now().isoformat(timespec='seconds'), 'status': status,
           'ability_id': link.ability_id if link else None,
           'technique': link.technique_id if link else None,
           'target': link.target if link else None,
           'choice': info}
    steps.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
    steps.flush()


def run_operation(client, scope, chooser, mode, stage, run_dir, log=print):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {'mode': mode, 'stage': stage, 'paw': scope.paw, 'max_steps': scope.max_steps,
               'steps_sent': 0, 'stop': None, 'started': datetime.now().isoformat(timespec='seconds')}
    with client.operation('red-run') as op_id, (run_dir / 'steps.jsonl').open('x', encoding='utf-8') as steps:
        summary['operation_id'] = op_id
        for step in range(1, scope.max_steps + 1):
            if scope.stop_requested():
                summary['stop'] = 'stop_file'
                break
            if client.is_done(op_id):
                summary['stop'] = 'operation_finished'
                break
            links = client.get_candidates(op_id, scope.paw)
            if not links:
                summary['stop'] = 'no_allowed_candidate'
                break
            idx, info = chooser.choose(links, stage, tick=step, horizon=scope.max_steps)
            if idx is None:
                summary['stop'] = 'no_valid_choice'
                _log(steps, step, None, info, 'not sent')
                break
            link = links[idx]
            try:
                scope.check_link(link)
            except ScopeViolation as exc:
                summary['stop'] = f'scope: {exc}'
                _log(steps, step, link, info, 'not sent')
                break
            link_id = client.submit(op_id, link)
            status = client.wait(op_id, link_id)
            summary['steps_sent'] += 1
            _log(steps, step, link, info, status)
            log(f'step {step}: {link.ability_id} -> {status}')
            if status != OK_STATUS:
                summary['stop'] = f'link status {status}'
                break
        else:
            summary['stop'] = 'budget_used'
    summary['finished'] = datetime.now().isoformat(timespec='seconds')
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary
