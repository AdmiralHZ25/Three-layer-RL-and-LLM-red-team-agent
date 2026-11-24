"""Unattended operation loop for Caldera. The operation is always finished at the end, even after an error.

The run stops, and the reason is written to summary.json, when:
  - the goal is reached (stop = goal_reached), see facts.py. The goal is checked only after a link that succeeded.
  - the step ceiling is used up without the goal (stop = budget_used). summary['goal_reached'] stays False.
  - the time ceiling is used up (stop = time_limit), when max_seconds is given.
  - the STOP file exists (stop = stop_file),
  - the operation is finished on the server (stop = operation_finished),
  - no allowed candidate is left (stop = no_allowed_candidate),
  - the chooser gives no valid choice; fail closed, nothing is sent (stop = no_valid_choice),
  - the scope refuses a link; nothing is sent (stop = scope: ...),
  - a link waits for a person on the Caldera server (untrusted or high_viz); stop = link status X,
  - max_failures links in a row did not succeed (stop = N link gagal berturut-turut ...),
  - stages are given and no candidate is in an open stage (stop = stage_blocked), see stages.py.

A link that fails does not stop the run by itself. Its output is still read and its facts are kept, but the goal is
checked only after a link that succeeded, so an error text that happens to contain an address cannot reach the goal.

A host that is in the scope (the agent's own host) is not counted as a new host, because 'host lain' means another
machine.

With memory (a RunMemory, see memory.py) the loop is agentic: abilities already sent are removed from the candidates
before the chooser is called, the chooser gets the recent history, and after each link the output is read back
(get_link_result) and written to memory and to steps.jsonl. The facts of each output are added to memory too.
Without memory the loop is unchanged.

goal and stages need memory, because the facts are kept there; without memory they raise ValueError before anything is
sent. With stages, the candidates are limited to the open stage before the chooser is called, and the chooser gets the
open stage's name as its stage argument (the stage argument of run_operation is then not used).
"""
import json
import time
from datetime import datetime
from pathlib import Path

from .facts import check_need, extract_facts, goal_reached
from .scope import ScopeViolation
from .stages import allowed_by_stage, check_stages, current_stage, stages_done

OK_STATUS = '0'
# Statuses that wait for a person on the Caldera server. Another step cannot help, so the run stops at once.
HUMAN_STATUSES = frozenset({'untrusted', 'high_viz'})
DEFAULT_MAX_FAILURES = 3


def _log(steps, step, link, info, status, output=None, link_id=None):
    # ability_name and executor say what Caldera was asked to run. link_id is the id of the row in the Caldera GUI, so a
    # failed step can be found there. Without them, a link shown as 'Manual Command' cannot be traced.
    rec = {'step': step, 'time': datetime.now().isoformat(timespec='seconds'), 'status': status,
           'ability_id': link.ability_id if link else None,
           'ability_name': link.name if link else None,
           'executor': link.executor if link else None,
           'technique': link.technique_id if link else None,
           'target': link.target if link else None,
           'link_id': link_id,
           'choice': info}
    if output is not None:
        rec['output'] = output
    steps.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
    steps.flush()


def _new_facts(facts, own_hosts):
    """The facts of one output, without the scope's own hosts (they are not a 'host lain')."""
    return [(kind, value) for kind, value in facts if not (kind == 'host' and str(value).lower() in own_hosts)]


def run_operation(client, scope, chooser, mode, stage, run_dir, log=print, memory=None, goal=None, stages=None,
                  max_seconds=None, max_failures=DEFAULT_MAX_FAILURES):
    if memory is None and (goal is not None or stages is not None):
        raise ValueError('goal and stages need memory: the facts are kept there')
    if goal is not None:
        check_need(goal)      # refused before any action is sent
    if stages is not None:
        check_stages(stages)
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError('max_seconds must be above 0')
    if max_failures < 1:
        raise ValueError('max_failures must be at least 1')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {'mode': mode, 'stage': stage, 'paw': scope.paw, 'max_steps': scope.max_steps,
               'steps_sent': 0, 'failures': 0, 'goal': list(goal) if goal is not None else None,
               'goal_reached': False, 'stop': None, 'started': datetime.now().isoformat(timespec='seconds')}
    own_hosts = {h.lower() for h in scope.lab_hosts}
    started = time.monotonic()
    consecutive = 0
    with client.operation('red-run') as op_id, (run_dir / 'steps.jsonl').open('x', encoding='utf-8') as steps:
        summary['operation_id'] = op_id
        for step in range(1, scope.max_steps + 1):
            if scope.stop_requested():
                summary['stop'] = 'stop_file'
                break
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                summary['stop'] = 'time_limit'
                break
            if client.is_done(op_id):
                summary['stop'] = 'operation_finished'
                break
            links = client.get_candidates(op_id, scope.paw)
            if memory is not None:
                links = memory.mask(links)  # an ability that was already sent is not offered again
            if not links:
                summary['stop'] = 'no_allowed_candidate'
                break
            step_stage = stage
            if stages is not None:
                done = stages_done(stages, memory.facts)
                links = allowed_by_stage(links, done, stages)
                if not links:
                    summary['stop'] = 'stage_blocked'
                    break
                step_stage = current_stage(stages, done)
            extra = {} if memory is None else {'history': memory.history_text()}
            idx, info = chooser.choose(links, step_stage, tick=step, horizon=scope.max_steps, **extra)
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
            output = None
            if memory is not None:
                raw = client.get_link_result(op_id, link_id)
                output = memory.record(step, link, status, raw)
                memory.add_facts(_new_facts(extract_facts(raw), own_hosts))  # the full text, not the 300-character summary
            _log(steps, step, link, info, status, output, link_id=link_id)
            log(f'step {step}: {link.ability_id} -> {status}')
            if status in HUMAN_STATUSES:
                summary['stop'] = f'link status {status}'
                break
            if status != OK_STATUS:
                consecutive += 1
                summary['failures'] += 1
                if consecutive >= max_failures:
                    summary['stop'] = f'{max_failures} link gagal berturut-turut (status terakhir {status})'
                    break
                continue
            consecutive = 0
            if goal is not None and goal_reached(memory.facts, goal):
                summary['stop'] = 'goal_reached'
                break
        else:
            summary['stop'] = 'budget_used'
    summary['goal_reached'] = summary['stop'] == 'goal_reached'
    summary['finished'] = datetime.now().isoformat(timespec='seconds')
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary
