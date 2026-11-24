"""Agentic Caldera run (menu g). A person writes the objective in words and sets the limits once.

For each run the LLM planner (see planner.py) first writes a short plan. For each step it picks up to three abilities
from a short list. The list is built from the objective, the plan and the memory of the run, and it can mix tactics.
The three abilities are sent together, so they run on one beacon. The run stops when the objective is reached with a
quote from the real output, or when a limit or a safety stop applies.

Stop reasons (summary.json, key 'stop'):
  objective_reached        the planner said done, and its quote is really in the output
  budget_used              the step ceiling is used up; the objective is not reached
  time_limit               the time ceiling is used up
  stop_file                the STOP file exists
  operation_finished       the operation is finished on the server
  no_allowed_candidate     the server offers nothing that is not blocked and not already sent
  planner_stopped: <why>   the planner itself answers ACTION: STOP with a reason (objective done or not reachable)
  planner_no_valid_choice  the LLM gave no valid answer; nothing is sent
  scope: ...               a link is outside the lab host or outside the offered list; nothing in that step is sent
  link status X            a link waits for a person on the Caldera server (untrusted, high_viz)
  N batch gagal berturut-turut   N steps in a row had no successful link
  objective_unverified     the planner claimed the objective, but the check step did not show it; the run stops

An ability is never offered to the planner when:
  - its name contains a blocked word (action_scope.FORBIDDEN_NAME_WORDS),
  - its command still has an unfilled Caldera fact variable (#{...}), so it cannot run as it is,
  - it would cut the agent's own control (restart, shutdown, logoff, Wi-Fi off),
  - once the run has seen that the Atomic Red Team module is missing, its command needs that module.
Output text is data from the target. It is labelled as data in every prompt.
"""
import base64
import json
import re
import time
from datetime import datetime
from pathlib import Path

from red.caldera.action_scope import forbidden_reason
from red.caldera.facts import extract_facts
from red.caldera.memory import RunMemory
from red.caldera.planner import (BATCH_MAX, CHECK_MAX_TOKENS, CHOICE_MAX_TOKENS, PLAN_MAX_TOKENS, SHORTLIST_SIZE,
                                 build_choice_prompt, build_done_prompt, build_plan_prompt, evidence_in_output,
                                 notes_text, parse_batch, parse_done, parse_plan, phrases_shown, required_phrases,
                                 shortlist)
from red.caldera.runner import _new_facts
from red.caldera.scope import Scope, ScopeViolation

OK_STATUS = '0'
HUMAN_STATUSES = frozenset({'untrusted', 'high_viz'})

ATOMIC_MISSING = 'PathToAtomicsFolder'   # the text that marks a command which needs the Atomic Red Team module on the VM
PLACEHOLDER = '#{'                        # a Caldera fact variable that is not filled in: the ability cannot run as it is
CONTROL_NAMES = ('restart system', 'logoff system', 'shutdown system', 'disrupt wifi')   # would cut the agent's control
CONTROL_COMMAND = re.compile(r'\bshutdown\b|\bRestart-Computer\b|\bStop-Computer\b|\blogoff\b', re.I)
_BASE64 = re.compile(r'^[A-Za-z0-9+/=]+$')


def _plain(text):
    """The command as plain text. Caldera may keep a command as base64; then it is decoded for the check. Other text is
    returned unchanged. The command that is sent to Caldera is never changed by this."""
    value = (text or '').strip()
    if len(value) >= 8 and len(value) % 4 == 0 and _BASE64.match(value):
        try:
            return base64.b64decode(value, validate=True).decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            return text or ''
    return text or ''


def _is_control(link, command):
    name = (link.name or '').lower()
    return any(word in name for word in CONTROL_NAMES) or bool(CONTROL_COMMAND.search(command))


def offered_links(links, blocked=frozenset()):
    """The links the planner may see. See the list at the top of this module for what is left out."""
    out = []
    for link in links:
        if forbidden_reason({'name': link.name, 'tactic': link.tactic}) is not None:
            continue
        command = _plain(link.command)
        if PLACEHOLDER in command or _is_control(link, command):
            continue
        if 'atomic' in blocked and ATOMIC_MISSING in command:
            continue
        out.append(link)
    return out


def _ask_choice(llm, prompt, count):
    """(kind, value, reply, error). kind 'stop': value is the reason. kind 'indices': value is the list of numbers.
    kind 'none': nothing valid, value is None."""
    res = llm.complete(prompt, num_predict=CHOICE_MAX_TOKENS)
    if not res.ok:
        return 'none', None, '', res.error or 'llm failed'
    kind, value = parse_batch(res.text, count)
    return kind, value, res.text, None


def _refusal(what, error, reply):
    why = error or ('jawaban tidak sesuai format' if reply else 'tidak ada jawaban')
    return f'planner_no_valid_choice ({what}: {why})'


def _objective_check(llm, objective, raw):
    """Ask the planner whether the objective is reached. A 'yes' counts only with a quote that is in the output."""
    res = llm.complete(build_done_prompt(objective, raw), num_predict=CHECK_MAX_TOKENS)
    if not res.ok:
        return {'done': False, 'evidence': '', 'verified': False, 'reply': '', 'error': res.error}
    done, evidence = parse_done(res.text)
    verified = done and evidence_in_output(evidence, raw)
    return {'done': done, 'evidence': evidence, 'verified': verified, 'reply': res.text[:300], 'error': None,
            'claimed_without_evidence': done and not verified}


def _log(steps, step, link, choice, status, output=None, link_id=None, check=None):
    rec = {'step': step, 'time': datetime.now().isoformat(timespec='seconds'), 'status': status,
           'ability_id': link.ability_id if link else None,
           'ability_name': link.name if link else None,
           'executor': link.executor if link else None,
           'technique': link.technique_id if link else None,
           'target': link.target if link else None,
           'link_id': link_id,
           'choice': choice}
    if output is not None:
        rec['output'] = output
    if check is not None:
        rec['objective_check'] = check
    steps.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
    steps.flush()


def _ask_plan(llm, prompt):
    """(kind, steps, reply, error). kind 'plan' gives the steps; anything else gives None steps."""
    res = llm.complete(prompt, num_predict=PLAN_MAX_TOKENS)
    if not res.ok:
        return 'none', None, '', res.error or 'llm failed'
    steps = parse_plan(res.text)
    if not steps:
        return 'none', None, res.text, None
    return 'plan', steps, res.text, None


def _finish(run_dir, summary):
    summary['finished'] = datetime.now().isoformat(timespec='seconds')
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')


def run_agentic(client, llm, objective, paw, lab_hosts, run_dir, stop_file, max_steps, max_seconds=None,
                max_failures=3, rl_chooser=None, experience_recorder=None, log=print):
    objective = (objective or '').strip()
    if not objective:
        raise ValueError('the objective is empty')
    if not lab_hosts:
        raise ValueError('the lab host list is empty')
    if not 1 <= max_steps <= 20:
        raise ValueError('max_steps must be between 1 and 20')
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError('max_seconds must be above 0')
    if max_failures < 1:
        raise ValueError('max_failures must be at least 1')
    stop_file = Path(stop_file)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    memory = RunMemory()
    mode = 'rl+llm' if rl_chooser is not None else 'llm'
    summary = {'objective': objective, 'planner': getattr(llm, 'model', ''), 'mode': mode, 'paw': paw,
               'max_steps': max_steps, 'steps_sent': 0, 'batches': 0, 'failures': 0,
               'objective_reached': False, 'evidence': None,
               'stop': None, 'started': datetime.now().isoformat(timespec='seconds'), 'blocked': [], 'plan': [],
               'unverified_claims': 0, 'required': list(required_phrases(objective))}
    own_hosts = {h.lower() for h in lab_hosts}
    started = time.monotonic()
    consecutive = 0
    blocked = set()   # families of abilities that this run has learned not to offer (see ATOMIC_MISSING)
    pending_check = False   # the planner claimed the objective but the output did not show it
    # The plan comes first, before any operation is created. If the planner gives no plan, nothing is sent.
    kind, plan, plan_reply, plan_err = _ask_plan(llm, build_plan_prompt(objective))
    if kind == 'plan' and len(plan) < 2:
        # One step is rarely enough for a whole objective. Ask once more; the second answer is used either way.
        kind, plan, plan_reply, plan_err = _ask_plan(llm, build_plan_prompt(objective))
    if kind != 'plan':
        summary['stop'] = _refusal('rencana', plan_err, plan_reply)
        _finish(run_dir, summary)
        return summary
    summary['plan'] = plan
    log('rencana:\n' + '\n'.join(f'  {i}. {step}' for i, step in enumerate(plan, 1)))
    with client.operation('red-agentic') as op_id, (run_dir / 'steps.jsonl').open('x', encoding='utf-8') as steps:
        summary['operation_id'] = op_id
        for step in range(1, max_steps + 1):
            if stop_file.exists():
                summary['stop'] = 'stop_file'
                break
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                summary['stop'] = 'time_limit'
                break
            if client.is_done(op_id):
                summary['stop'] = 'operation_finished'
                break
            links = memory.mask(offered_links(client.get_all_candidates(op_id, paw), frozenset(blocked)))
            verifying = pending_check
            if verifying:
                # A claim was made but the output does not show it: this step may only check, never change the VM.
                links = [link for link in links if (link.tactic or '') == 'discovery']
            if not links:
                summary['stop'] = 'objective_unverified' if verifying else 'no_allowed_candidate'
                break
            short = shortlist(links, objective, plan, SHORTLIST_SIZE)
            notes = notes_text(memory.facts)
            history = memory.history_text()

            # RL gate: try to pick one ability when the agent is confident enough.
            rl_pick = None
            rl_step_info = None
            if rl_chooser is not None:
                _rl_idx, rl_step_info = rl_chooser.choose(short, tick=step, horizon=max_steps)
                if _rl_idx is not None and rl_step_info.get('gate_passed'):
                    rl_pick = short[_rl_idx]

            # LLM fills remaining slots from the shortlist (minus the RL pick, if any).
            llm_short = [link for link in short if link is not rl_pick]
            llm_budget = BATCH_MAX - (1 if rl_pick else 0)

            if llm_short:
                kind, value, reply, err = _ask_choice(
                    llm, build_choice_prompt(objective, llm_short, notes, history, plan, verify=verifying),
                    len(llm_short))
            else:
                kind, value, reply, err = 'indices', [], '', None

            if kind == 'stop':
                # The planner decided by itself that the objective is done or cannot be reached. Its reason is kept.
                summary['stop'] = f'planner_stopped: {value}'
                _log(steps, step, None, {'planner': 'aksi', 'reply': reply[:300], 'rl_info': rl_step_info},
                     'not sent')
                break

            llm_links = [llm_short[i] for i in value[:llm_budget]] if kind == 'indices' else []
            batch = ([rl_pick] if rl_pick else []) + llm_links

            if not batch:
                summary['stop'] = _refusal('aksi', err, reply)
                _log(steps, step, None, {'planner': 'aksi', 'reply': reply[:300], 'rl_info': rl_step_info,
                                         'error': err}, 'not sent')
                break
            ids = frozenset(candidate.ability_id for candidate in links)
            scope = Scope(paw=paw, lab_hosts=tuple(lab_hosts), allowed_abilities=ids, max_steps=max_steps,
                          stop_file=stop_file)
            try:
                for link in batch:
                    scope.check_link(link)
            except ScopeViolation as exc:
                summary['stop'] = f'scope: {exc}'
                for link in batch:
                    _log(steps, step, link, {'batch_size': len(batch), 'reply': reply[:300]}, 'not sent')
                break
            choice_base = {'batch_size': len(batch), 'reply': reply[:300], 'rl_info': rl_step_info}
            # Record step start for experience collection (before submission).
            n_facts_before = len(memory.facts)
            if experience_recorder is not None:
                for link in batch:
                    experience_recorder.record_step(link, tick=step, n_facts=n_facts_before)
            link_ids = [client.submit(op_id, link, allowed=set(ids)) for link in batch]
            statuses = [client.wait(op_id, link_id) for link_id in link_ids]
            summary['steps_sent'] += len(batch)
            summary['batches'] += 1
            raws = []
            outputs = []
            for link, link_id, status in zip(batch, link_ids, statuses):
                raw = client.get_link_result(op_id, link_id)
                raws.append(raw or '')
                outputs.append(memory.record(step, link, status, raw))
                memory.add_facts(_new_facts(extract_facts(raw), own_hosts))
                if ATOMIC_MISSING in (raw or '') and 'atomic' not in blocked:
                    # Learned from the output, not chosen by a person: the Atomic module is missing on the VM, so no
                    # ability that needs it is offered again in this run.
                    blocked.add('atomic')
                    summary['blocked'] = sorted(blocked)
            succeeded = [raw for raw, status in zip(raws, statuses) if status == OK_STATUS]
            check = _objective_check(llm, objective, '\n'.join(succeeded)) if succeeded else None
            if check is not None:
                # The objective may name the words the output must show (memuat "..."). The quote of the planner alone
                # is not enough: the output must show those words too.
                mode, phrases = required_phrases(objective)
                check['required_shown'] = phrases_shown(mode, phrases, '\n'.join(succeeded))
                check['verified'] = bool(check['verified'] and check['required_shown'])
            for link, link_id, status, output in zip(batch, link_ids, statuses, outputs):
                record_check = check if status == OK_STATUS else None
                _log(steps, step, link, {**choice_base, 'tactic': link.tactic}, status, output, link_id=link_id,
                     check=record_check)
            log(f'step {step}: ' + '; '.join(f'[{link.tactic}] {link.name or link.ability_id} -> {status}'
                                              for link, status in zip(batch, statuses)))
            human = [status for status in statuses if status in HUMAN_STATUSES]
            if human:
                summary['stop'] = f'link status {human[0]}'
                break
            summary['failures'] += sum(1 for status in statuses if status != OK_STATUS)
            if verifying:
                # The check step did not show the objective: stop instead of acting on an unproven claim.
                verify_ok = bool(check and check['verified'])
                if verify_ok:
                    summary['objective_reached'] = True
                    summary['evidence'] = check['evidence']
                    summary['stop'] = 'objective_reached'
                else:
                    summary['stop'] = 'objective_unverified'
                if experience_recorder is not None:
                    n_facts_after = len(memory.facts)
                    for link, status in zip(batch, statuses):
                        experience_recorder.close_step(
                            status=status, n_facts_after=n_facts_after,
                            next_links=[], next_tick=step + 1,
                            objective_reached=verify_ok,
                        )
                break
            if not succeeded:
                consecutive += 1
                if consecutive >= max_failures:
                    summary['stop'] = f'{max_failures} batch gagal berturut-turut (status terakhir {statuses[-1]})'
                    break
                continue
            consecutive = 0
            step_objective_reached = bool(check and check['verified'])
            if step_objective_reached:
                summary['objective_reached'] = True
                summary['evidence'] = check['evidence']
                summary['stop'] = 'objective_reached'
            # Close experience records for this step now that outcome is known.
            if experience_recorder is not None:
                next_links_for_exp = shortlist(
                    memory.mask(offered_links(client.get_all_candidates(op_id, paw), frozenset(blocked))),
                    objective, plan, SHORTLIST_SIZE,
                ) if not step_objective_reached else []
                n_facts_after = len(memory.facts)
                for link, status in zip(batch, statuses):
                    experience_recorder.close_step(
                        status=status,
                        n_facts_after=n_facts_after,
                        next_links=next_links_for_exp,
                        next_tick=step + 1,
                        objective_reached=step_objective_reached,
                    )
            if step_objective_reached:
                break
            if check and check['done']:
                # The planner claims the objective, but the output does not show it. The next step may only check.
                pending_check = True
                summary['unverified_claims'] += 1
        else:
            summary['stop'] = 'budget_used'
    if experience_recorder is not None:
        experience_recorder.flush()
    _finish(run_dir, summary)
    return summary
