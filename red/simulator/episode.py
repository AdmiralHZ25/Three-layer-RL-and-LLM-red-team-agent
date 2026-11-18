"""Instrumented episode loop derived from red.simulator.cage4; never executed on import."""
import json
import random
import time
from collections import Counter
import numpy as np
from red.simulator.cage4 import (DIM, REDS, Knowledge, Milestones, TernaryEnum, InvalidAction,
                            make_env, snapshot, candidates, confirmed_impact,
                            privileged_hosts, privilege_gain)
from red.simulator.chooser import EpsilonGreedyChooser


def outcome_status(executed, invalid, filtered, success, failed):
    if invalid:
        return 'invalid'
    if filtered:
        return 'filtered'
    if not executed:
        return 'censored'
    if success:
        return 'success'
    if failed:
        return 'failure'
    return 'unknown'


def candidate_features(next_candidates):
    """Use the same feature width as the v3 DQN, including terminal states."""
    return (np.stack([c[2] for c in next_candidates]) if next_candidates
            else np.empty((0, DIM), dtype=np.float32))


def episode(seed, cfg, net, eps, learner, blue, scale, policy, emit, chooser=None, transition_router=None):
    # transition_router (optional): object with add(choice, feature, reward, next_candidates, used). When set it
    # receives each transition INSTEAD of ``learner`` (used by the stage specialists to mask the target per
    # stage); None keeps the original path unchanged.
    env = make_env(seed, cfg.steps, blue)
    control = env.environment_controller
    knowledge = Knowledge()
    for agent in REDS:
        knowledge.observe(agent, env.get_observation(agent))
    version = getattr(cfg, 'reward_version', 'v1')
    mask = getattr(cfg, 'mask_last_withdraw', False)
    milestones = Milestones(cfg.weights, version)
    rng = random.Random(seed + 200000)
    if chooser is None:
        chooser = EpsilonGreedyChooser(net, eps, rng)
    counts, components, latencies = Counter(), Counter(), []
    first_impact, ticks, decisions = None, 0, 0
    max_zone, successes = 0, Counter()
    # Initial state is baseline, not a progress reward.
    previous = snapshot(env)
    reached = {k: previous[k] for k in ('restricted', 'foothold', 'privilege')}
    milestones.seen.update(k for k in ('restricted', 'foothold', 'privilege') if previous[k])
    priv_before = privileged_hosts(env) if version == 'v2' else None
    current = candidates(env, knowledge, ticks, cfg.steps, mask)
    while current and ticks < cfg.steps and not control.done:
        started = time.perf_counter()
        choice = chooser.choose(current, knowledge, ticks)
        idx = choice.index
        if net is not None:
            latencies.append((time.perf_counter() - started) * 1000)
        agent, action, feature, target = current[idx]
        name = type(action).__name__
        counts[name] += 1
        decisions += 1
        knowledge.attempts[(agent, name, target)] += 1
        emit({'kind': 'selected', 'decision': decisions, 'tick': ticks,
              'agent': agent, 'action': name, 'target': target,
              'mask_valid_at_selection': True,
              'parameters': {k: str(v) for k, v in action.get_params().items()
                             if k in ('agent', 'session', 'target_session', 'hostname', 'ip_address', 'subnet')}})
        outcome = None
        executed = False
        invalid = False
        filtered = False
        new_hosts = 0
        used = 0
        observed_progress = {k: False for k in reached}
        while ticks < cfg.steps and not control.done:
            control.step({agent: action} if used == 0 else {})
            ticks += 1
            used += 1
            if used == 1:
                emit({'kind': 'submitted', 'decision': decisions, 'tick': ticks,
                      'agent': agent, 'action': name})
            # Read the individual action observation before combining with the
            # successful automatic RedSessionCheck observation.
            actions = control.action.get(agent, [])
            actual = actions[0] if actions else None
            if isinstance(actual, InvalidAction):
                invalid = True
                executed = True
            elif actual is action:
                executed = True
            if executed:
                # control.action is populated before filter_actions; missing
                # sessions can remove an action before execution. Match the
                # actual observation, never the automatic RedSessionCheck.
                matched = [o.data for o in control.observation[agent].observations
                           if o.data.get('action') is actual]
                if matched:
                    outcome = matched[0].get('success')
                else:
                    filtered = True
            for red in REDS:
                new_hosts += knowledge.observe(red, env.get_observation(red))
            state_now = snapshot(env)
            for key in reached:
                reached[key] = reached[key] or state_now[key]
                observed_progress[key] = observed_progress[key] or state_now[key]
            if executed:
                break
        failed = executed and not invalid and not filtered and outcome == TernaryEnum.FALSE
        success = executed and not invalid and not filtered and outcome == TernaryEnum.TRUE
        if invalid:
            successes['invalid'] += 1
        if failed:
            successes['failed'] += 1
            successes['failed_' + name] += 1
        if filtered:
            successes['filtered_actions'] += 1
        if executed and not invalid and not filtered:
            successes['valid_executions'] += 1
            successes['valid_' + name] += 1
        if not executed:
            successes['censored_actions'] += 1
        if success:
            successes['successful_' + name] += 1
        now = snapshot(env)
        impact = confirmed_impact(action, success, previous, now)
        if impact and first_impact is None:
            first_impact = ticks
        flags = {k: success and observed_progress[k] and not previous[k] for k in reached}
        flags['impact'] = impact
        if version == 'v2':
            priv_now = privileged_hosts(env)
            flags['privilege'] = privilege_gain(action, success, priv_before, priv_now)
            priv_before = priv_now
        if reached['foothold']:
            max_zone = 2
        elif reached['restricted']:
            max_zone = max(max_zone, 1)
        # Unique observations give a bounded discovery signal. Success is
        # required so RedSessionCheck cannot reward an unrelated failure.
        parts = milestones.reward(flags, discovered=success and new_hosts > 0,
                                  invalid=invalid, failed=failed, ticks=used, scale=scale,
                                  impact_host=getattr(action, 'hostname', ''))
        status = outcome_status(executed, invalid, filtered, success, failed)
        emit({'kind': 'outcome', 'decision': decisions, 'tick': ticks,
              'agent': agent, 'action': name, 'target': target,
              'status': status, 'simulator_outcome': str(outcome),
              'valid_execution': bool(executed and not invalid and not filtered),
              'operational_impact_confirmed': bool(impact),
              'restricted_seen': bool(reached['restricted']),
              'operational_foothold_seen': bool(reached['foothold']),
              'operational_privilege_seen': bool(reached['privilege']),
              'progress_origin': 'unknown', 'reward_components': parts})
        components.update(parts)
        next_candidates = candidates(env, knowledge, ticks, cfg.steps, mask) if not control.done and ticks < cfg.steps else []
        if transition_router is not None:
            transition_router.add(choice, feature, sum(parts.values()), next_candidates, used)
        elif learner is not None:
            next_features = candidate_features(next_candidates)
            learner.add(feature, sum(parts.values()), next_features, not next_candidates, used)
        current, previous = next_candidates, now
    termination = ('simulator_done' if control.done else
                   'horizon' if ticks >= cfg.steps else 'no_candidates')
    row = {'policy': policy, 'seed': seed, 'blue': 'random' if blue else 'sleep',
           'termination_reason': termination,
           'episode_return': sum(components.values()), 'length_ticks': ticks,
           'decisions': decisions, 'impact_success': int(first_impact is not None),
           'time_to_impact': first_impact if first_impact is not None else '',
           'time_to_impact_censored': int(first_impact is None), 'max_zone': max_zone,
           'restricted': int(reached['restricted']),
           'foothold': int(reached['foothold']),
           'privilege': int(reached['privilege']),
           'successful_exploit': successes['successful_ExploitRemoteService'],
           'successful_privilege_escalation': successes['successful_PrivilegeEscalate'],
           'impact_attempts': counts['Impact'], 'impact_valid_executions': successes['valid_Impact'],
           'impact_successful_executions': successes['successful_Impact'],
           'invalid_actions': successes['invalid'], 'failed_actions': successes['failed'],
           'filtered_actions': successes['filtered_actions'],
           'valid_executions': successes['valid_executions'], 'censored_actions': successes['censored_actions'],
           'latency_mean_ms': float(np.mean(latencies)) if latencies else '',
           'rl_decision_latencies_ms': json.dumps(latencies),
           'action_distribution': json.dumps(dict(counts), sort_keys=True)}
    row.update({f'reward_{k}': components[k] for k in ('step','restricted','foothold','privilege','impact','discovery','invalid','failed')})
    return row


