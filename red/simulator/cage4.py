"""Offline CAGE-4 experiment. See CAGE4_V3_EXPERIMENT_REPORT.md for protocol.

Learner = DDQN (Double DQN target) since 9 Nov 2026. The old DQN code, its backup and
the rl_cage4/v3 results were permanently deleted on 9 Nov 2026 at the researcher's request;
manual_training/run_2000 (DQN) is still present only as diagnostic evidence.

The policy consumes only red observations and action spaces. Simulator state is
read by the isolated evaluator for milestone rewards and outcome verification.
"""
import argparse
import csv
import inspect
import itertools
import json
import math
import random
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party' / 'cyborg'))
from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, cc4BlueRandomAgent
from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions import InvalidAction
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

REDS = [f'red_agent_{i}' for i in range(6)]
NAMES = ('AggressiveServiceDiscovery', 'DegradeServices', 'DiscoverDeception',
         'DiscoverRemoteSystems', 'ExploitRemoteService', 'Impact',
         'PrivilegeEscalate', 'Sleep', 'StealthServiceDiscovery', 'Withdraw')
DIM = len(NAMES) + 12


def zone(host):
    host = str(host).lower()
    if 'operational_zone' in host:
        return 2
    if 'restricted_zone' in host:
        return 1
    return 0


@dataclass
class Weights:
    step: float = -0.05
    restricted: float = 5.0
    foothold: float = 10.0
    privilege: float = 15.0
    impact: float = 100.0
    discovery: float = 0.2
    invalid: float = -1.0
    failed: float = -0.1


REWARD_VERSIONS = ('v1', 'v2')


class Milestones:
    """v1 (default): each milestone type paid once per episode.
    v2: Impact paid once per unique operational host (``impact_host``); privilege flag is supplied
    by the caller (PrivilegeEscalate on its own target host, see ``privilege_gain``)."""
    def __init__(self, weights, reward_version='v1'):
        if reward_version not in REWARD_VERSIONS:
            raise ValueError(f'reward_version must be one of {REWARD_VERSIONS}, got {reward_version!r}')
        self.weights = weights
        self.reward_version = reward_version
        self.seen = set()
        self.impact_hosts = set()

    def reward(self, flags, discovered=False, invalid=False, failed=False, ticks=1, scale=1.0, impact_host=None):
        parts = {'step': self.weights.step * ticks, 'restricted': 0., 'foothold': 0.,
                 'privilege': 0., 'impact': 0., 'discovery': 0.,
                 'invalid': self.weights.invalid if invalid else 0.,
                 'failed': self.weights.failed if failed else 0.}
        for key in ('restricted', 'foothold', 'privilege', 'impact'):
            if key == 'impact' and self.reward_version == 'v2':
                if flags.get('impact'):
                    if impact_host is None:
                        raise ValueError('reward_version v2 needs impact_host for a confirmed Impact')
                    if impact_host not in self.impact_hosts:
                        self.impact_hosts.add(impact_host)
                        parts['impact'] = self.weights.impact
                continue
            if flags.get(key) and key not in self.seen:
                self.seen.add(key)
                parts[key] = getattr(self.weights, key) * (1.0 if key == 'impact' else scale)
        if discovered:
            parts['discovery'] = self.weights.discovery * scale
        return parts


def make_env(seed, steps, blue):
    sg = EnterpriseScenarioGenerator(blue_agent_class=cc4BlueRandomAgent if blue else SleepAgent,
                                     green_agent_class=EnterpriseGreenAgent,
                                     red_agent_class=SleepAgent, steps=steps)
    env = CybORG(scenario_generator=sg, seed=seed)
    # This source constructs Blue without passing its RNG. Seed each Blue stream
    # explicitly after construction; creating a fresh environment avoids reset leakage.
    for name, interface in env.environment_controller.agent_interfaces.items():
        if name.startswith('blue_agent_'):
            interface.agent.np_random = np.random.default_rng(seed + 10000 + int(name.rsplit('_', 1)[1]))
    return env


def valid_action(action, space):
    if not space['action'].get(type(action), False):
        return False
    for key, value in action.get_params().items():
        if key in space and (value not in space[key] or not space[key][value]):
            return False
    return True


class Knowledge:
    """Only red-observed information; maps IPs to hostnames when observed."""
    def __init__(self):
        self.ips = {}
        self.current_sessions = {a: set() for a in REDS}
        self.attempts = Counter()
        self.known = set()

    def observe(self, agent, obs):
        sessions = set()
        new = 0
        for key, info in obs.items():
            if not isinstance(info, dict):
                continue
            host = info.get('System info', {}).get('Hostname', str(key))
            if host not in self.known:
                self.known.add(host)
                new += 1
            for iface in info.get('Interface', []):
                if 'ip_address' in iface:
                    self.ips[str(iface['ip_address'])] = host
            for session in info.get('Sessions', []):
                if session.get('agent') == agent:
                    sessions.add(host)
        self.current_sessions[agent] = sessions
        return new


def candidates(env, knowledge, tick, horizon, mask_last_withdraw=False):
    """Enumerate the true-mask Cartesian product; no hidden topology is consulted.

    mask_last_withdraw (default off): drop a Withdraw aimed at the only host where any red agent
    still has a session (policy-visible ``knowledge`` only)."""
    result = []
    session_hosts = set().union(*knowledge.current_sessions.values())
    last_host = next(iter(session_hosts)) if mask_last_withdraw and len(session_hosts) == 1 else None
    for agent in REDS:
        if not knowledge.current_sessions[agent]:
            continue
        space = env.get_action_space(agent)
        for cls in sorted(space['action'], key=lambda c: c.__name__):
            if not space['action'][cls] or cls.__name__ not in NAMES:
                continue
            keys, choices = [], []
            for name, param in inspect.signature(cls).parameters.items():
                if name == 'self':
                    continue
                if name not in space:
                    if param.default is inspect.Parameter.empty:
                        choices = []
                        break
                    continue
                options = sorted((k for k, v in space[name].items() if v), key=str)
                if not options:
                    choices = []
                    break
                keys.append(name)
                choices.append(options)
            else:
                for values in itertools.product(*choices):
                    action = cls(**dict(zip(keys, values)))
                    if not valid_action(action, space):
                        continue
                    target = getattr(action, 'hostname', getattr(action, 'ip_address', getattr(action, 'subnet', '')))
                    host = knowledge.ips.get(str(target), str(target))
                    if last_host is not None and cls.__name__ == 'Withdraw' and host == last_host:
                        continue
                    repeat = knowledge.attempts[(agent, cls.__name__, str(target))]
                    vector = np.zeros(DIM, dtype=np.float32)
                    vector[NAMES.index(cls.__name__)] = 1
                    vector[len(NAMES):] = [tick / horizon, min(len(knowledge.known), 100)/100,
                        min(len(space['ip_address']) and sum(space['ip_address'].values()), 100)/100,
                        min(len(knowledge.current_sessions[agent]), 10)/10,
                        zone(host)/2, float(host in knowledge.current_sessions[agent]),
                        min(repeat, 5)/5, float('server' in host),
                        float(agent in ('red_agent_1', 'red_agent_3')),
                        float(agent in ('red_agent_2', 'red_agent_4')),
                        float(bool(target)), min(sum(bool(v) for v in space['subnet'].values()), 10)/10]
                    result.append((agent, action, vector, str(target)))
    return result


def snapshot(env):
    """Privileged evaluator only: never consumed by the policy or candidate builder."""
    state = env.environment_controller.state
    sessions = [s for agent in REDS for s in state.sessions[agent].values() if s.active]
    return {'restricted': any(zone(s.hostname) == 1 for s in sessions),
            'foothold': any(zone(s.hostname) == 2 for s in sessions),
            'privilege': any(zone(s.hostname) == 2 and s.has_privileged_access() for s in sessions),
            'impacts': {h: host.get_impact_count() for h, host in state.hosts.items() if zone(h) == 2}}


def privileged_hosts(env):
    """Privileged evaluator only (like snapshot): hostnames with an active privileged red session."""
    state = env.environment_controller.state
    return {s.hostname for agent in REDS for s in state.sessions[agent].values()
            if s.active and s.has_privileged_access()}


def privilege_gain(action, success, before_hosts, after_hosts):
    """reward v2: PrivilegeEscalate succeeded and a privileged red session newly exists on ITS target host,
    which must be operational (same zone as the v1 privilege milestone)."""
    host = getattr(action, 'hostname', None)
    return (type(action).__name__ == 'PrivilegeEscalate' and bool(success) and host is not None
            and zone(host) == 2 and host in after_hosts and host not in before_hosts)


def confirmed_impact(action, success, before, after):
    """Require the executed Impact to increment its own operational host counter."""
    host = getattr(action, 'hostname', '')
    return (type(action).__name__ == 'Impact' and success and zone(host) == 2
            and after['impacts'].get(host, 0) > before['impacts'].get(host, 0))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(DIM, 128), nn.ReLU(), nn.Linear(128, 128),
                                    nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class Learner:
    def __init__(self, lr, seed, batch, warmup):
        self.online, self.target = Net(), Net()
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = deque(maxlen=20000)
        self.rng = random.Random(seed)
        self.batch, self.warmup, self.updates = batch, warmup, 0
        self.losses = []

    def add(self, feature, reward, next_features, done, ticks):
        self.replay.append((feature, reward, next_features, done, ticks))
        if len(self.replay) < max(self.batch, self.warmup):
            return
        batch = self.rng.sample(self.replay, self.batch)
        features = torch.tensor(np.stack([x[0] for x in batch]))
        with torch.no_grad():
            ys = []
            for _, reward, nxt, done, ticks in batch:
                if len(nxt) and not done:
                    # Double DQN (van Hasselt et al., AAAI 2016, arXiv:1509.06461):
                    # the online network SELECTS the next candidate, the target network EVALUATES it.
                    # (Plain DQN used max over the target network, which overestimates Q.)
                    nxt_t = torch.tensor(nxt)
                    best = int(torch.argmax(self.online(nxt_t)).item())
                    bootstrap = self.target(nxt_t)[best].item()
                else:
                    bootstrap = 0.
                ys.append(reward + (0.97 ** ticks) * bootstrap)
            y = torch.tensor(ys, dtype=torch.float32)
        loss = nn.functional.smooth_l1_loss(self.online(features), y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss; training stopped')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        self.losses.append(float(loss.detach()))
        if self.updates % 100 == 0:
            self.target.load_state_dict(self.online.state_dict())


def episode(seed, cfg, net, eps, learner, blue, scale, policy):
    env = make_env(seed, cfg.steps, blue)
    control = env.environment_controller
    knowledge = Knowledge()
    for agent in REDS:
        knowledge.observe(agent, env.get_observation(agent))
    milestones = Milestones(cfg.weights)
    rng = random.Random(seed + 200000)
    counts, components, latencies = Counter(), Counter(), []
    first_impact, ticks, decisions = None, 0, 0
    max_zone, successes = 0, Counter()
    # Initial state is baseline, not a progress reward.
    previous = snapshot(env)
    reached = {k: previous[k] for k in ('restricted', 'foothold', 'privilege')}
    milestones.seen.update(k for k in ('restricted', 'foothold', 'privilege') if previous[k])
    current = candidates(env, knowledge, ticks, cfg.steps)
    while current and ticks < cfg.steps and not control.done:
        started = time.perf_counter()
        if net is None or rng.random() < eps:
            idx = rng.randrange(len(current))
        else:
            with torch.no_grad():
                scores = net(torch.tensor(np.stack([c[2] for c in current])))
                idx = int(torch.argmax(scores).item())
        if net is not None:
            latencies.append((time.perf_counter() - started) * 1000)
        agent, action, feature, target = current[idx]
        name = type(action).__name__
        counts[name] += 1
        decisions += 1
        knowledge.attempts[(agent, name, target)] += 1
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
        if reached['foothold']:
            max_zone = 2
        elif reached['restricted']:
            max_zone = max(max_zone, 1)
        # Unique observations give a bounded discovery signal. Success is
        # required so RedSessionCheck cannot reward an unrelated failure.
        parts = milestones.reward(flags, discovered=success and new_hosts > 0,
                                  invalid=invalid, failed=failed, ticks=used, scale=scale)
        components.update(parts)
        next_candidates = candidates(env, knowledge, ticks, cfg.steps) if not control.done and ticks < cfg.steps else []
        if learner is not None:
            next_features = np.stack([c[2] for c in next_candidates]) if next_candidates else np.empty((0, DIM), dtype=np.float32)
            learner.add(feature, sum(parts.values()), next_features, not next_candidates, used)
        current, previous = next_candidates, now
    row = {'policy': policy, 'seed': seed, 'blue': 'random' if blue else 'sleep',
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


def save_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=300)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--epsilon-start', type=float, default=1.0)
    p.add_argument('--epsilon-end', type=float, default=0.08)
    p.add_argument('--epsilon-fraction', type=float, default=0.7)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--warmup', type=int, default=64)
    p.add_argument('--checkpoint-interval', type=int, default=50)
    p.add_argument('--curriculum', default='0.2,0.3', help='fraction passive Blue, then active Blue with boosted shaping')
    # DDQN results go to a new folder so the earlier DQN results in rl_cage4/v3 stay intact.
    p.add_argument('--output', type=Path, default=ROOT / 'runs' / 'v3_ddqn')
    for key, value in vars(Weights()).items():
        p.add_argument('--reward-' + key, type=float, default=value)
    cfg = p.parse_args()
    cfg.weights = Weights(**{key: getattr(cfg, 'reward_' + key) for key in vars(Weights())})
    fractions = [float(x) for x in cfg.curriculum.split(',')]
    if (cfg.episodes < 1 or cfg.steps < 4 or cfg.eval_episodes < 1 or cfg.batch_size < 1 or cfg.warmup < 1
            or cfg.checkpoint_interval < 1 or len(fractions) != 2 or min(fractions) < 0
            or sum(fractions) > 1 or not 0 <= cfg.epsilon_end <= cfg.epsilon_start <= 1
            or cfg.epsilon_fraction <= 0 or cfg.lr <= 0 or cfg.eval_seed < cfg.seed + cfg.episodes and cfg.eval_seed + cfg.eval_episodes > cfg.seed
            or not cfg.weights.impact > 2 * max(cfg.weights.restricted, cfg.weights.foothold, cfg.weights.privilege, 0)):
        p.error('Invalid configuration, overlapping seed ranges, or Impact reward not largest')
    cfg.output.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.set_num_threads(1)
    learner = Learner(cfg.lr, cfg.seed, cfg.batch_size, cfg.warmup)
    config = vars(cfg).copy(); config['output'] = str(cfg.output); config['weights'] = vars(cfg.weights)
    (cfg.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    train, start = [], time.perf_counter()
    def checkpoint(path):
        torch.save({'state_dict': learner.online.state_dict(), 'target_state_dict': learner.target.state_dict(),
                    'optimizer': learner.optimizer.state_dict(), 'action_names': NAMES, 'feature_dim': DIM,
                    'config': config, 'updates': learner.updates}, path)
    for ep in range(cfg.episodes):
        progress = ep / cfg.episodes
        blue = progress >= fractions[0]
        scale = 2.0 if progress < sum(fractions) else 1.0
        eps = max(cfg.epsilon_end, cfg.epsilon_start - (cfg.epsilon_start-cfg.epsilon_end)*progress/cfg.epsilon_fraction)
        row = episode(cfg.seed + ep, cfg, learner.online, eps, learner, blue, scale, 'train')
        row.update(episode=ep, epsilon=eps, shaping_scale=scale)
        train.append(row)
        if (ep+1) % cfg.checkpoint_interval == 0:
            checkpoint(cfg.output / f'checkpoint_{ep+1}.pt')
        print(f"train {ep+1}/{cfg.episodes} ticks={row['length_ticks']} return={row['episode_return']:.2f} impact={row['impact_success']} updates={learner.updates}", flush=True)
    elapsed = time.perf_counter()-start
    checkpoint(cfg.output / 'model.pt')
    # Validate deserialization and equivalence before evaluation.
    ckpt = torch.load(cfg.output / 'model.pt', map_location='cpu', weights_only=False)
    evaluated = Net(); evaluated.load_state_dict(ckpt['state_dict']); evaluated.eval()
    test = []
    for policy, net in [('dqn', evaluated), ('random', None)]:
        for i in range(cfg.eval_episodes):
            row = episode(cfg.eval_seed + i, cfg, net, 0., None, True, 1., policy)
            test.append(row)
            print(f"eval {policy} seed={row['seed']} return={row['episode_return']:.2f} impact={row['impact_success']}", flush=True)
    save_csv(cfg.output / 'training.csv', train); save_csv(cfg.output / 'evaluation.csv', test)
    summary = {'training_seconds': elapsed, 'updates': learner.updates,
               'loss_min': min(learner.losses) if learner.losses else None,
               'loss_max': max(learner.losses) if learner.losses else None,
               'evaluation': {policy: {key: float(np.mean([r[key] for r in test if r['policy'] == policy]))
                    for key in ('episode_return','impact_success','restricted','foothold','privilege','invalid_actions','failed_actions','length_ticks')}
                    for policy in ('dqn','random')}}
    (cfg.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot([x['episode'] for x in train], [x['episode_return'] for x in train])
    ax.set(xlabel='Training episode', ylabel='Episode return', title='CAGE-4 v3 training (curriculum rewards)')
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(cfg.output / 'learning_curve.png', dpi=130); plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
