"""Confidence Gate: decide whether the main head's choice is trusted, from K-head Q-values.

All functions take a Q matrix of shape [N candidates, K heads] (nested lists, numpy array or torch
tensor) and the index of the main head.

    tau   (agreement) = fraction of heads whose argmax candidate equals the main head's argmax candidate
    sigma (spread)    = population std (ddof=0), across heads, of the Q-values of the main head's chosen candidate
    pass              = tau >= theta and sigma <= delta

Ties in argmax resolve to the first (lowest-index) candidate, as in the flat DDQN agent.
"""
import numpy as np


def _matrix(qmatrix):
    q = np.asarray(qmatrix, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] < 1:
        raise ValueError('qmatrix must have shape [N >= 1 candidates, K >= 1 heads]')
    return q


def agreement(qmatrix, main_head=0):
    q = _matrix(qmatrix)
    picks = q.argmax(axis=0)                  # per-head argmax candidate
    return float(np.mean(picks == picks[main_head]))


def spread(qmatrix, main_head=0):
    q = _matrix(qmatrix)
    chosen = int(q[:, main_head].argmax())
    return float(np.std(q[chosen, :]))        # population std


def gate_pass(tau, sigma, theta, delta):
    return bool(tau >= theta and sigma <= delta)


class ConfidenceGate:
    def __init__(self, theta, delta):
        self.theta = theta
        self.delta = delta

    def decide(self, qmatrix, main_head=0):
        q = _matrix(qmatrix)
        tau = agreement(q, main_head)
        sigma = spread(q, main_head)
        return {'action': int(q[:, main_head].argmax()), 'tau': tau, 'sigma': sigma,
                'passed': gate_pass(tau, sigma, self.theta, self.delta)}
