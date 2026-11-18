"""Scorer handed to the shared episode loop (red.simulator.episode.episode) as its ``net``.

The loop calls ``net(stack_of_candidate_features)`` and takes the argmax, so this wrapper returns the
MAIN head's Q-value per candidate ([N]). With ``record=True`` it additionally computes the Confidence
Gate statistics (tau, sigma) from the full [N, K] Q matrix of that same call and keeps them as
``pending``; the runner's event hook consumes them when the loop emits the 'selected' event, which
supplies the decision number and tick. Recording is read-only: no extra forward pass changes the
scores, no RNG is used.
"""
import numpy as np

from red.agents import confidence


class MainHeadScorer:
    def __init__(self, net, main_head=0, record=False):
        self.net = net
        self.main_head = main_head
        self.record = record
        self.pending = None

    def __call__(self, features):
        q = self.net(features)                      # [N, K]
        if self.record:
            qn = q.detach().cpu().numpy().astype(np.float64)
            self.pending = {'n_candidates': int(qn.shape[0]),
                            'main_action_index': int(qn[:, self.main_head].argmax()),
                            'tau': confidence.agreement(qn, self.main_head),
                            'sigma': confidence.spread(qn, self.main_head)}
        return q[:, self.main_head]

    def consume(self):
        """Return the stats of the latest scored decision once, then clear them."""
        stats, self.pending = self.pending, None
        return stats
