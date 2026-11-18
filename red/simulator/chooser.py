"""Action-selection seam for the instrumented episode loop.

``episode()`` in ``red/simulator/episode.py`` selects one candidate per
decision. This module lifts that decision behind a small interface so later
work (confidence gate, LLM, router) can swap the selector without touching the
loop. The default adapter, :class:`EpsilonGreedyChooser`, reproduces the
original inline logic verbatim -- including the RNG call order -- so an episode
driven by it is identical to the pre-seam loop.
"""
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class Choice:
    """One selection: which candidate, who chose it, and optional metadata.

    ``index`` indexes into the candidate list passed to ``choose``. ``source``
    labels who made the choice ('random', 'rl', and later 'llm', 'api',
    'defer'). ``meta`` carries adapter-specific fields (e.g. tau/sigma) for the
    decision log; it is empty for the default chooser.
    """

    index: int
    source: str
    meta: dict = field(default_factory=dict)


class EpsilonGreedyChooser:
    """Default selector: epsilon-greedy over a Q-network, or uniform random.

    Mirrors ``red/simulator/episode.py`` (pre-seam) exactly::

        net is None OR rng.random() < eps  -> rng.randrange(len(candidates))
        otherwise                          -> argmax of net(features)

    The same ``rng`` object and call order are preserved, so results are
    unchanged. ``knowledge`` and ``tick`` are part of the interface for future
    adapters and ignored here.
    """

    def __init__(self, net, eps, rng):
        self.net = net
        self.eps = eps
        self.rng = rng

    def choose(self, candidates, knowledge=None, tick=None):
        if self.net is None or self.rng.random() < self.eps:
            idx = self.rng.randrange(len(candidates))
            source = 'random'
        else:
            with torch.no_grad():
                scores = self.net(torch.tensor(np.stack([c[2] for c in candidates])))
                idx = int(torch.argmax(scores).item())
            source = 'rl'
        return Choice(index=idx, source=source)
