"""22-feature vector for one Caldera candidate, in the layout of red.simulator.cage4.candidates().

Layout (cage4.py): 10 action one-hot values in NAMES order, then 12 values. Caldera has no CAGE hosts, subnets, or
session counts, so the 12 values are neutral (0) except the time value tick / horizon.

APPROXIMATION, REVIEW_REQUIRED: the RL values for these vectors are heuristic. They are not the CAGE-4 policy.
"""
import numpy as np

from red.agents.network import FEATURE_DIM
from red.simulator.cage4 import NAMES


def caldera_features(cage_class, tick, horizon):
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    vec[NAMES.index(cage_class)] = 1.0          # action one-hot
    vec[len(NAMES)] = tick / max(horizon, 1)     # time within the run (the first of the 12 values)
    return vec
