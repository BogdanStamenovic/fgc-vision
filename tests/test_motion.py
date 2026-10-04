from __future__ import annotations

import numpy as np

from fgc_vision import motion


def test_straight_line_speed_and_path() -> None:
    t = np.arange(0, 10, 0.1)
    trk = np.c_[t, 1 + 0.5 * t, np.full_like(t, 2.0)]   # 0.5 m/s for ~10 s
    m = motion.metrics(trk)
    assert abs(m["mean_speed_mps"] - 0.5) < 0.02
    assert abs(m["path_m"] - 0.5 * m["seconds_observed"]) < 0.05
    assert m["idle_seconds"] == 0.0


def test_jitter_does_not_become_speed() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, 0.1)
    # 1.5 cm: the worst per-axis std measured on stationary robots before "go" (t2-1)
    trk = np.c_[t, 3 + rng.normal(0, 0.015, len(t)), 3 + rng.normal(0, 0.015, len(t))]
    m = motion.metrics(trk)
    assert m["idle_seconds"] > 0.6 * m["seconds_observed"]


def test_gaps_are_not_bridged() -> None:
    t = np.r_[np.arange(0, 3, 0.1), np.arange(10, 13, 0.1)]
    x = np.r_[np.zeros(30), np.full(30, 5.0)]               # teleport during the gap
    m = motion.metrics(np.c_[t, x, np.zeros_like(t)])
    assert m["path_m"] < 0.5


def test_duplicate_timestamps_are_dropped() -> None:
    t = np.arange(0, 5, 0.1)
    a = np.c_[t, 0.3 * t, np.zeros_like(t)]
    m = motion.metrics(np.r_[a, a + [0, 0.01, 0]])
    assert np.isfinite(m["rms_jerk_mps3"])
