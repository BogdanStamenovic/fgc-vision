from __future__ import annotations

import numpy as np

from fgc_vision import calibrate as C


def test_apply_and_invert_roundtrip() -> None:
    H = np.array([[0.01, 0.002, -1.0], [0.0005, -0.02, 12.0], [0.0, -0.0008, 1.0]])
    p = np.r_[0.05, H.flatten()[:8]]
    xy = np.array([[1.0, 1.0], [3.5, 3.5], [6.0, 2.0]])
    uv = C.invert(p, xy)
    assert np.allclose(C.apply(p, uv), xy, atol=1e-3)
