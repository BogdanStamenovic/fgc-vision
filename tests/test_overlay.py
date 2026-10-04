from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from fgc_vision import overlay


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_reads_synthetic_timer() -> None:
    f = np.zeros((1080, 1920, 3), np.uint8)
    x0, y0, x1, y1 = overlay.ROI["timer"]
    f[y0:y1, x0:x1] = (0, 170, 255)
    cv2.putText(f, "1:15", (x0 + 25, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 7)
    assert overlay.remaining(f) == 75


def test_sync_time_mapping() -> None:
    s = overlay.Sync(start_frame=900, fps=60.0, flips_seen=1, readings=1)
    assert s.t(960) == 1.0
