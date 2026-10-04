"""Image -> field-floor coordinates in metres, one homography per fixed camera.

Field frame (2025, 7 m x 7 m, manual 2025 section 2): origin at the front-left inner
corner of the guardrail as seen from the field camera, x to the right along the front
rail, y away from the camera toward the back rail (MITIGATORS, DISPENSERS). Red alliance
stands on the left side (x = 0), blue on the right (x = 7).

Calibration points are clicked by eye on a 1080p frame and stored in
`calib/<camera>.json` as {"img": [[u, v], ...], "field": [[x, y], ...], "check": {...}}.
Only floor-level points are valid: a point above the floor (tower tops, AprilTags, a robot's
top edge) projects to the wrong place. A tracked robot is therefore located by the bottom
centre of its box, which is roughly where it touches the floor; for a robot hanging on a
rope this is wrong by design and such samples are flagged by the caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CALIB = Path(__file__).resolve().parent / "calib"
FIELD_2025 = (7.0, 7.0)


@dataclass
class Camera:
    name: str
    H: np.ndarray          # image (u, v, 1) -> field (x, y, w)
    reproj_px: float       # RMS reprojection error of the calibration points, image px
    check: dict

    def to_field(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        p = np.c_[uv, np.ones(len(uv))] @ self.H.T
        return p[:, :2] / p[:, 2:3]

    def to_image(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        p = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(self.H).T
        return p[:, :2] / p[:, 2:3]


def fit(img: list[list[float]], fld: list[list[float]]) -> tuple[np.ndarray, float]:
    src = np.asarray(img, np.float64)
    dst = np.asarray(fld, np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    back = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    return H, float(np.sqrt(((back - src) ** 2).sum(1).mean()))


def load(name: str, root: Path = CALIB) -> Camera:
    d = json.loads((root / f"{name}.json").read_text())
    H, err = fit(d["img"], d["field"])
    return Camera(name, H, err, d.get("check", {}))


def foot(box: list[float]) -> tuple[float, float]:
    """Bottom centre of a box: the robot's floor contact, as seen from a front camera."""
    return (box[0] + box[2]) / 2, box[3]


# 2025 per-stream camera: the per-field cameras did not move within a day, but field 5
# was reframed between day 1 and day 3, so calibration is per (day, field).
STREAM_CAMERA_2025 = {
    "Hy2VGJjoMoo": "2025_d1f1", "t4IdgPIlFyg": "2025_d1f2", "YRncWpIEcEQ": "2025_d1f3",
    "XJjwMkiiwuQ": "2025_d1f4", "bipuBmCye9g": "2025_d1f5",
    "FbAeVwfciMQ": "2025_d3f1", "EMmcTueSVu0": "2025_d3f2", "-fznHfw67Mg": "2025_d3f3",
    "W_ZVyhQoIOg": "2025_d3f4", "Js0dd0Aytj4": "2025_d3f5",
}
