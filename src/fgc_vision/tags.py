"""Human-in-the-loop identity via fgc-scout's tag endpoints.

fgc-vision asks; scouts answer in the app. Requests are idempotent by id, so re-running a
match re-posts nothing new. Endpoints (fgc-scout server.mjs):
  POST /api/tag-image?id=<id>     jpeg body
  POST /api/tag-request           {"id","matchKey","kind","t","prompt","context",
                                   "candidates","imageId","priority"}
  GET  /api/tags?matchKey=<key>   requests with "answer" (team code or "none")
Auth: header X-Scout-Key, same as fgc-matchwatch. URL and key come from SCOUT_URL /
SCOUT_KEY (fgc-matchwatch's matchwatch.env).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


class TagError(RuntimeError):
    pass


@dataclass
class TagRequest:
    matchKey: str
    kind: str                 # start | join | end | unsure
    t: float                  # match seconds
    prompt: str
    candidates: list[str]
    priority: float = 0.0
    context: str = ""
    imageId: str = ""
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            h = hashlib.sha1(f"{self.matchKey}|{self.kind}|{self.t:.1f}|{self.prompt}".encode())
            self.id = f"v-{self.matchKey}-{h.hexdigest()[:16]}"


def crop_jpeg(frame: np.ndarray, box: list[float], ctx: float = 1.6, size: int = 360) -> bytes:
    """The robot in question in an orange box, with context around it."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = max(x1 - x0, y1 - y0) * ctx
    a, b = int(max(0, cx - h)), int(max(0, cy - h))
    c, d = int(min(frame.shape[1], cx + h)), int(min(frame.shape[0], cy + h))
    img = frame.copy()
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 140, 255), 3)
    out = cv2.resize(img[b:d, a:c], (size, size), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise TagError("jpeg encode failed")
    return buf.tobytes()


class Client:
    def __init__(self, url: str | None = None, key: str | None = None, dry_run: bool = False):
        self.url = (url or os.environ.get("SCOUT_URL", "http://127.0.0.1:3077")).rstrip("/")
        self.key = key if key is not None else os.environ.get("SCOUT_KEY", "")
        self.dry_run = dry_run
        self.sent: list[dict[str, Any]] = []

    def _req(self, path: str, data: bytes | None = None, ctype: str = "application/json") -> Any:
        if self.dry_run and data is not None:
            self.sent.append({"path": path, "bytes": len(data)})
            return {"ok": True, "dry_run": True}
        r = urllib.request.Request(f"{self.url}{path}", data=data,
                                   method="POST" if data is not None else "GET",
                                   headers={"X-Scout-Key": self.key, "Content-Type": ctype})
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise TagError(f"{path} -> {e.code}: {e.read()[:200]!r}") from e

    def ask(self, req: TagRequest, jpeg: bytes | None = None) -> str:
        if jpeg is not None:
            if not req.imageId:
                req.imageId = req.id
            self._req(f"/api/tag-image?id={req.imageId}", jpeg, "image/jpeg")
        self._req("/api/tag-request", json.dumps(asdict(req)).encode())
        return req.id

    def answers(self, match_key: str) -> dict[str, str]:
        d = self._req(f"/api/tags?matchKey={match_key}")
        return {t["id"]: t["answer"] for t in d.get("tags", []) if t.get("answer")}
