"""Fetch a time window of a post-live / live YouTube stream by DASH fragment number.

Why: `yt-dlp --download-sections` fails on post_live DASH (ffmpeg exit 183), and
`--live-from-start` pulls the whole stream (≈19 GB at 1080p for a 7 h field day).
yt-dlp's own JSON lists every fragment URL (`&sq=N`, `target_duration` 5 s), so a match
window is ~40 fragments, fetched exactly as yt-dlp's DASH downloader would fetch them.
Live segments are self-initialising fMP4, so concatenation plays; ffmpeg remuxes to a
clean mp4 with timestamps starting at 0.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .sources import ytdlp


class DashError(RuntimeError):
    pass


def formats(video: str, cache: Path | None = None, max_age: float = 3 * 3600) -> dict:
    if cache and cache.exists() and time.time() - cache.stat().st_mtime < max_age:
        return json.loads(cache.read_text())
    p = subprocess.run([ytdlp(), "--js-runtimes", "node", "--no-warnings", "-J", "--", video],
                       capture_output=True, text=True, check=False, timeout=180)
    if p.returncode != 0:
        raise DashError(f"yt-dlp -J {video}: {p.stderr.strip()[-300:]}")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(p.stdout)
    return json.loads(p.stdout)


def fetch_window(video: str, t0: float, t1: float, out: Path, fmt: str = "299",
                 cache_dir: Path | None = None,
                 log: Callable[[str], None] = lambda s: None) -> tuple[Path, float]:
    """Download [t0, t1] seconds of the stream. Returns (path, clip_offset_s): the stream
    time of the clip's first frame (fragment-aligned, so <= t0)."""
    info = formats(video, cache_dir / f"{video}.json" if cache_dir else None)
    f = next((x for x in info["formats"] if x["format_id"] == fmt), None)
    if f is None or not f.get("fragments"):
        raise DashError(f"{video}: no fragmented format {fmt}")
    dur = float(f.get("target_duration") or 5)
    frags = f["fragments"]
    a, b = int(t0 // dur), min(len(frags) - 1, int(t1 // dur) + 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        raw = Path(td) / "raw.mp4"
        with raw.open("wb") as fh:
            for i in range(a, b + 1):
                for attempt in range(3):
                    try:
                        req = urllib.request.Request(frags[i]["url"],
                                                     headers=f.get("http_headers") or {})
                        with urllib.request.urlopen(req, timeout=60) as r:
                            fh.write(r.read())
                        break
                    except urllib.error.HTTPError as e:
                        if e.code in (403, 429):
                            raise DashError(f"YouTube refused fragment {i} ({e.code}); "
                                            "real stop, not retried") from e
                        if attempt == 2:
                            raise
                        time.sleep(3)
        p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c", "copy",
                            "-movflags", "+faststart", str(out)],
                           capture_output=True, text=True, check=False)
        if p.returncode != 0 or not out.exists():
            raise DashError(f"remux failed: {p.stderr[-300:]}")
    log(f"{video} sq {a}-{b} -> {out.name} ({out.stat().st_size >> 20} MiB)")
    return out, a * dur
