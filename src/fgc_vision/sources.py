"""Where a match is on video, and fetching just that piece of the stream.

Stream discovery and match alignment belong to fgc-matchwatch; this module only
consumes them. Placement comes from `fgc-matchwatch align <video>` (cached as
JSON per stream), which is accurate to about 3 s. The overlay timer
(`overlay.py`) then pins the start to the frame.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA = Path(os.environ.get("FGC_VISION_DATA", "/mnt/offload/fgc-vision"))
MATCHWATCH = Path(os.environ.get("FGC_MATCHWATCH_DATA", "/mnt/offload/fgc-matchwatch"))
MATCHWATCH_BIN = os.environ.get("FGC_MATCHWATCH_BIN",
                                str(Path.home() / "data/fgc-matchwatch/.venv/bin/fgc-matchwatch"))
API = "https://api.first.global/v1?year={year}"

# 1080p60 H.264 DASH. H.264 rather than VP9 because OpenCV/ffmpeg decode it
# about twice as fast on this CPU, and decode is the CPU-side bottleneck.
FORMAT = "299/137/298/136"
PRE_S = 15.0     # matchwatch is within ~3 s; keep slack for the countdown
MATCH_S = 150.0
POST_S = 15.0


class SourceError(RuntimeError):
    pass


@dataclass
class Placement:
    key: str
    video: str
    start: float          # seconds into the stream (matchwatch's estimate)
    confident: bool
    field: int | None
    title: str


def ytdlp() -> str:
    here = MATCHWATCH / "venv/bin/yt-dlp"
    if here.exists():
        return str(here)
    found = shutil.which("yt-dlp")
    if not found:
        raise SourceError("yt-dlp not found")
    return found


def official(year: int, max_age: float = 3600) -> dict[str, Any]:
    cache = DATA / f"api{year}.json"
    import time
    if cache.exists() and time.time() - cache.stat().st_mtime < max_age:
        return json.loads(cache.read_text())
    if year == 2025 and (MATCHWATCH / "api2025.json").exists():
        d = json.loads((MATCHWATCH / "api2025.json").read_text())
    else:
        with urllib.request.urlopen(API.format(year=year), timeout=60) as r:
            d = json.loads(r.read())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d))
    return d


def match_key(m: dict[str, Any]) -> str:
    return f"{m['tournamentKey']}-{m['id']}"


def find_match(year: int, key: str) -> dict[str, Any]:
    for m in official(year)["matches"]:
        if match_key(m) == key:
            return m
    raise SourceError(f"no official match {key} in {year}")


def streams(year: int) -> dict[str, Any]:
    p = MATCHWATCH / f"streams-{year}.json"
    if not p.exists():
        raise SourceError(f"{p} missing: run `fgc-matchwatch streams discover` first")
    return json.loads(p.read_text())


def alignment(year: int, video: str, refresh: bool = False) -> list[dict[str, Any]]:
    cache = DATA / "align" / f"{year}_{video}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    p = subprocess.run([MATCHWATCH_BIN, "--year", str(year), "-q", "align", video],
                       capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise SourceError(f"fgc-matchwatch align {video} failed: {p.stderr.strip()[-300:]}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(p.stdout)
    return json.loads(p.stdout)


def place(year: int, key: str, log: Callable[[str], None] = lambda s: None) -> Placement:
    """Find a match in the streams. Field streams first, then the main feed."""
    m = find_match(year, key)
    cands = sorted(streams(year).values(),
                   key=lambda s: (s.get("field") != m.get("field"), s.get("field") is None))
    for s in cands:
        if s.get("field") not in (None, m.get("field")):
            continue
        try:
            al = alignment(year, s["video"])
        except SourceError as e:
            log(f"skip {s['video']}: {e}")
            continue
        for a in al:
            if a["key"] == key:
                return Placement(key, s["video"], float(a["start"]), bool(a["confident"]),
                                 s.get("field"), s.get("title", ""))
    raise SourceError(f"{key} is not placed in any transcribed stream")


def clip_path(key: str, year: int) -> Path:
    return DATA / "video" / f"{year}_{key}.mp4"


def fetch(year: int, key: str, log: Callable[[str], None] = lambda s: None,
          video: str | None = None, start: float | None = None) -> tuple[Path, Placement]:
    """Download [start-PRE_S, start+MATCH_S+POST_S] of the match at 1080p."""
    if video is None or start is None:
        pl = place(year, key, log)
    else:
        pl = Placement(key, video, start, True, None, "manual")
    out = clip_path(key, year)
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists():
        return out, pl
    out.parent.mkdir(parents=True, exist_ok=True)
    a, b = max(0.0, pl.start - PRE_S), pl.start + MATCH_S + POST_S
    log(f"{key}: {pl.video} {a:.0f}-{b:.0f}s")
    tmpl = str(out.with_suffix("")) + ".%(ext)s"
    p = subprocess.run([ytdlp(), "--js-runtimes", "node", "--no-warnings", "-q", "--no-progress",
                        "-f", FORMAT, "--download-sections", f"*{a:.2f}-{b:.2f}",
                        "-o", tmpl, "--", pl.video], capture_output=True, text=True, check=False)
    if p.returncode != 0 or not out.exists():
        raise SourceError(f"download of {key} failed: {p.stderr.strip()[-300:]}")
    meta.write_text(json.dumps({"key": key, "video": pl.video, "stream_offset": a,
                                "matchwatch_start": pl.start, "confident": pl.confident,
                                "field": pl.field, "title": pl.title}))
    return out, pl
