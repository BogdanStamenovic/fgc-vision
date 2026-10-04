"""Command-line interface for fgc-vision.

stdout carries only JSON results; progress, warnings and errors go to stderr.
Exit codes: 0 success, 1 an operation failed, 2 usage error, 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import NoReturn

from . import __version__
from .sources import DATA, SourceError


class _UsageError(Exception):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


DEFAULT_WEIGHTS = str(DATA / "runs/detect/runs/yolo11s_1280/weights/best.pt")


def _build_parser() -> argparse.ArgumentParser:
    p = _ArgumentParser(prog="fgc-vision", description=(
        "Detect, track and identify robots in FIRST Global Challenge field livestreams."))
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("-v", "--verbose", action="store_true", help="print detailed progress")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress non-error output")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("place", help="where a match is in the streams (via fgc-matchwatch)")
    s.add_argument("key")
    s = sub.add_parser("fetch", help="download the match's piece of its field stream (1080p)")
    s.add_argument("key")
    s.add_argument("--video", help="override: YouTube id")
    s.add_argument("--start", type=float, help="override: match start, seconds into the video")
    s.add_argument("--dry-run", action="store_true", help="only print what would be fetched")
    s = sub.add_parser("check-mapping", help=(
        "which station each per-robot official field belongs to (vs ranking totals)"))
    s.add_argument("--tournament", default="t2")
    s = sub.add_parser("sync", help="match start frame of a clip from the overlay timer")
    s.add_argument("clip")
    s = sub.add_parser("track", help="detect + track robots through a fetched match")
    s.add_argument("key")
    s.add_argument("--tracker", choices=["bytetrack", "botsort"], default="botsort")
    s.add_argument("--weights", default=DEFAULT_WEIGHTS)
    s.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    s = sub.add_parser("chains", help="stitch tracks into per-robot chains + climb detection")
    s.add_argument("key")
    s.add_argument("--tracker", choices=["bytetrack", "botsort"], default="botsort")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        print(f"fgc-vision: error: {exc}", file=sys.stderr)
        return 2

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr)

    def out(obj: object) -> None:
        print(json.dumps(obj, indent=1, default=str))

    try:
        from . import sources
        if args.cmd == "place":
            out(sources.place(args.year, args.key, log).__dict__)
        elif args.cmd == "fetch":
            if args.dry_run:
                pl = (sources.place(args.year, args.key, log) if args.video is None
                      else sources.Placement(args.key, args.video, args.start or 0, True, None, ""))
                out({"would_fetch": pl.__dict__, "to": str(sources.clip_path(args.key, args.year)),
                     "window": [pl.start - sources.PRE_S,
                                pl.start + sources.MATCH_S + sources.POST_S]})
            else:
                path, _ = sources.fetch(args.year, args.key, log, args.video, args.start)
                out({"clip": str(path)})
        elif args.cmd == "check-mapping":
            from . import mapping
            found = mapping.check(sources.official(args.year), args.tournament)
            out({"year": args.year, "best": found[:1], "top": found[:8],
                 "note": "perm '123' means Robot One/Two/Three = station x1/x2/x3"})
            if not found or found[0]["exact"] < found[0]["teams"]:
                log("no permutation reproduces a ranking total for every team")
                return 1
        elif args.cmd == "sync":
            from . import overlay
            out(overlay.sync(args.clip, log).__dict__)
        elif args.cmd == "track":
            from . import overlay, track
            clip = sources.clip_path(args.key, args.year)
            if not clip.exists():
                log(f"{clip} missing: run `fgc-vision fetch {args.key}` first")
                return 1
            tdir = DATA / "tracks"
            tdir.mkdir(parents=True, exist_ok=True)
            sp = tdir / f"{args.key}.sync.json"
            sy = (overlay.Sync(**json.loads(sp.read_text())) if sp.exists()
                  else overlay.sync(str(clip), log))
            sp.write_text(json.dumps(sy.__dict__))
            dev: str | int = int(args.device) if args.device.isdigit() else args.device
            res = track.run(str(clip), args.weights, sy, kind=args.tracker, conf=0.1,
                            device=dev, log=log)
            dest = tdir / f"{args.key}.{args.tracker}.json"
            track.save(res, dest)
            out({"tracks": str(dest), "frames": len(res["frames"]),
                 "ms_per_frame_total": round(res["ms_per_frame_total"], 1)})
        elif args.cmd == "chains":
            from . import identity as ident
            res = json.loads((DATA / "tracks" / f"{args.key}.{args.tracker}.json").read_text())
            chains = ident.stitch(ident.tracklets(res))
            m = sources.find_match(args.year, args.key)
            out({"key": args.key, "official_end": ident.official_end(m),
                 "stations": ident.stations(m),
                 "chains": [{"slot": c.slot, "side": c.side, "parts": len(c.parts),
                             "coverage": round(ident.coverage(c), 3), "climb": ident.climb(c)}
                            for c in chains],
                 "warning": "identity measured unusable on 2025 footage (31-46% wrong joins, 17% of match "
                            "time verifiably identified); do not attribute per-team stats"})
        return 0
    except (SourceError, OSError, ValueError, RuntimeError) as exc:
        print(f"fgc-vision: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
