"""Stitch tracks into chains, detect climbs, compare to official end-game values, and render
one verification strip per match (row = chain, 10 crops spread over the match).

usage: identity_run.py KEY TRACKER
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/bodas/data/fgc-vision/src")
from fgc_vision import identity as I  # noqa: E402
from fgc_vision import sources  # noqa: E402

D = Path("/mnt/offload/fgc-vision")


def strip(key: str, res: dict, chains: list[I.Chain], out: Path, n: int = 10) -> None:
    cap = cv2.VideoCapture(str(D / "video" / f"2025_{key}.mp4"))
    fps, s0 = res["fps"], res["start_frame"]
    S = 110
    img = np.zeros((S * len(chains), S * n + 70, 3), np.uint8)
    for r, ch in enumerate(chains):
        smp = ch.samples()
        idx = np.linspace(0, len(smp) - 1, n).astype(int)
        part_of = {}
        for p in ch.parts:
            for t in p.t:
                part_of[t] = p.id
        cv2.putText(img, f"{r}:{ch.side[0]}", (2, r * S + 60), 0, 0.6, (255, 255, 255), 2)
        for c, i in enumerate(idx):
            t, b = smp[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s0 + t * fps))
            ok, f = cap.read()
            if not ok:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            h = max(b[2] - b[0], b[3] - b[1]) * 0.75 + 10
            x0, y0 = int(max(0, cx - h)), int(max(0, cy - h))
            crop = f[y0:int(cy + h), x0:int(cx + h)]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (S - 4, S - 4))
            cv2.putText(crop, f"{t:.0f}s #{part_of.get(t, '?')}", (2, 12), 0, 0.38,
                        (0, 255, 255), 1)
            img[r * S + 2: r * S + S - 2, 70 + c * S + 2: 70 + c * S + S - 2] = crop
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    key, kind = sys.argv[1], sys.argv[2]
    res = json.loads((D / "tracks" / f"{key}.{kind}.json").read_text())
    trs = I.tracklets(res)
    chains = I.stitch(trs)
    m = sources.find_match(2025, key)
    off, st = I.official_end(m), I.stations(m)
    print(f"{key} {kind}: {len(trs)} tracklets in field mask -> {len(chains)} chains; "
          f"tracklets used {sum(len(c.parts) for c in chains)}")
    print("  official:", {s: {i: (st[s].get(i), v) for i, v in off[s].items()} for s in off})
    summary = []
    for ch in chains:
        cl = I.climb(ch)
        cov = I.coverage(ch)
        summary.append({"slot": ch.slot, "side": ch.side, "parts": len(ch.parts),
                        "coverage": round(cov, 3), "climb": cl,
                        "start": [round(v) for v in ch.parts[0].c(0)],
                        "end": [round(v) for v in ch.last()], "end_t": ch.end})
        print(f"  chain {ch.slot} {ch.side:4} parts={len(ch.parts):2d} cov={cov:.2f} "
              f"start={summary[-1]['start']} end={summary[-1]['end']}@{ch.end:.0f}s climb={cl}")
    (D / "tracks" / f"{key}.{kind}.chains.json").write_text(json.dumps(summary, indent=1))
    strip(key, res, chains, D / "tracks" / f"{key}.{kind}.strip.jpg")


if __name__ == "__main__" and len(sys.argv) == 3:
    main()


def joins(key: str, kind: str, out: Path, per_img: int = 24) -> int:
    """Montage every tracklet join: [last crop of previous | first crop of next]."""
    res = json.loads((D / "tracks" / f"{key}.{kind}.json").read_text())
    chains = I.stitch(I.tracklets(res))
    cap = cv2.VideoCapture(str(D / "video" / f"2025_{key}.mp4"))
    fps, s0 = res["fps"], res["start_frame"]
    S = 90

    def crop(t, b):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s0 + t * fps)); ok, f = cap.read()
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        h = max(b[2] - b[0], b[3] - b[1]) * 0.7 + 8
        c = f[int(max(0, cy - h)):int(cy + h), int(max(0, cx - h)):int(cx + h)]
        return cv2.resize(c, (S, S))
    items = []
    for ch in chains:
        for a, b in zip(ch.parts, ch.parts[1:]):
            items.append((ch.slot, crop(a.t[-1], a.box[-1]), crop(b.t[0], b.box[0]), a.t[-1], b.t[0]))
    cols = 4
    for k in range(0, len(items), per_img):
        chunk = items[k:k + per_img]
        rows = (len(chunk) + cols - 1) // cols
        img = np.full((rows * (S + 16), cols * (2 * S + 14), 3), 40, np.uint8)
        for i, (slot, c1, c2, t1, t2) in enumerate(chunk):
            r, c = divmod(i, cols)
            y, x = r * (S + 16) + 14, c * (2 * S + 14)
            img[y:y + S, x:x + S] = c1; img[y:y + S, x + S + 2:x + 2 * S + 2] = c2
            cv2.putText(img, f"J{k + i} c{slot} {t1:.0f}->{t2:.0f}s", (x, y - 3), 0, 0.4, (0, 255, 255), 1)
        cv2.imwrite(str(out).replace(".jpg", f"_{k // per_img}.jpg"), img)
    return len(items)


if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "joins":
    print("joins:", joins(sys.argv[1], sys.argv[2], D / "tracks" / f"{sys.argv[1]}.{sys.argv[2]}.joins.jpg"))
