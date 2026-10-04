"""Calibrate every 2025 camera in both modes; keep the mode whose tape check is closest to
the 0.50 m regional-zone line (tape not found -> rail residuals decide, flagged)."""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/bodas/data/fgc-vision/src")
from fgc_vision import calibrate as C
D = Path("/mnt/offload/fgc-vision")
CAMS = {'2025_d1f1': ('t2-1', [[150, 650], [640, 372], [1290, 372], [1760, 722]]),
        '2025_d3f1': ('t2-361', [[130, 640], [620, 370], [1300, 370], [1690, 690]]),
        '2025_d1f2': ('t2-16', [[120, 620], [560, 270], [1480, 270], [1780, 600]]),
        '2025_d3f2': ('t2-304', [[120, 590], [580, 340], [1400, 340], [1760, 570]]),
        '2025_d1f3': ('t2-2', [[140, 680], [600, 380], [1360, 380], [1800, 650]]),
        '2025_d3f3': ('t2-308', [[140, 620], [600, 370], [1300, 370], [1760, 630]]),
        '2025_d3f5': ('t4-3', [[140, 610], [600, 330], [1360, 330], [1780, 630]])}
summary = {}
for cam, (k, h) in CAMS.items():
    best = None
    for mode in ("v0", "iter"):
        d = C.calibrate(str(D / f"video/2025_{k}.mp4"), h, None, mode)
        tapes = [v["line_dist_from_rail_m"] for kk, v in d["tape_check"].items()
                 if kk.endswith("_k1") and v["line_dist_from_rail_m"] is not None]
        tape_err = min((abs(t - 0.5) for t in tapes), default=None)
        rails = [v["median_abs_m"] for v in d["rail_residuals"].values()]
        score = (tape_err if tape_err is not None else 1.0) + (0 if len(rails) >= 3 else 1.0)
        print(cam, mode, "k1", round(d["k1"], 3), "rails", d["rail_residuals"], "tape", tapes, flush=True)
        if best is None or score < best[0]:
            best = (score, mode, d, tape_err)
    score, mode, d, tape_err = best
    d["mode"] = mode
    d["accepted"] = tape_err is not None and tape_err < 0.06 and len(d["rail_residuals"]) >= 3
    (D / "calib").mkdir(exist_ok=True)
    (D / "calib" / f"{cam}.json").write_text(json.dumps(d, indent=1))
    C.calibrate(str(D / f"video/2025_{k}.mp4"), h, D / "calib" / f"{cam}.json", mode)
    d2 = json.loads((D / "calib" / f"{cam}.json").read_text()); d2.update(mode=mode, accepted=d["accepted"])
    (D / "calib" / f"{cam}.json").write_text(json.dumps(d2, indent=1))
    summary[cam] = {"mode": mode, "accepted": d["accepted"], "tape_err_m": tape_err,
                    "rails_m": {n: v["median_abs_m"] for n, v in d["rail_residuals"].items()}}
print(json.dumps(summary, indent=1))
(D / "calib" / "summary.json").write_text(json.dumps(summary, indent=1))
