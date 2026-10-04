"""Track clips with each tracker; save JSON to /mnt/offload/fgc-vision/tracks/."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/bodas/data/fgc-vision/src")
from fgc_vision import overlay, track
D = Path("/mnt/offload/fgc-vision")
W = str(D / "runs/detect/runs/yolo11s_1280/weights/best.pt")
(D / "tracks").mkdir(exist_ok=True)
for key in sys.argv[1].split(","):
    clip = str(D / "video" / f"2025_{key}.mp4")
    sp = D / "tracks" / f"{key}.sync.json"
    if sp.exists():
        s = overlay.Sync(**json.loads(sp.read_text()))
    else:
        s = overlay.sync(clip, print); sp.write_text(json.dumps(s.__dict__))
    for kind in sys.argv[2].split(","):
        res = track.run(clip, W, s, kind=kind, conf=0.1, buffer=100, log=print)
        track.save(res, D / "tracks" / f"{key}.{kind}.json")
