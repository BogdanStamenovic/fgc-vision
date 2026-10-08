"""Run every candidate detector on the same frames and save raw detections.

usage: detect_bench.py FRAMES_DIR OUT_JSON [models...]
Each frame is a PNG named <clip>__<sec>.png. Output: {model: {frame: [[x0,y0,x1,y1,score,cls],..]}}
plus per-model seconds/frame. cls is 'robot' or 'ball'.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/offload/fgc-vision/weights/hf")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

W = Path("/mnt/offload/fgc-vision/weights")
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU") != "1" else "cpu"


def yolo_world(name: str):
    from ultralytics import YOLOWorld
    m = YOLOWorld(str(W / name))
    m.set_classes(["robot", "ball"])

    def run(img: Image.Image):
        r = m.predict(img, conf=0.05, imgsz=1280, device=DEV, verbose=False)[0]
        out = []
        for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            out.append([*b, s, ["robot", "ball"][int(c)]])
        return out
    return run


def owlv2():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    mid = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(mid)
    model = Owlv2ForObjectDetection.from_pretrained(mid).to(DEV).eval()
    q = [["a photo of a robot", "a photo of a ball"]]

    def run(img: Image.Image):
        # OWLv2 pads to a square at 960 px, so a 1920x1080 frame is seen at
        # half resolution. Tiling into two halves keeps robots ~2x bigger.
        out = []
        w, h = img.size
        for x0 in (0, w // 2 - 120):
            tile = img.crop((x0, 0, x0 + w // 2 + 120, h))
            inp = proc(text=q, images=tile, return_tensors="pt").to(DEV)
            with torch.no_grad():
                o = model(**inp)
            side = max(tile.size)
            res = proc.post_process_grounded_object_detection(
                o, threshold=0.05, target_sizes=torch.tensor([[side, side]]).to(DEV))[0]
            for b, s, lab in zip(res["boxes"].tolist(), res["scores"].tolist(),
                                 res["labels"].tolist()):
                out.append([b[0] + x0, b[1], b[2] + x0, b[3], s, ["robot", "ball"][lab]])
        return nms(out)
    return run


def gdino():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(DEV).eval()

    def run(img: Image.Image):
        inp = proc(images=img, text="robot. ball.", return_tensors="pt").to(DEV)
        # fp16 autocast: fp32 needs >2.1 GiB at 800x1333, more than the VRAM left
        # beside cvoiced.
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            o = model(**inp)
        res = proc.post_process_grounded_object_detection(
            o, inp.input_ids, threshold=0.1, text_threshold=0.1,
            target_sizes=[img.size[::-1]])[0]
        out = []
        labels = res.get("text_labels", res.get("labels"))
        for b, s, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
            cls = "ball" if "ball" in str(lab) else "robot"
            out.append([*b, s, cls])
        return out
    return run


def nms(dets, thr=0.5):
    dets = sorted(dets, key=lambda d: -d[4])
    keep = []
    for d in dets:
        if all(d[5] != k[5] or iou(d, k) < thr for k in keep):
            keep.append(d)
    return keep


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def yolo_ft(path: str):
    from ultralytics import YOLO
    m = YOLO(path)

    def run(img: Image.Image):
        r = m.predict(img, conf=0.05, imgsz=1280, device=DEV, verbose=False)[0]
        return [[*b, s, ["robot", "ball"][int(c)]] for b, s, c in
                zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    return run


RUNS = Path("/mnt/offload/fgc-vision/runs/detect/runs")
MODELS = {
    "ft-yolo11n": lambda: yolo_ft(str(RUNS / "yolo11n_1280/weights/best.pt")),
    "ft26-yolo11s": lambda: yolo_ft("/mnt/offload/fgc-vision/runs26/y11s/weights/best.pt"),
    "ft-yolo11s": lambda: yolo_ft(str(RUNS / "yolo11s_1280/weights/best.pt")),
    "yolow-s": lambda: yolo_world("yolov8s-worldv2.pt"),
    "yolow-l": lambda: yolo_world("yolov8l-worldv2.pt"),
    "owlv2-b": owlv2,
    "gdino-t": gdino,
}


def main() -> None:
    frames = sorted(Path(sys.argv[1]).glob("*.png"))
    out_p = Path(sys.argv[2])
    names = sys.argv[3:] or list(MODELS)
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    for n in names:
        run = MODELS[n]()
        dets, ts = {}, []
        for f in frames:
            img = Image.open(f).convert("RGB")
            if DEV == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            dets[f.stem] = run(img)
            if DEV == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        res[n + ("@cpu" if DEV == "cpu" else "")] = {
            "dets": dets, "sec_per_frame": float(np.median(ts[1:] or ts)),
            "vram_peak_mib": (torch.cuda.max_memory_allocated() >> 20) if DEV == "cuda" else 0}
        print(n, DEV, f"{np.median(ts[1:] or ts):.3f}s/frame", file=sys.stderr, flush=True)
        del run
        if DEV == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        out_p.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
