"""Assign fragments to start galleries with several embedding models; score against
labels/frag_ids_<key>.json. usage: gallery_classify.py KEY"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, torch
D = Path("/mnt/offload/fgc-vision"); os.environ.setdefault("HF_HOME", str(D / "weights/hf"))
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

def clip_embed(name):
    m = CLIPModel.from_pretrained(name, torch_dtype=torch.float16).eval().cuda(); p = CLIPProcessor.from_pretrained(name)
    def f(imgs):
        x = p(images=list(imgs), return_tensors="pt")["pixel_values"].cuda().half()
        with torch.no_grad(): e = m.get_image_features(pixel_values=x)
        e = e if torch.is_tensor(e) else e.pooler_output
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
    return f

def dino_embed(name):
    m = AutoModel.from_pretrained(name, torch_dtype=torch.float16).eval().cuda(); p = AutoImageProcessor.from_pretrained(name)
    def f(imgs):
        x = p(images=list(imgs), return_tensors="pt")["pixel_values"].cuda().half()
        with torch.no_grad(): o = m(pixel_values=x)
        e = torch.cat([o.last_hidden_state[:, 0], o.last_hidden_state[:, 1:].mean(1)], -1)
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
    return f

MODELS = {"clip-b32": lambda: clip_embed("openai/clip-vit-base-patch32"),
          "clip-l14": lambda: clip_embed("openai/clip-vit-large-patch14"),
          "dinov2-b": lambda: dino_embed("facebook/dinov2-base"),
          "dinov2-l": lambda: dino_embed("facebook/dinov2-large")}

def main():
    key = sys.argv[1]
    z = np.load(D / "reid" / f"{key}.npz"); gal, frag = z["gal"], z["frag"]; meta = json.loads(str(z["meta"]))
    lab = json.loads(Path(f"/home/bodas/data/fgc-vision/labels/frag_ids_{key}.json").read_text())
    truth = {i: int(k[1]) for k, v in lab.items() if k.startswith("G") for i in v}
    nonrob = set(lab.get("N", []))
    rgb = lambda a: a[..., ::-1].copy()
    out = {}
    for name, mk in MODELS.items():
        f = mk()
        G = np.stack([f(rgb(g)) for g in gal])            # 6 x 12 x d
        F = np.stack([f(rgb(x)) for x in frag])           # n x 8 x d
        gm = G.mean(1); gm /= np.linalg.norm(gm, axis=1, keepdims=True)
        fm = F.mean(1); fm /= np.linalg.norm(fm, axis=1, keepdims=True)
        S = fm @ gm.T                                     # n x 6
        pred = S.argmax(1)
        # temporal exclusivity: greedy by margin; overlapping fragments may not share a robot
        order = np.argsort(-(np.sort(S, 1)[:, -1] - np.sort(S, 1)[:, -2]))
        fr = meta["frag"]; assigned = {}
        for i in order:
            for g in np.argsort(-S[i]):
                if all(not (fr[j]["t0"] < fr[i]["t1"] and fr[i]["t0"] < fr[j]["t1"]) for j, gg in assigned.items() if gg == g):
                    assigned[i] = int(g); break
            else: assigned[i] = int(pred[i])
        acc = lambda P: np.mean([P[i] == t for i, t in truth.items()])
        side = lambda P: np.mean([(P[i] < 3) == (t < 3) for i, t in truth.items()])
        top = S.max(1)
        rob_top = [top[i] for i in truth]; non_top = [top[i] for i in nonrob]
        auc = np.mean([(a > b) + 0.5 * (a == b) for a in rob_top for b in non_top]) if non_top else float("nan")
        out[name] = {"acc_argmax": acc(pred), "acc_exclusive": acc(assigned), "side_acc": side(pred),
                     "nonrobot_auc": auc, "n": len(truth)}
        per = {g: f"{sum(pred[i] == g for i, t in truth.items() if t == g)}/{sum(1 for t in truth.values() if t == g)}" for g in sorted(set(truth.values()))}
        print(f"{name:9} argmax {acc(pred):.2f}  exclusive {acc(assigned):.2f}  alliance {side(pred):.2f}  robot-vs-nonrobot AUC {auc:.2f}  per-robot {per}", flush=True)
        del f; torch.cuda.empty_cache()
    print("chance: 1/6 = 0.17 (alliance 0.50)")
    (D / "reid" / f"{key}.classify.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__": main()
