"""Pit photo -> stream crop retrieval, scored on the hand team labels of labels/gt26_sheet.json.
For each pit team in a match: rank all robot boxes on the sheet by cosine similarity to the
pit photo (several embeddings; pit photo used whole and as 5 crops). Report rank of the
best correct box, precision@k (k = number of true boxes) and the chance level."""
import json, os, sys, glob
import cv2, numpy as np, torch
sys.path.insert(0, "/home/bodas/data/fgc-vision/bench"); import gt26 as G
os.environ.setdefault("HF_HOME", "/mnt/offload/fgc-vision/weights/hf")
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
LAB = json.load(open("/home/bodas/data/fgc-vision/labels/gt26_sheet.json"))
def models():
    dm = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float16).eval().cuda(); dp = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.float16).eval().cuda(); cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    def dino(ims):
        x = dp(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        with torch.no_grad(): o = dm(pixel_values=x)
        e = torch.cat([o.last_hidden_state[:, 0], o.last_hidden_state[:, 1:].mean(1)], -1).float(); return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
    def clip(ims):
        x = cp(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        with torch.no_grad(): e = cm.get_image_features(pixel_values=x)
        e = e if torch.is_tensor(e) else e.pooler_output; return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
    return {"dinov2-b": dino, "clip-l14": clip}
def pit_views(path):
    im = cv2.imread(path); h, w = im.shape[:2]; s = min(h, w)
    views = [im, im[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]]
    for fx, fy in ((0.15, 0.2), (0.35, 0.2), (0.15, 0.4), (0.35, 0.4)):
        views.append(im[int(fy*h):int((fy+0.55)*h), int(fx*w):int((fx+0.5)*w)])
    # stream-like: downscale to ~100 px and back up (domain gap probe)
    small = cv2.resize(cv2.resize(views[1], (96, 96), interpolation=cv2.INTER_AREA), (224, 224), interpolation=cv2.INTER_CUBIC)
    views.append(small)
    return [cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in views]
M = models(); out = {}
for key, lab in LAB.items():
    if key.startswith("_"): continue
    res, _, _ = G.load(key); crops, names = [], []
    for t in lab["times"]:
        tt, bs = G.boxes_at(res, float(t)); f = G.frame(key, res, tt)
        for i, r in enumerate(bs):
            n = f"{int(tt)}.{i}"
            if n in lab["N"] or n in lab["D2"]: continue
            crops.append(cv2.cvtColor(G.crop(f, r[1:5], 224), cv2.COLOR_BGR2RGB)); names.append(n)
    teams = [k for k in lab if k not in ("times", "N", "D2")]
    pits = {t: glob.glob(f"/mnt/offload/fgc-vision/2026/pit/{t}_*.jpg") for t in teams}
    for mname, f in M.items():
        C = f(crops)
        for t in teams:
            if not pits[t]: continue
            P = f(pit_views(pits[t][0])); s = (C @ P.T).max(1)
            order = [names[i] for i in np.argsort(-s)]; truth = set(lab[t]); k = len(truth)
            best = min(order.index(n) for n in truth if n in order) + 1
            pk = len(truth & set(order[:k])) / k
            out.setdefault(key, {}).setdefault(t, {})[mname] = {"boxes": len(names), "true": k, "best_rank": best, "precision_at_k": round(pk, 2), "chance_p_at_k": round(k / len(names), 2)}
            print(key, t, mname, out[key][t][mname], flush=True)
json.dump(out, open("/mnt/offload/fgc-vision/2026/pit_reid.json", "w"), indent=1)
