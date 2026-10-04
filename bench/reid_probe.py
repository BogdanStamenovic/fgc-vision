"""Does appearance separate same-robot joins from switches? AUC of several similarities
on the hand-judged joins of t2-1 BoT-SORT (labels/joins_t2-1_botsort.json)."""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, "/home/bodas/data/fgc-vision/src")
from fgc_vision import identity as I
D = Path("/mnt/offload/fgc-vision")
res = json.loads((D / "tracks/t2-1.botsort.json").read_text())
chains = I.stitch(I.tracklets(res))
cap = cv2.VideoCapture(str(D / "video/2025_t2-1.mp4")); fps, s0 = res["fps"], res["start_frame"]
def crop(t, b, pad=0.1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(s0 + t * fps)); ok, f = cap.read()
    w, h = b[2]-b[0], b[3]-b[1]
    return f[int(max(0,b[1]-pad*h)):int(b[3]+pad*h), int(max(0,b[0]-pad*w)):int(b[2]+pad*w)]
pairs = []
for ch in chains:
    for a, b in zip(ch.parts, ch.parts[1:]):
        # several crops per side: last/first 5 samples, robust to one bad box
        A = [crop(t, x) for t, x in list(zip(a.t, a.box))[-5:]]
        B = [crop(t, x) for t, x in list(zip(b.t, b.box))[:5]]
        pairs.append((A, B))
lab = json.loads(Path("/home/bodas/data/fgc-vision/labels/joins_t2-1_botsort.json").read_text())
y = {i: 1 for i in lab["S"]} | {i: 0 for i in lab["X"] + lab["Xp"]}
def hist(c):
    hsv = cv2.cvtColor(cv2.resize(c, (64, 64)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]); return cv2.normalize(h, h).flatten()
from transformers import CLIPModel, CLIPProcessor
import os; os.environ.setdefault("HF_HOME", str(D / "weights/hf"))
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda(); p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
def emb(cs):
    x = p(images=[cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in cs], return_tensors="pt")["pixel_values"].cuda()
    with torch.no_grad(): e = m.get_image_features(pixel_values=x)
    e = e if torch.is_tensor(e) else (e.pooler_output if getattr(e, "image_embeds", None) is None else e.image_embeds)
    e = e if e.shape[-1] == 512 else m.visual_projection(e)
    e = e / e.norm(dim=-1, keepdim=True); return e.mean(0)
def size(cs): return np.median([c.shape[0] * c.shape[1] for c in cs])
feats = {"hsv_hist": [], "clip": [], "size_ratio": []}
for A, B in pairs:
    feats["hsv_hist"].append(np.mean([cv2.compareHist(hist(a), hist(b), cv2.HISTCMP_CORREL) for a in A for b in B]))
    feats["clip"].append(float(emb(A) @ emb(B)))
    sa, sb = size(A), size(B); feats["size_ratio"].append(min(sa, sb) / max(sa, sb))
def auc(s, yy):
    pos = [v for v, t in zip(s, yy) if t]; neg = [v for v, t in zip(s, yy) if not t]
    return np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
idx = sorted(y); yy = [y[i] for i in idx]
print(f"{len(pairs)} joins, {sum(yy)} same / {len(yy)-sum(yy)} switch")
for k, v in feats.items():
    s = [v[i] for i in idx]
    print(f"  {k:10} AUC {auc(s, yy):.2f}  same mean {np.mean([a for a,t in zip(s,yy) if t]):.3f}  switch mean {np.mean([a for a,t in zip(s,yy) if not t]):.3f}")
