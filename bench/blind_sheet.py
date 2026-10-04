"""Blind test of the consensus: 24 team rows drawn at random from the real assignment or the
shuffled control, labelled only R0..R23. Key goes to pre/blind_key.json (do not read before
scoring). usage: blind_sheet.py EMB_TAG SEED"""
import json, random, sys, cv2, numpy as np
from collections import defaultdict
D = "/mnt/offload/fgc-vision/pre"; tag, seed = sys.argv[1], int(sys.argv[2])
z = np.load(f"{D}/galleries.npz"); crops = z["crops"]
def rows_of(path):
    res = json.load(open(path)); tr = defaultdict(list)
    for a in res["alliances"]:
        for r, t in zip(a["rows"], a["spot_team"]): tr[t].append(r)
    return {t: rs for t, rs in tr.items() if len(rs) >= 4}
real, ctrl = rows_of(f"{D}/consensus_{tag}_c.json"), rows_of(f"{D}/consensus_{tag}_shuffled_c.json")
rng = random.Random(seed); items = []
teams_r = rng.sample(sorted(real), 12); teams_c = rng.sample(sorted(ctrl), 12)
items = [("real", t, real[t]) for t in teams_r] + [("ctrl", t, ctrl[t]) for t in teams_c]
rng.shuffle(items)
S = 100; img = np.full((len(items) * (S + 4), 60 + 6 * S, 3), 30, np.uint8)
for i, (_, _, rs) in enumerate(items):
    cv2.putText(img, f"R{i}", (4, i * (S + 4) + 55), 0, 0.7, (0, 255, 255), 2)
    for c, row in enumerate(rng.sample(rs, min(6, len(rs)))):
        img[i * (S + 4):i * (S + 4) + S, 60 + c * S:60 + (c + 1) * S] = cv2.resize(crops[row][0], (S, S))
cv2.imwrite(f"{D}/blind_{seed}.jpg", img)
json.dump([{"row": i, "kind": k, "team": t, "n": min(6, len(rs))} for i, (k, t, rs) in enumerate(items)], open(f"{D}/blind_key_{seed}.json", "w"))
print("sheet written; rows", len(items))
