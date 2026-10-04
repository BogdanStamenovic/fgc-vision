"""Cross-match appearance consensus: which start robot is which team.

Each match-alliance has 3 gallery robots and 3 known teams; the unknown is a permutation.
Objective: sum over teams of pairwise cosine similarity between the embeddings assigned to
that team in all its matches. Coordinate ascent over the 6 permutations per alliance, 30
random restarts, best objective kept. Alliances without exactly 3 gallery robots are skipped.

usage: consensus.py EMBEDDING [--shuffle]   (--shuffle = control with random teams)
Writes pre/consensus_<emb>.json and a per-team montage pre/teams_<emb>_<n>.jpg.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

D = Path("/mnt/offload/fgc-vision")
PERMS = list(itertools.permutations(range(3)))


def main() -> None:
    emb_name = sys.argv[1]
    shuffle = "--shuffle" in sys.argv
    z = np.load(D / "pre" / "galleries.npz")
    meta = json.loads(str(z["meta"]))
    E = z[emb_name]
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    api = json.load(open(D / "api2025.json"))
    M = {f"{m['tournamentKey']}-{m['id']}": m for m in api["matches"]}
    groups = defaultdict(list)   # (key, side) -> row indices, left to right
    for i, r in enumerate(meta):
        groups[(r["key"], r["side"])].append(i)
    al = []   # (key, side, rows, teams in station order)
    for (key, side), rows in groups.items():
        if len(rows) != 3:
            continue
        st = sorted([p for p in M[key]["participants"]
                     if p["station"] // 10 == (1 if side == "red" else 2) and p["station"] % 10 <= 3],
                    key=lambda p: p["station"])
        teams = [p["country"] for p in st]
        if len(teams) == 3:
            al.append((key, side, rows, teams))
    if shuffle:   # control: same structure, team labels scrambled across alliances
        random.seed(1)
        pool = [t for a in al for t in a[3]]
        random.shuffle(pool)
        al = [(a[0], a[1], a[2], pool[3 * i:3 * i + 3]) for i, a in enumerate(al)]
    S = E @ E.T

    def objective(assign):
        team_rows = defaultdict(list)
        for a, p in zip(al, assign):
            for j, ti in enumerate(p):
                team_rows[a[3][ti]].append(a[2][j])
        tot = 0.0
        for rs in team_rows.values():
            for x, y in itertools.combinations(rs, 2):
                tot += S[x, y]
        return tot, team_rows

    best = (-1e9, None)
    rng = random.Random(0)
    for _restart in range(30):
        assign = [rng.choice(PERMS) for _ in al]
        for _it in range(20):
            changed = False
            _, team_rows = objective(assign)
            for k, a in enumerate(al):
                def gain(p):
                    g = 0.0
                    for j, ti in enumerate(p):
                        others = [r for r in team_rows[a[3][ti]] if r not in a[2]]
                        g += sum(S[a[2][j], r] for r in others)
                    return g
                bp = max(PERMS, key=gain)
                if bp != assign[k]:
                    # update team_rows incrementally
                    for j, ti in enumerate(assign[k]):
                        team_rows[a[3][ti]].remove(a[2][j])
                    for j, ti in enumerate(bp):
                        team_rows[a[3][ti]].append(a[2][j])
                    assign[k] = bp
                    changed = True
            if not changed:
                break
        obj, _ = objective(assign)
        if obj > best[0]:
            best = (obj, list(assign))
    obj, assign = best
    _, team_rows = objective(assign)
    # per-alliance margin: best perm gain minus second best, given everyone else
    margins = []
    for k, a in enumerate(al):
        def gain(p):
            return sum(S[a[2][j], r] for j, ti in enumerate(p)
                       for r in team_rows[a[3][ti]] if r not in a[2])
        g = sorted((gain(p) for p in PERMS), reverse=True)
        margins.append(g[0] - g[1])
    # within-team vs between-team similarity of the final assignment
    within = [S[x, y] for rs in team_rows.values() for x, y in itertools.combinations(rs, 2)]
    allpairs = S[np.triu_indices(len(E), 1)]
    print(f"{emb_name}{' SHUFFLED' if shuffle else ''}: {len(al)} alliances, {len(team_rows)} teams; "
          f"objective {obj:.1f}; mean within-team sim {np.mean(within):.3f} vs all pairs "
          f"{np.mean(allpairs):.3f}; median alliance margin {np.median(margins):.3f}")
    res = {"alliances": [{"key": a[0], "side": a[1], "rows": a[2], "teams": a[3],
                          "spot_team": [a[3][ti] for ti in p], "margin": float(mg)}
                         for a, p, mg in zip(al, assign, margins)]}
    tag = emb_name + ("_shuffled" if shuffle else "")
    (D / "pre" / f"consensus_{tag}.json").write_text(json.dumps(res, indent=1))
    # montage: one row per team (teams with >= 4 appearances), first crop of each assigned row
    crops = z["crops"]
    teams = sorted([t for t, rs in team_rows.items() if len(rs) >= 4])
    S2 = 100
    for k in range(0, len(teams), 18):
        chunk = teams[k:k + 18]
        img = np.full((len(chunk) * (S2 + 4), 70 + 7 * S2, 3), 30, np.uint8)
        for r, t in enumerate(chunk):
            cv2.putText(img, t, (4, r * (S2 + 4) + 55), 0, 0.7, (0, 255, 255), 2)
            for c, row in enumerate(team_rows[t][:7]):
                img[r * (S2 + 4):r * (S2 + 4) + S2, 70 + c * S2:70 + (c + 1) * S2] = \
                    cv2.resize(crops[row][0], (S2, S2))
        cv2.imwrite(str(D / "pre" / f"teams_{tag}_{k // 18}.jpg"), img)


if __name__ == "__main__":
    main()
