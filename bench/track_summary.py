import json, sys, collections
for p in sys.argv[1:]:
    d = json.load(open(p)); T = collections.defaultdict(list)
    for fr in d["frames"]:
        for r in fr["robots"]: T[r[0]].append((fr["t"], r[1:5]))
    life = {k: (v[0][0], v[-1][0], len(v)) for k, v in T.items()}
    long = {k: v for k, v in life.items() if v[2] >= 20}
    at = lambda t: sorted(k for k, v in T.items() if any(abs(x[0]-t) < 0.15 for x in v))
    print(p.split('/')[-1], "tracks:", len(T), ">=2s:", len(long), "| at t=1:", at(1.0), "| at t=149:", at(149.0))
    for k, (a, b, n) in sorted(long.items(), key=lambda x: x[1][0]):
        c0 = T[k][0][1]; c1 = T[k][-1][1]
        print(f"   id{k:3d} {a:6.1f}-{b:6.1f}s n={n:4d} start@({(c0[0]+c0[2])/2:.0f},{(c0[1]+c0[3])/2:.0f}) end@({(c1[0]+c1[2])/2:.0f},{(c1[1]+c1[3])/2:.0f})")
