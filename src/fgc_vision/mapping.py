"""Check which station each per-robot field in the official match details belongs to.

The details name per-robot fields like `redRobotOneParking`; nothing says that "One" is
station x1. The rankings, however, carry per-team season totals. If some ranking field
equals the per-team sum of a per-robot field under exactly one robot->station permutation,
that permutation is the mapping. 2025: `protectionPoints` == sum of `*Robot<k>Parking` under
One/Two/Three = x1/x2/x3 for 181/181 teams; the best other permutation matches 11/181.

Field names change every season, so robot fields and ranking fields are found generically.
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from typing import Any

ROBOT_FIELD = re.compile(r"^(red|blue)Robot(One|Two|Three)(.+)$")
IDX = {"One": 1, "Two": 2, "Three": 3}


def robot_suffixes(matches: list[dict[str, Any]]) -> list[str]:
    out: set[str] = set()
    for m in matches:
        for k in (m.get("details") or {}):
            g = ROBOT_FIELD.match(k)
            if g:
                out.add(g.group(3))
    return sorted(out)


def check(data: dict[str, Any], tournament: str = "t2", tol: float = 1e-6) -> list[dict[str, Any]]:
    matches = [m for m in data["matches"] if m.get("tournamentKey") == tournament and m.get("played")]
    ranks = {r["team"]["country"]: r for r in data.get("rankings", [])
             if r.get("tournamentKey") in (tournament, None)}
    rank_fields = sorted({k for r in ranks.values() for k, v in r.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)})
    results = []
    for suf in robot_suffixes(matches):
        for perm in itertools.permutations((1, 2, 3)):
            tot: dict[str, float] = defaultdict(float)
            for m in matches:
                d = m.get("details") or {}
                for p in m["participants"]:
                    side = "red" if p["station"] // 10 == 1 else "blue"
                    k = p["station"] % 10
                    if k > 3:
                        continue
                    word = {v: w for w, v in IDX.items()}[perm[k - 1]]
                    v = d.get(f"{side}Robot{word}{suf}")
                    if isinstance(v, (int, float)):
                        tot[p["country"]] += float(v)
            for rf in rank_fields:
                both = [c for c in tot if c in ranks]
                exact = sum(abs(tot[c] - float(ranks[c][rf])) <= tol for c in both)
                if both and exact:
                    results.append({"robotField": suf, "rankingField": rf,
                                    "perm": "".join(map(str, perm)),
                                    "exact": exact, "teams": len(both)})
    results.sort(key=lambda r: -r["exact"])
    return results
