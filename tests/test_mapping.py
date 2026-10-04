from __future__ import annotations

from fgc_vision import mapping


def _data(swap: bool) -> dict:
    # two matches; team A always at station 11 with value 0.5 under the true mapping
    def m(i, vals):
        return {"tournamentKey": "t2", "id": i, "played": True,
                "participants": [{"station": 11, "country": "A"}, {"station": 12, "country": "B"},
                                 {"station": 13, "country": "C"}],
                "details": {"redRobotOneParking": vals[0], "redRobotTwoParking": vals[1],
                            "redRobotThreeParking": vals[2]}}
    ms = [m(1, (0.5, 0.125, 0.0)), m(2, (0.5, 0.25, 0.125))]
    ranks = [{"tournamentKey": "t2", "pts": v, "team": {"country": c}}
             for c, v in (("A", 1.0), ("B", 0.375), ("C", 0.125))]
    if swap:
        for r in ranks:
            r["pts"] = {"A": 0.375, "B": 1.0, "C": 0.125}[r["team"]["country"]]
    return {"matches": ms, "rankings": ranks}


def test_identity_mapping_found() -> None:
    best = mapping.check(_data(False))[0]
    assert (best["perm"], best["exact"], best["rankingField"]) == ("123", 3, "pts")


def test_swapped_mapping_found() -> None:
    assert mapping.check(_data(True))[0]["perm"] == "213"
