from __future__ import annotations

from fgc_vision import identity as I


def _tr(i: int, t0: float, t1: float, x0: float, x1: float, y: float = 500.0,
        y1: float | None = None) -> I.Tracklet:
    n = int((t1 - t0) * 10) + 1
    tr = I.Tracklet(i)
    for k in range(n):
        f = k / max(n - 1, 1)
        x = x0 + (x1 - x0) * f
        yy = y + ((y1 if y1 is not None else y) - y) * f
        tr.t.append(round(t0 + k * 0.1, 2))
        tr.box.append([x - 40, yy - 40, x + 40, yy + 40])
    return tr


def test_inside_field_poly() -> None:
    assert I.inside(I.FIELD_POLY_2025, 960, 500)
    assert not I.inside(I.FIELD_POLY_2025, 1755, 319)   # arena wall sign
    assert not I.inside(I.FIELD_POLY_2025, 817, 174)    # tower top


def test_static_furniture_dropped() -> None:
    assert I.static(_tr(1, 0, 30, 900, 902))
    assert not I.static(_tr(2, 0, 30, 400, 900))


def test_stitch_links_through_gap_and_keeps_sides() -> None:
    a = _tr(1, 0, 10, 300, 400)          # red half
    b = _tr(2, 0, 10, 1500, 1400)        # blue half
    a2 = _tr(3, 12, 20, 430, 500)        # reappears 2 s later near a's end
    chains = I.stitch([a, b, a2], n_robots=2)
    by_side = {c.side: [p.id for p in c.parts] for c in chains}
    assert by_side == {"red": [1, 3], "blue": [2]}


def test_climb_signature() -> None:
    up = _tr(1, 120, 150, 800, 805, y=500, y1=380)     # rises 120 px, no sideways motion
    flat = _tr(2, 120, 150, 600, 900, y=500)
    assert I.climb(I.Chain(0, "red", [up]))["climbed"]
    assert not I.climb(I.Chain(1, "red", [flat]))["climbed"]


def test_official_end_and_stations() -> None:
    m = {"participants": [{"station": 11, "country": "GRN"}, {"station": 23, "country": "MGL"}],
         "details": {"redRobotOneParking": 0, "blueRobotThreeParking": 0.5}}
    assert I.official_end(m) == {"red": {1: 0.0}, "blue": {3: 0.5}}
    assert I.stations(m) == {"red": {1: "GRN"}, "blue": {3: "MGL"}}
