"""Checks that a drawing is one a record can be read back from.

A generator that emits an outline nothing checked is a generator that can emit a picture two
different records both explain, and a model held against such a picture is asked to guess. Every
constraint here is a way the image could stop determining its record:

* the part must be one closed area, so its edges form a single cycle;
* consecutive edges must turn, and turn enough to see, or two edges render as one straight
  stroke and where one ends is not visible;
* edges that do not share a corner must stay apart, or two strokes read as one;
* nothing may cross or touch anything it does not share a corner with, part or annotation alike.

The checks are geometric rather than a proxy for one, and a sample that fails any of them is thrown
away rather than repaired — a repair is a change to the record the picture was drawn from.
"""

import math
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

from generator import (
    AnnotationText,
    AnnotationTextRefId,
    BothSidedArrow,
    ConnectTwoElementsWithId,
    Element,
    HelpLine,
    PartLine,
    PartLineWithId,
)

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


class Invalid(Exception):
    """Raised with the constraint a drawing broke."""


# -- small geometry --------------------------------------------------------------------------


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _same_point(p: Point, q: Point, tol: float) -> bool:
    return _close(p[0], q[0], tol) and _close(p[1], q[1], tol)


def _length(s: Segment) -> float:
    (x1, y1), (x2, y2) = s
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _point_segment_distance(p: Point, s: Segment) -> float:
    (x1, y1), (x2, y2) = s
    dx, dy = x2 - x1, y2 - y1
    span = dx * dx + dy * dy
    if span == 0.0:
        return ((p[0] - x1) ** 2 + (p[1] - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / span))
    nx, ny = x1 + t * dx, y1 + t * dy
    return ((p[0] - nx) ** 2 + (p[1] - ny) ** 2) ** 0.5


def segment_distance(a: Segment, b: Segment) -> float:
    """How far two segments stay apart; zero when they touch or cross."""
    if _segments_cross(a, b):
        return 0.0
    return min(
        _point_segment_distance(a[0], b),
        _point_segment_distance(a[1], b),
        _point_segment_distance(b[0], a),
        _point_segment_distance(b[1], a),
    )


def _orientation(p: Point, q: Point, r: Point) -> float:
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _segments_cross(a: Segment, b: Segment) -> bool:
    d1, d2 = _orientation(b[0], b[1], a[0]), _orientation(b[0], b[1], a[1])
    d3, d4 = _orientation(a[0], a[1], b[0]), _orientation(a[0], a[1], b[1])
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False


# -- the checks ------------------------------------------------------------------------------


def outline_of(actions: Sequence[Element]) -> Tuple[List[Tuple[str, Segment]], List[Tuple[str, str]]]:
    """The part's edges by id, and the pairs the record says are joined."""
    edges = [
        (a.id, (a.part_line.start_point, a.part_line.end_point))
        for a in actions
        if isinstance(a, PartLineWithId)
    ]
    joins = [(a.id1, a.id2) for a in actions if isinstance(a, ConnectTwoElementsWithId)]
    return edges, joins


def check_one_closed_area(edges, joins, tol: float) -> None:
    """One connected area: every edge meets exactly two others, in a single cycle."""
    if len(edges) < 4:
        raise Invalid(f"an outline needs at least four edges, found {len(edges)}")
    degree = defaultdict(int)
    for a, b in joins:
        degree[a] += 1
        degree[b] += 1
    ids = {i for i, _ in edges}
    for i in ids:
        if degree[i] != 2:
            raise Invalid(f"edge {i} joins {degree[i]} others, so the outline is not one closed area")

    # one cycle rather than several
    neighbours = defaultdict(list)
    for a, b in joins:
        neighbours[a].append(b)
        neighbours[b].append(a)
    start = next(iter(ids))
    seen, stack = set(), [start]
    while stack:
        here = stack.pop()
        if here in seen:
            continue
        seen.add(here)
        stack.extend(neighbours[here])
    if seen != ids:
        raise Invalid(f"the outline falls into more than one loop ({len(seen)} of {len(ids)} edges reachable)")

    # every joined pair really shares a corner
    by_id = dict(edges)
    for a, b in joins:
        pa, pb = by_id[a], by_id[b]
        if not any(_same_point(p, q, tol) for p in pa for q in pb):
            raise Invalid(f"edges {a} and {b} are joined but share no corner")


def check_edges_turn(edges, joins, tol: float, min_turn: float = 0.0) -> None:
    """Consecutive edges must turn, by at least `min_turn` degrees, or the two render as one
    straight stroke."""
    by_id = dict(edges)
    least_sine = math.sin(math.radians(min_turn))
    for a, b in joins:
        (a1, a2), (b1, b2) = by_id[a], by_id[b]
        da = (a2[0] - a1[0], a2[1] - a1[1])
        db = (b2[0] - b1[0], b2[1] - b1[1])
        cross = da[0] * db[1] - da[1] * db[0]
        if abs(cross) <= tol:
            raise Invalid(f"edges {a} and {b} are collinear, so where one ends is not visible")
        sine = abs(cross) / (math.hypot(*da) * math.hypot(*db))
        if sine < least_sine - 1e-9:
            turn = math.degrees(math.asin(min(1.0, sine)))
            raise Invalid(f"edges {a} and {b} turn by only {turn:.1f} degrees, under the {min_turn} minimum")


def check_minimum_edge_length(edges, min_length: float) -> None:
    for i, segment in edges:
        if _length(segment) < min_length - 1e-9:
            raise Invalid(f"edge {i} is {_length(segment):.4f} long, under the {min_length} minimum")


def check_minimum_gap(edges, joins, min_gap: float) -> None:
    """Edges that share no corner must stay apart, or two strokes read as one."""
    joined = {frozenset(pair) for pair in joins}
    for idx, (i, a) in enumerate(edges):
        for j, b in edges[idx + 1:]:
            if frozenset((i, j)) in joined:
                continue
            gap = segment_distance(a, b)
            if gap < min_gap - 1e-9:
                raise Invalid(f"edges {i} and {j} come within {gap:.4f}, under the {min_gap} minimum")


def _drawn_segments(actions: Sequence[Element]) -> List[Tuple[str, Segment]]:
    """Every stroke that reaches the picture, part and annotation alike."""
    drawn: List[Tuple[str, Segment]] = []
    for n, a in enumerate(actions):
        if isinstance(a, PartLineWithId):
            drawn.append((f"part:{a.id}", (a.part_line.start_point, a.part_line.end_point)))
        elif isinstance(a, PartLine):
            drawn.append((f"part:{n}", (a.start_point, a.end_point)))
        elif isinstance(a, HelpLine) and not getattr(a, "formative_only", False):
            drawn.append((f"help:{n}", (a.start_point, a.end_point)))
        elif isinstance(a, BothSidedArrow) and not getattr(a, "formative_only", False):
            drawn.append((f"arrow:{n}", (a.start_point, a.end_point)))
    return drawn


def _without_start(s: Segment, trim: float) -> Segment:
    """The segment with its first `trim` trimmed off, so touching at its start does not count."""
    (x1, y1), (x2, y2) = s
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length <= trim:
        return ((x2, y2), (x2, y2))
    t = trim / length
    return ((x1 + dx * t, y1 + dy * t), (x2, y2))


def check_nothing_overlaps(actions: Sequence[Element], joins, min_gap: float) -> None:
    """No stroke may cross or touch another it does not share a corner with.

    A dimension's help line is allowed to begin on the edge it measures — that is what attaches it
    — so it is checked from just past its start. Everywhere else it has to keep its distance, from
    that edge and from every other.
    """
    joined = {frozenset((f"part:{a}", f"part:{b}")) for a, b in joins}
    drawn = _drawn_segments(actions)
    for idx, (name, a) in enumerate(drawn):
        for other, b in drawn[idx + 1:]:
            if frozenset((name, other)) in joined:
                continue
            left, right = a, b
            if name.startswith("help:") and other.startswith("part:"):
                left = _without_start(a, min_gap)
            elif other.startswith("help:") and name.startswith("part:"):
                right = _without_start(b, min_gap)
            if segment_distance(left, right) < min_gap - 1e-9:
                raise Invalid(f"{name} and {other} overlap or come too close to tell apart")


def check(
    actions: Sequence[Element],
    min_edge_length: float,
    min_gap: float,
    tol: float = 1e-6,
    min_turn: float = 0.0,
) -> None:
    """Raises [[Invalid]] with the first constraint the drawing breaks."""
    edges, joins = outline_of(actions)
    check_one_closed_area(edges, joins, tol)
    check_edges_turn(edges, joins, tol, min_turn)
    check_minimum_edge_length(edges, min_edge_length)
    check_minimum_gap(edges, joins, min_gap)
    check_nothing_overlaps(actions, joins, min_gap)


def why_invalid(
    actions: Sequence[Element],
    min_edge_length: float,
    min_gap: float,
    min_turn: float = 0.0,
) -> Optional[str]:
    """The constraint the drawing breaks, or `None` when it breaks none."""
    try:
        check(actions, min_edge_length, min_gap, min_turn=min_turn)
        return None
    except Invalid as invalid:
        return str(invalid)
