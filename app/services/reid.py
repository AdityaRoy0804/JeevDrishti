from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["matched", "review", "unknown"]


@dataclass(frozen=True)
class Thresholds:
    match: float
    review: float
    min_margin: float


@dataclass(frozen=True)
class Hit:
    animal_id: str
    reference_id: str
    similarity: float


@dataclass(frozen=True)
class Candidate:
    animal_id: str
    similarity: float
    best_reference_id: str


@dataclass(frozen=True)
class Decision:
    status: Status
    similarity: float
    margin: float
    top_candidates: list[Candidate]


def aggregate_by_animal(hits: list[Hit], top_n: int = 5) -> list[Candidate]:
    """Max similarity per animal, ranked descending."""
    best: dict[str, Hit] = {}
    for h in hits:
        if h.animal_id not in best or h.similarity > best[h.animal_id].similarity:
            best[h.animal_id] = h
    ranked = sorted(best.values(), key=lambda h: h.similarity, reverse=True)
    return [Candidate(h.animal_id, h.similarity, h.reference_id) for h in ranked[:top_n]]


def decide(candidates: list[Candidate], t: Thresholds) -> Decision:
    if not candidates:
        return Decision("unknown", 0.0, 0.0, [])
    top1 = candidates[0].similarity
    top2 = candidates[1].similarity if len(candidates) > 1 else 0.0  # no runner-up
    margin = top1 - top2
    if top1 >= t.match and margin >= t.min_margin:
        status: Status = "matched"
    elif top1 >= t.review:
        status = "review"
    else:
        status = "unknown"
    return Decision(status, top1, margin, candidates)
