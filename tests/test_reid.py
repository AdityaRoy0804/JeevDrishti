import numpy as np
from app.services.reid import Hit, Thresholds, aggregate_by_animal, decide
from app.services.vector_store import VectorMeta, VectorStore

T = Thresholds(match=0.80, review=0.65, min_margin=0.05)


def cands(*pairs):
    return aggregate_by_animal([Hit(a, f"R-{i}", s) for i, (a, s) in enumerate(pairs)])


def test_aggregation_takes_max_per_animal():
    c = cands(("T-1", 0.5), ("T-1", 0.9), ("T-2", 0.6))
    assert [x.animal_id for x in c] == ["T-1", "T-2"] and c[0].similarity == 0.9


def test_decisions():
    assert decide(cands(("T-1", 0.93), ("T-2", 0.61)), T).status == "matched"
    assert decide(cands(("T-1", 0.85), ("T-2", 0.83)), T).status == "review"
    assert decide(cands(("T-1", 0.40)), T).status == "unknown"
    assert decide([], T).status == "unknown"


def test_flank_filter():
    vs = VectorStore()
    v = np.array([[1, 0, 0, 0]], dtype="float32")
    vs.add(v, [VectorMeta("1", "T-1", "right")])
    assert vs.search(v[0], flank="left") == []
    assert len(vs.search(v[0], flank="right")) == 1
    assert len(vs.search(v[0])) == 1
