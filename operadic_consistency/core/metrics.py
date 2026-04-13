#core/metrics.py

from collections import Counter
from math import log
from typing import Any, Dict, Mapping, Sequence, Tuple, List, Optional

from operadic_consistency.core.consistency import ConsistencyReport
from operadic_consistency.core.transforms import CollapsePlan

def _answer_key(text: str) -> str:
    # Default fallback key if no normalizer is used; keep simple.
    return text

def answer_distribution(
    report: ConsistencyReport,
    *,
    use_normalized: bool = True,
) -> Mapping[str, int]:
    """
    Histogram of root answers across runs.
    If use_normalized=True and normalized_root is present, use that; otherwise use raw text.
    """
    keys: List[str] = []
    for r in report.runs:
        if use_normalized and r.normalized_root is not None:
            keys.append(r.normalized_root)
        else:
            keys.append(_answer_key(r.root_answer.text))
    return dict(Counter(keys))

def mode_answer(
    report: ConsistencyReport,
    *,
    use_normalized: bool = True,
) -> Optional[Tuple[str, int]]:
    """Return (most_common_answer, count), or None if no runs."""
    dist = answer_distribution(report, use_normalized=use_normalized)
    if not dist:
        return None
    k = max(dist, key=lambda x: dist[x])
    return (k, dist[k])

def agreement_rate(
    report: ConsistencyReport,
    *,
    use_normalized: bool = True,
) -> float:
    """
    Fraction of runs that equal the modal answer.
    Returns 0.0 if there are no runs.
    """
    dist = answer_distribution(report, use_normalized=use_normalized)
    n = sum(dist.values())
    if n == 0:
        return 0.0
    m = max(dist.values())
    return m / n

def shannon_entropy(dist: Mapping[str, int]) -> float:
    """
    Shannon entropy (natural log) of a discrete distribution given as counts.
    """
    total = sum(dist.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in dist.values():
        p = c / total
        ent -= p * log(p)
    return ent

def inconsistency_witnesses(
    report: ConsistencyReport,
    *,
    use_normalized: bool = True,
    max_per_answer: int = 3,
) -> Mapping[str, Sequence[CollapsePlan]]:
    """
    For each distinct answer, return up to max_per_answer CollapsePlans that produced it.
    """
    out: Dict[str, List[CollapsePlan]] = {}
    for r in report.runs:
        if use_normalized and r.normalized_root is not None:
            k = r.normalized_root
        else:
            k = _answer_key(r.root_answer.text)

        if k not in out:
            out[k] = []
        if len(out[k]) < max_per_answer:
            out[k].append(r.plan)
    return out

def summarize_report(
    report: ConsistencyReport,
    *,
    use_normalized: bool = True,
    top_k: int = 5,
    max_witnesses_per_answer: int = 2,
) -> Mapping[str, Any]:
    """
    Produce a compact summary dict:
      - num_runs, num_unique_answers
      - mode answer and fraction
      - entropy
      - top_k answers
      - witness plans
    """
    dist = answer_distribution(report, use_normalized=use_normalized)
    total = sum(dist.values())

    top = Counter(dist).most_common(top_k)
    mode = top[0] if top else None
    mode_frac = (mode[1] / total) if mode and total > 0 else None

    witnesses = inconsistency_witnesses(
        report,
        use_normalized=use_normalized,
        max_per_answer=max_witnesses_per_answer,
    )

    return {
        "num_runs": total,
        "num_unique_answers": len(dist),
        "mode_answer": mode[0] if mode else None,
        "mode_count": mode[1] if mode else None,
        "mode_fraction": mode_frac,
        "entropy": shannon_entropy(dist),
        "top_answers": top,
        "witness_plans": {k: [p.cut_edges for p in ps] for k, ps in witnesses.items()},
    }
