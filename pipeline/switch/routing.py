"""Which banked frame a prompt switch recalls. Plain Python, so the rule can be checked without
a model: the frame bank computes the similarities and hands them over.
"""
from typing import List, Optional, Sequence

ROUTING_MODES = ("key", "truth")
TIE_RULES = ("recent", "first")
TIE_MARGIN = 1e-6          # similarities this close to the best one count as tied


def pick_banked(candidates: Sequence[dict], sims: Sequence[float], *, mode: str, subject: int,
                tie: str, threshold: float) -> Optional[dict]:
    """The entry to recall among `candidates`, oldest first, or None.

    key    the entries whose similarity is within TIE_MARGIN of the best one, if that best
           reaches `threshold`; -inf as threshold takes the best whatever its score
    truth  the entries that show `subject`, the ceiling of the same bank
    `tie` then keeps the most recent of them, or the first."""
    if mode == "truth":
        pool: List[dict] = [e for e in candidates if e["subject"] == subject]
    else:
        best = max(sims) if candidates else float("-inf")
        pool = [e for e, s in zip(candidates, sims) if s >= best - TIE_MARGIN] if best >= threshold else []
    if not pool:
        return None
    return pool[-1] if tie == "recent" else pool[0]
