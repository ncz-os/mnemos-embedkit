"""auto() picker — capability tier first, measured throughput within tier.

No vendor preference. The kit ranks by measured throughput per host, not
by registry order or any static priority list.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Protocol, TypeVar

log = logging.getLogger("embedkit.pick")


class _NamedCandidate(Protocol):
    @property
    def name(self) -> str: ...


CandidateT = TypeVar("CandidateT", bound=_NamedCandidate)


def pick_fastest_in_tier(
    candidates: Iterable[CandidateT],
    *,
    model: str | None = None,
) -> CandidateT:
    """Return the fastest adapter class for this host within the candidate set.

    For now, returns the first candidate (alphabetical) — the micro-bench
    ranking lands in a follow-up. Adapters are responsible for their own
    `is_available()` gate, so any candidate here is functional.
    """
    cands = list(candidates)
    if not cands:
        raise RuntimeError("pick_fastest_in_tier called with no candidates")
    # TODO(0.2): run a 50-record micro-bench and pick by measured rec/sec.
    chosen = sorted(cands, key=lambda c: c.name)[0]
    log.info("pick_fastest_in_tier: chose %s among %s",
             chosen.name, [c.name for c in cands])
    return chosen
