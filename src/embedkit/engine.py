"""Engine — the user-facing facade.

Quick start:

    eng = embedkit.Engine.auto()        # picks fastest adapter on this host
    vec = eng.embed("Hello world")

Explicit adapter pick:

    eng = embedkit.Engine(adapter="cix-npu", model="bge-small-zh-v1.5")
"""
from __future__ import annotations

import logging
from typing import Any

from .adapters import AbstractAdapter, all_adapters
from .models import resolve_model

log = logging.getLogger("embedkit.engine")


class Engine:
    """User-facing wrapper. Holds one bound adapter for the lifetime of the call.

    The kit is silicon-agnostic. Engine.auto() picks by capability tier
    (NPU > GPU > CPU) and measured throughput within tier, NOT by vendor.
    """

    def __init__(self, adapter: str | None = None, model: str | None = None, **kwargs: Any) -> None:
        if adapter is None:
            raise ValueError(
                "Engine() requires either adapter='...' or use Engine.auto()"
            )
        cls = self._lookup(adapter)
        if cls is None:
            raise ValueError(
                f"Unknown adapter '{adapter}'. Use embedkit.Engine.auto() to list options."
            )
        ok, why = cls.is_available()
        if not ok:
            raise RuntimeError(f"Adapter '{adapter}' is not available on this host: {why}")
        model_path = resolve_model(adapter, model)
        self._adapter: AbstractAdapter = cls(model_path=model_path, **kwargs)
        self._adapter.warmup()

    @classmethod
    def auto(
        cls,
        prefer_tier: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Engine:
        """Pick the fastest available adapter on this host.

        Selection policy:
          1. Filter to adapter classes whose is_available() returns ok=True.
          2. Group by capability tier (NPU > GPU > CPU).
          3. Within the highest-populated tier, run a 50-record micro-bench
             on the host and pick the fastest. Cache per (host, model).

        prefer_tier="cpu" forces CPU even if NPU/GPU exist (useful for
        thermal-budget tests).
        """
        from .pick import pick_fastest_in_tier  # local import to break cycle

        candidates = [a for a in all_adapters() if a.is_available()[0]]
        if not candidates:
            raise RuntimeError(
                "No embed adapter available on this host. "
                "Install at minimum: pip install mnemos-embedkit[cpu-llamacpp]"
            )
        if prefer_tier:
            candidates = [a for a in candidates if a.tier == prefer_tier]
            if not candidates:
                raise RuntimeError(f"No adapter available in tier '{prefer_tier}'.")
        for tier in ("npu", "gpu", "cpu"):
            in_tier = [a for a in candidates if a.tier == tier]
            if in_tier:
                cls_pick = pick_fastest_in_tier(in_tier, model=model)
                eng = cls.__new__(cls)
                model_path = resolve_model(cls_pick.name, model)
                eng._adapter = cls_pick(model_path=model_path, **kwargs)
                eng._adapter.warmup()
                return eng
        raise RuntimeError("No adapter available after tier filter.")

    def embed(self, text: str) -> list[float]:
        return self._adapter.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._adapter.embed_batch(texts)

    def info(self) -> dict[str, Any]:
        return dict(self._adapter.info())

    def close(self) -> None:
        self._adapter.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def _lookup(name: str) -> type[AbstractAdapter] | None:
        for cls in all_adapters():
            if cls.name == name:
                return cls
        return None

    @staticmethod
    def list_adapters() -> list[dict[str, Any]]:
        """Return [{name, tier, available, reason}] for every registered adapter."""
        out: list[dict[str, Any]] = []
        for cls in all_adapters():
            ok, why = cls.is_available()
            out.append({"name": cls.name, "tier": cls.tier, "available": ok, "reason": why})
        return out
