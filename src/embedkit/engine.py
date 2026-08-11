"""Engine — the user-facing facade.

Quick start:

    eng = embedkit.Engine.auto(model="bge-small-zh-v1.5")
    vec = eng.embed("Hello world")

Explicit adapter pick:

    eng = embedkit.Engine(adapter="cix-npu", model="bge-small-zh-v1.5")
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
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
        model_path = self._existing_model_path(adapter, model)
        self._adapter = self._start_adapter(cls, model_path, kwargs)

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
        failures: list[str] = []
        for tier in ("npu", "gpu", "cpu"):
            in_tier = [a for a in candidates if a.tier == tier]
            while in_tier:
                cls_pick = pick_fastest_in_tier(in_tier, model=model)
                in_tier.remove(cls_pick)
                try:
                    model_path = cls._existing_model_path(cls_pick.name, model)
                    adapter_instance = cls._start_adapter(cls_pick, model_path, kwargs)
                except Exception as exc:
                    failures.append(f"{cls_pick.name}: {exc}")
                    log.warning("auto-selection rejected %s: %s", cls_pick.name, exc)
                    continue
                eng = cls.__new__(cls)
                eng._adapter = adapter_instance
                return eng
        detail = "; ".join(failures) or "no candidates remained"
        raise RuntimeError(f"No adapter could be started: {detail}")

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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except Exception as cleanup_error:
            if exc_value is not None:
                exc_value.add_note(f"Engine cleanup also failed: {cleanup_error}")
            log.exception("engine cleanup failed while handling %r", exc_value)

    @staticmethod
    def _lookup(name: str) -> type[AbstractAdapter] | None:
        for cls in all_adapters():
            if cls.name == name:
                return cls
        return None

    @staticmethod
    def list_adapters(model: str | None = None) -> list[dict[str, Any]]:
        """Return [{name, tier, available, reason}] for every registered adapter."""
        out: list[dict[str, Any]] = []
        for cls in all_adapters():
            ok, why = cls.is_available()
            if ok:
                try:
                    Engine._existing_model_path(cls.name, model)
                except (OSError, ValueError) as exc:
                    ok, why = False, str(exc)
            out.append({"name": cls.name, "tier": cls.tier, "available": ok, "reason": why})
        return out

    @staticmethod
    def _existing_model_path(adapter_name: str, model: str | None) -> str:
        model_path = resolve_model(adapter_name, model)
        path = Path(model_path)
        if not path.is_file():
            requested = "default model" if model is None else f"model {model!r}"
            raise ValueError(
                f"{requested} is not installed for adapter '{adapter_name}': {model_path}. "
                "Pass a path to a provisioned model."
            )
        adapter_cls = Engine._lookup(adapter_name)
        if adapter_cls is None:
            raise ValueError(f"Unknown adapter '{adapter_name}'")
        actual_format = path.suffix.removeprefix(".").lower()
        if actual_format != adapter_cls.model_format.lower():
            raise ValueError(
                f"Model {model_path!r} has format '{actual_format or 'unknown'}', but adapter "
                f"'{adapter_name}' requires '{adapter_cls.model_format}'."
            )
        return model_path

    @staticmethod
    def _start_adapter(
        adapter_cls: type[AbstractAdapter],
        model_path: str,
        kwargs: dict[str, Any],
    ) -> AbstractAdapter:
        adapter = adapter_cls(model_path=model_path, **kwargs)
        try:
            adapter.warmup()
        except BaseException:
            try:
                adapter.close()
            except Exception:
                log.exception("adapter cleanup failed after warmup failure")
            raise
        return adapter
