from __future__ import annotations

from pathlib import Path

import pytest

from embedkit import engine
from embedkit.adapters.base import AbstractAdapter
from embedkit.engine import Engine


class _WorkingAdapter(AbstractAdapter):
    name = "working"
    tier = "cpu"
    model_format = "gguf"
    closed = 0

    def embed(self, text: str) -> list[float]:
        return [1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        type(self).closed += 1

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        return True, "runtime ready"


class _WarmupFailureAdapter(_WorkingAdapter):
    name = "a-warmup-failure"
    closed = 0

    def warmup(self) -> None:
        raise RuntimeError("warmup failed")


def test_explicit_engine_closes_adapter_after_warmup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.touch()
    _WarmupFailureAdapter.closed = 0
    monkeypatch.setattr(engine, "all_adapters", lambda: [_WarmupFailureAdapter])

    with pytest.raises(RuntimeError, match="warmup failed"):
        Engine(adapter=_WarmupFailureAdapter.name, model=str(model))

    assert _WarmupFailureAdapter.closed == 1


def test_auto_falls_through_after_warmup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.touch()
    _WarmupFailureAdapter.closed = 0
    _WorkingAdapter.closed = 0
    monkeypatch.setattr(engine, "all_adapters", lambda: [_WarmupFailureAdapter, _WorkingAdapter])

    selected = Engine.auto(model=str(model))

    assert selected.info()["adapter"] == "working"
    assert _WarmupFailureAdapter.closed == 1
    selected.close()


def test_list_adapters_is_unavailable_when_default_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.gguf"
    monkeypatch.setattr(engine, "all_adapters", lambda: [_WorkingAdapter])
    monkeypatch.setattr(engine, "resolve_model", lambda adapter, model: str(missing))

    status = Engine.list_adapters()

    assert status[0]["available"] is False
    assert "is not installed" in status[0]["reason"]


def test_shipping_registry_contains_only_implemented_adapters() -> None:
    names = {adapter.name for adapter in engine.all_adapters()}
    assert names == {"cpu-llamacpp", "nvidia-cuda", "cix-npu"}
