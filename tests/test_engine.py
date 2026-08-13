from __future__ import annotations

import subprocess
import sys
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


def test_auto_does_not_probe_lower_tiers_after_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.touch()

    class NPUAdapter(_WorkingAdapter):
        name = "npu"
        tier = "npu"

    class LowerAdapter(_WorkingAdapter):
        name = "lower"

        @classmethod
        def is_available(cls) -> tuple[bool, str]:
            raise AssertionError("lower tier should not be probed")

    monkeypatch.setattr(engine, "all_adapters", lambda: [NPUAdapter, LowerAdapter])

    selected = Engine.auto(model=str(model))

    assert selected.info()["adapter"] == "npu"
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


def test_list_adapters_checks_the_requested_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class CPUAdapter(_WorkingAdapter):
        name = "cpu-llamacpp"

    class CUDAAdapter(_WorkingAdapter):
        name = "nvidia-cuda"
        model_format = "onnx"

    class CixAdapter(_WorkingAdapter):
        name = "cix-npu"
        model_format = "cix"

    adapters = [CPUAdapter, CUDAAdapter, CixAdapter]
    monkeypatch.setattr(engine, "all_adapters", lambda: adapters)
    for suffix, compatible_adapter in (
        ("gguf", "cpu-llamacpp"),
        ("onnx", "nvidia-cuda"),
        ("cix", "cix-npu"),
    ):
        model = tmp_path / f"custom.{suffix}"
        model.touch()

        status = Engine.list_adapters(model=str(model))

        assert [item["name"] for item in status if item["available"]] == [compatible_adapter]


class _CleanupFailureAdapter(_WorkingAdapter):
    name = "cleanup-failure"

    def close(self) -> None:
        raise RuntimeError("cleanup failed")


def test_context_preserves_body_exception_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.touch()
    monkeypatch.setattr(engine, "all_adapters", lambda: [_CleanupFailureAdapter])

    with pytest.raises(ValueError, match="inference failed") as exc_info:
        with Engine(adapter=_CleanupFailureAdapter.name, model=str(model)):
            raise ValueError("inference failed")

    assert exc_info.value.__notes__ == ["Engine cleanup also failed: cleanup failed"]


def test_context_propagates_cleanup_failure_without_body_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.touch()
    monkeypatch.setattr(engine, "all_adapters", lambda: [_CleanupFailureAdapter])

    with pytest.raises(RuntimeError, match="cleanup failed"):
        with Engine(adapter=_CleanupFailureAdapter.name, model=str(model)):
            pass


def test_shipping_registry_contains_only_implemented_adapters() -> None:
    names = {adapter.name for adapter in engine.all_adapters()}
    assert names == {"cpu-llamacpp", "nvidia-cuda", "cix-npu"}


def test_import_does_not_load_numpy_or_adapter_implementations() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, embedkit; "
                "print('numpy' in sys.modules); "
                "print(sorted(m for m in sys.modules "
                "if m.startswith('embedkit.adapters.')))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["False", "['embedkit.adapters.base']"]
