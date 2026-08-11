# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jason Perlow
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from embedkit.adapters import cpu_llamacpp
from embedkit.adapters.cpu_llamacpp import CPULlamaCppAdapter

AVAILABLE = CPULlamaCppAdapter.is_available()[0]
MODEL_ENV = "EMBEDKIT_TEST_CPU_LLAMACPP_MODEL"


def _model_path() -> str:
    value = os.environ.get(MODEL_ENV)
    if not value:
        pytest.skip(f"set {MODEL_ENV} to run cpu-llamacpp adapter smoke tests")
    if not Path(value).exists():
        pytest.skip(f"{MODEL_ENV} path does not exist: {value}")
    return value


def test_is_available_safe() -> None:
    started = time.perf_counter()
    for _ in range(1000):
        ok, reason = CPULlamaCppAdapter.is_available()
        assert isinstance(ok, bool)
        assert isinstance(reason, str)
        assert reason
    assert time.perf_counter() - started < 10.0


def test_embedding_preserves_native_magnitude_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            pass

        def create_embedding(self, text: str) -> dict[str, object]:
            return {"data": [{"embedding": [3.0, 4.0]}]}

    monkeypatch.setattr(
        cpu_llamacpp.importlib,
        "import_module",
        lambda name: SimpleNamespace(Llama=FakeLlama),
    )

    adapter = CPULlamaCppAdapter("model.gguf")
    assert adapter.embed("hello") == [3.0, 4.0]


def test_embedding_normalization_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            pass

        def create_embedding(self, text: str) -> dict[str, object]:
            return {"data": [{"embedding": [3.0, 4.0]}]}

    monkeypatch.setattr(
        cpu_llamacpp.importlib,
        "import_module",
        lambda name: SimpleNamespace(Llama=FakeLlama),
    )

    adapter = CPULlamaCppAdapter("model.gguf", normalize=True)
    assert adapter.embed("hello") == pytest.approx([0.6, 0.8])


@pytest.mark.skipif(not AVAILABLE, reason="adapter unavailable")
def test_embed_dim_consistent() -> None:
    adapter = CPULlamaCppAdapter(
        _model_path(),
        n_ctx=int(os.environ.get("EMBEDKIT_TEST_N_CTX", "2048")),
    )
    try:
        short = adapter.embed("hello")
        long = adapter.embed("this is a longer embedding input " * 64)
        assert len(short) == adapter.embed_dim
        assert len(long) == adapter.embed_dim
    finally:
        adapter.close()


@pytest.mark.skipif(not AVAILABLE, reason="adapter unavailable")
def test_close_idempotent() -> None:
    adapter = CPULlamaCppAdapter(
        _model_path(),
        n_ctx=int(os.environ.get("EMBEDKIT_TEST_N_CTX", "2048")),
    )
    adapter.close()
    adapter.close()
