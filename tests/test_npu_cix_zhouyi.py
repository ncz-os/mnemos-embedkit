# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jason Perlow
from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embedkit.adapters import npu_cix_zhouyi
from embedkit.adapters.npu_cix_zhouyi import CixZhouyiAdapter

AVAILABLE = CixZhouyiAdapter.is_available()[0]
MODEL_ENV = "EMBEDKIT_TEST_CIX_NPU_MODEL"
TOKENIZER_ENV = "EMBEDKIT_TEST_CIX_NPU_TOKENIZER"


def _model_path() -> str:
    value = os.environ.get(MODEL_ENV)
    if not value:
        pytest.skip(f"set {MODEL_ENV} to run cix-npu adapter smoke tests")
    if not Path(value).exists():
        pytest.skip(f"{MODEL_ENV} path does not exist: {value}")
    return value


def _tokenizer_path() -> str | None:
    value = os.environ.get(TOKENIZER_ENV)
    if value and not Path(value).exists():
        pytest.skip(f"{TOKENIZER_ENV} path does not exist: {value}")
    return value


def test_is_available_safe() -> None:
    started = time.perf_counter()
    for _ in range(1000):
        ok, reason = CixZhouyiAdapter.is_available()
        assert isinstance(ok, bool)
        assert isinstance(reason, str)
        assert reason
    assert time.perf_counter() - started < 10.0


def test_configured_missing_library_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDKIT_CIX_LIBNOE", "/definitely/missing/libnoe.so")
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: str(path) == "/dev/aipu" or original_exists(path),
    )

    ok, reason = CixZhouyiAdapter.is_available()

    assert ok is False
    assert reason == "configured libnoe path is not a file: /definitely/missing/libnoe.so"


def test_configured_unloadable_library_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    library = tmp_path / "libnoe.so"
    library.touch()
    monkeypatch.setenv("EMBEDKIT_CIX_LIBNOE", str(library))
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: str(path) == "/dev/aipu" or original_exists(path),
    )
    monkeypatch.setattr(
        npu_cix_zhouyi.ctypes,
        "CDLL",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("invalid ELF header")),
    )

    ok, reason = CixZhouyiAdapter.is_available()

    assert ok is False
    assert "invalid ELF header" in reason


def test_configured_loadable_library_is_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    library = tmp_path / "libnoe.so"
    library.touch()
    monkeypatch.setenv("EMBEDKIT_CIX_LIBNOE", str(library))
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: str(path) == "/dev/aipu" or original_exists(path),
    )
    monkeypatch.setattr(npu_cix_zhouyi.ctypes, "CDLL", lambda *args, **kwargs: object())

    ok, reason = CixZhouyiAdapter.is_available()

    assert ok is True
    assert str(library) in reason


class _FakeFunction:
    def __init__(self, result: int = 0) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, *args: object) -> int:
        self.calls += 1
        return self.result


class _FakeTensorDescriptorFunction(_FakeFunction):
    def __call__(self, *args: object) -> int:
        self.calls += 1
        if self.result != 0:
            return self.result
        tensor_type = int(args[2])
        tensor_index = int(args[3])
        desc = args[4]._obj  # type: ignore[attr-defined]
        if tensor_type == npu_cix_zhouyi.NOE_TENSOR_TYPE_INPUT:
            desc.size = 256 * np.dtype(np.int32).itemsize
            desc.data_type = 6
        elif tensor_index == 1:
            desc.size = 2 * np.dtype(np.float32).itemsize
            desc.scale = 1.0
            desc.data_type = 7
        return 0


class _FakeLib:
    def __init__(self) -> None:
        for name in (
            "noe_init_context",
            "noe_deinit_context",
            "noe_load_graph",
            "noe_unload_graph",
            "noe_create_job",
            "noe_clean_job",
            "noe_load_tensor",
            "noe_get_tensor",
            "noe_job_infer_sync",
            "noe_get_tensor_descriptor",
        ):
            setattr(self, name, _FakeFunction())
        self.noe_get_tensor_descriptor = _FakeTensorDescriptorFunction()


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, lib: _FakeLib) -> None:
    tokenizer = SimpleNamespace(from_pretrained=lambda path: object())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    monkeypatch.setattr(npu_cix_zhouyi.ctypes, "CDLL", lambda *args, **kwargs: lib)


def test_constructor_cleans_graph_and_context_after_descriptor_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = _FakeLib()
    lib.noe_get_tensor_descriptor.result = 5
    _install_fake_runtime(monkeypatch, lib)

    with pytest.raises(RuntimeError, match="get_pooled_desc"):
        CixZhouyiAdapter("model.cix", lib_path="libnoe.so")

    assert lib.noe_unload_graph.calls == 1
    assert lib.noe_deinit_context.calls == 1


def test_constructor_deinitializes_context_after_graph_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = _FakeLib()
    lib.noe_load_graph.result = 5
    _install_fake_runtime(monkeypatch, lib)

    with pytest.raises(RuntimeError, match="load_graph"):
        CixZhouyiAdapter("model.cix", lib_path="libnoe.so")

    assert lib.noe_unload_graph.calls == 0
    assert lib.noe_deinit_context.calls == 1


def test_constructor_rejects_token_length_that_mismatches_native_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = _FakeLib()
    _install_fake_runtime(monkeypatch, lib)

    with pytest.raises(ValueError, match="input tensor 0 requires 1024 bytes"):
        CixZhouyiAdapter("model.cix", max_tokens=128, lib_path="libnoe.so")

    assert lib.noe_unload_graph.calls == 1
    assert lib.noe_deinit_context.calls == 1


def test_constructor_accepts_default_length_matching_native_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lib = _FakeLib()
    _install_fake_runtime(monkeypatch, lib)

    adapter = CixZhouyiAdapter("model.cix", lib_path="libnoe.so")

    assert adapter.max_tokens == 256
    assert adapter._input_sizes == (1024, 1024, 1024)
    assert lib.noe_get_tensor_descriptor.calls == 4
    adapter.close()


def _adapter_for_job_test(lib: _FakeLib) -> CixZhouyiAdapter:
    adapter = CixZhouyiAdapter.__new__(CixZhouyiAdapter)
    adapter._closed = False
    adapter._lib = lib
    adapter._ctx = None
    adapter._graph_id = ctypes.c_uint64(1)
    adapter._timeout_ms = 10
    adapter._normalize = False
    adapter._pooled_dtype = np.float32
    adapter._zero_point = 0
    adapter._scale = 1.0
    adapter.embed_dim = 2
    adapter._input_sizes = (4, 4, 4)
    adapter._tokenize = lambda text: (
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
    )
    return adapter


def test_job_cleanup_failure_is_raised() -> None:
    lib = _FakeLib()
    lib.noe_clean_job.result = 9
    adapter = _adapter_for_job_test(lib)

    with pytest.raises(RuntimeError, match="clean_job returned 0x9"):
        adapter.embed("hello")


def test_job_cleanup_failure_preserves_inference_error(caplog: pytest.LogCaptureFixture) -> None:
    lib = _FakeLib()
    lib.noe_job_infer_sync.result = 3
    lib.noe_clean_job.result = 9
    adapter = _adapter_for_job_test(lib)

    with pytest.raises(RuntimeError, match="infer returned 0x3"):
        adapter.embed("hello")
    assert "job cleanup failed" in caplog.text


def test_nonzero_zero_point_is_subtracted_during_dequantization() -> None:
    lib = _FakeLib()
    adapter = _adapter_for_job_test(lib)
    adapter._zero_point = 5
    adapter._scale = 2.0

    assert adapter.embed("hello") == [-10.0, -10.0]


def test_close_retries_graph_without_deinitializing_its_context() -> None:
    lib = _FakeLib()
    lib.noe_unload_graph.result = 4
    adapter = _adapter_for_job_test(lib)
    adapter._ctx = object()
    adapter._tokenizer = object()
    adapter._graph_loaded = True
    adapter._context_initialized = True

    with pytest.raises(RuntimeError, match="unload_graph"):
        adapter.close()
    assert lib.noe_deinit_context.calls == 0

    lib.noe_unload_graph.result = 0
    adapter.close()
    assert lib.noe_unload_graph.calls == 2
    assert lib.noe_deinit_context.calls == 1
    assert adapter._closed is True
    assert adapter._tokenizer is None


def test_close_does_not_retry_an_unloaded_graph() -> None:
    lib = _FakeLib()
    lib.noe_deinit_context.result = 5
    adapter = _adapter_for_job_test(lib)
    adapter._ctx = object()
    adapter._tokenizer = object()
    adapter._graph_loaded = True
    adapter._context_initialized = True

    with pytest.raises(RuntimeError, match="deinit_context"):
        adapter.close()

    lib.noe_deinit_context.result = 0
    adapter.close()
    assert lib.noe_unload_graph.calls == 1
    assert lib.noe_deinit_context.calls == 2
    assert adapter._closed is True


def test_close_success_is_idempotent() -> None:
    lib = _FakeLib()
    adapter = _adapter_for_job_test(lib)
    adapter._ctx = object()
    adapter._tokenizer = object()
    adapter._graph_loaded = True
    adapter._context_initialized = True

    adapter.close()
    adapter.close()

    assert lib.noe_unload_graph.calls == 1
    assert lib.noe_deinit_context.calls == 1
    assert adapter._closed is True


@pytest.mark.skipif(not AVAILABLE, reason="adapter unavailable")
def test_embed_dim_consistent() -> None:
    adapter = CixZhouyiAdapter(
        _model_path(),
        tokenizer_path=_tokenizer_path(),
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
    adapter = CixZhouyiAdapter(
        _model_path(),
        tokenizer_path=_tokenizer_path(),
    )
    adapter.close()
    adapter.close()
