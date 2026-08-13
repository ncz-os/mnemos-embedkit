# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jason Perlow
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embedkit.adapters.gpu_nvidia_cuda import NvidiaCUDAAdapter

AVAILABLE = NvidiaCUDAAdapter.is_available()[0]
MODEL_ENV = "EMBEDKIT_TEST_NVIDIA_CUDA_MODEL"
TOKENIZER_ENV = "EMBEDKIT_TEST_NVIDIA_CUDA_TOKENIZER"


def _model_path() -> str:
    value = os.environ.get(MODEL_ENV)
    if not value:
        pytest.skip(f"set {MODEL_ENV} to run nvidia-cuda adapter smoke tests")
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
        ok, reason = NvidiaCUDAAdapter.is_available()
        assert isinstance(ok, bool)
        assert isinstance(reason, str)
        assert reason
    assert time.perf_counter() - started < 10.0


def test_cuda_batching_is_bounded() -> None:
    batch_lengths: list[int] = []

    class FakeTokenizer:
        def __call__(self, texts: list[str], **kwargs: object) -> dict[str, np.ndarray]:
            batch_lengths.append(len(texts))
            return {
                "input_ids": np.ones((len(texts), 2), dtype=np.int64),
                "attention_mask": np.ones((len(texts), 2), dtype=np.int64),
            }

    class FakeSession:
        def run(self, outputs: object, feeds: object) -> list[np.ndarray]:
            batch = len(feeds["input_ids"])
            return [np.ones((batch, 2), dtype=np.float32)]

    adapter = NvidiaCUDAAdapter.__new__(NvidiaCUDAAdapter)
    adapter._session = FakeSession()
    adapter._tokenizer = FakeTokenizer()
    adapter._input_meta = [SimpleNamespace(name="input_ids", type="tensor(int64)")]
    adapter._output_names = ["sentence_embedding"]
    adapter._normalize = False
    adapter.batch_size = 2

    assert adapter.embed_batch(["a", "b", "c", "d", "e"]) == [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    assert batch_lengths == [2, 2, 1]


def test_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        NvidiaCUDAAdapter("model.onnx", batch_size=0)


@pytest.mark.skipif(not AVAILABLE, reason="adapter unavailable")
def test_embed_dim_consistent() -> None:
    adapter = NvidiaCUDAAdapter(
        _model_path(),
        tokenizer_path=_tokenizer_path(),
        max_tokens=128,
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
    adapter = NvidiaCUDAAdapter(
        _model_path(),
        tokenizer_path=_tokenizer_path(),
        max_tokens=128,
    )
    adapter.close()
    adapter.close()
