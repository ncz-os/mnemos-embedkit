"""Lazy registry of implemented adapters.

Importing :mod:`embedkit` must not import every optional inference stack. The
registry therefore keeps only lightweight metadata and resolves an adapter
class when its availability is probed or it is constructed.
"""
from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .base import AbstractAdapter

AvailabilityGate = Callable[[], tuple[bool, str] | None]


def _module_gate(module: str, distribution: str) -> tuple[bool, str] | None:
    if importlib.util.find_spec(module) is None:
        return False, f"{distribution} not installed"
    return None


def _cix_gate() -> tuple[bool, str] | None:
    if not Path("/dev/aipu").exists():
        return False, "/dev/aipu not present (Cix NPU not in this kernel)"
    return None


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """Lightweight, class-like adapter registry entry."""

    name: str
    tier: str
    model_format: str
    module: str
    class_name: str
    gate: AvailabilityGate | None = None

    def load(self) -> type[AbstractAdapter]:
        module = importlib.import_module(self.module, package=__package__)
        return cast(type[AbstractAdapter], getattr(module, self.class_name))

    def is_available(self) -> tuple[bool, str]:
        if self.gate is not None:
            result = self.gate()
            if result is not None:
                return result
        return self.load().is_available()

    def __call__(self, *args: Any, **kwargs: Any) -> AbstractAdapter:
        return self.load()(*args, **kwargs)


_REGISTRY: tuple[AdapterSpec, ...] = (
    AdapterSpec(
        "cpu-llamacpp",
        "cpu",
        "gguf",
        ".cpu_llamacpp",
        "CPULlamaCppAdapter",
        lambda: _module_gate("llama_cpp", "llama-cpp-python"),
    ),
    AdapterSpec(
        "nvidia-cuda",
        "gpu",
        "onnx",
        ".gpu_nvidia_cuda",
        "NvidiaCUDAAdapter",
        lambda: _module_gate("onnxruntime", "onnxruntime"),
    ),
    AdapterSpec(
        "cix-npu",
        "npu",
        "cix",
        ".npu_cix_zhouyi",
        "CixZhouyiAdapter",
        _cix_gate,
    ),
)


def all_adapters() -> list[AdapterSpec]:
    """Return every implemented adapter without importing adapter modules."""
    return list(_REGISTRY)


__all__ = ["AbstractAdapter", "AdapterSpec", "all_adapters"]
