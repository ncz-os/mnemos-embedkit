"""Registry of implemented adapters.

Vendor runtimes are imported only by availability probes and constructors,
so importing this registry remains safe when optional runtimes are absent.
"""
from __future__ import annotations

from .base import AbstractAdapter
from .cpu_llamacpp import CPULlamaCppAdapter
from .gpu_nvidia_cuda import NvidiaCUDAAdapter
from .npu_cix_zhouyi import CixZhouyiAdapter

# Only implemented, contract-tested adapters belong in the shipping registry.
# Optional runtime dependencies are imported by constructors and availability
# probes, so importing these classes does not require the vendor runtime.
_REGISTRY: tuple[type[AbstractAdapter], ...] = (
    CPULlamaCppAdapter,
    NvidiaCUDAAdapter,
    CixZhouyiAdapter,
)


def all_adapters() -> list[type[AbstractAdapter]]:
    """Return every implemented adapter class."""
    return list(_REGISTRY)


__all__ = ["AbstractAdapter", "all_adapters"]
