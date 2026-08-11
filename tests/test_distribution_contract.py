from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

import embedkit
from embedkit.bench import _texts_from_corpus

ROOT = Path(__file__).parents[1]


def test_public_version_matches_distribution_metadata() -> None:
    assert embedkit.__version__ == version("mnemos-embedkit")


def test_primary_install_instructions_use_distribution_name() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install embedkit[" not in readme
    assert "mnemos-embedkit[all-cpu]" in readme


def test_packaging_advertises_only_implemented_runtime_extras() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = set(project["optional-dependencies"])
    assert extras == {
        "cpu-llamacpp",
        "gpu-nvidia-cuda",
        "npu-cix",
        "all-cpu",
        "all-x86-cuda",
        "all-arm-cix",
        "all",
        "dev",
    }


def test_included_smoke_corpus_is_readable() -> None:
    texts = _texts_from_corpus(ROOT / "benches" / "corpora" / "smoke.json")
    assert len(texts) == 3


def test_mnemos_docs_do_not_claim_integration_exists() -> None:
    setup = (ROOT / "docs" / "CIX-SKY1-NPU-STANDALONE-SETUP.md").read_text(
        encoding="utf-8"
    )
    assert "MNEMOS core does not currently import Embedkit" in setup
    assert "it calls `embedkit.Engine.auto()` on ingest" not in setup
