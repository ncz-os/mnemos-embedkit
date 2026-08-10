# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jason Perlow
from __future__ import annotations

import json
from pathlib import Path

import pytest

from embedkit import bench, doctor


def test_bench_command_help() -> None:
    with pytest.raises(SystemExit) as exc_info:
        bench.main(["--help"])
    assert exc_info.value.code == 0


def test_bench_command_runs_corpus(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class FakeEngine:
        @classmethod
        def auto(cls, model: str | None = None) -> FakeEngine:
            return cls()

        def embed(self, text: str) -> list[float]:
            return [1.0]

        def info(self) -> dict[str, object]:
            return {"adapter": "fake", "embed_dim": 1}

        def close(self) -> None:
            pass

    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"records": [{"payload": {"content": "hello"}}]}))
    monkeypatch.setattr(bench, "Engine", FakeEngine)

    assert bench.main(["--corpus", str(corpus), "--engines", "auto"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["records_embedded"] == 1


def test_doctor_reports_adapters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        doctor.Engine,
        "list_adapters",
        lambda: [{"name": "fake", "tier": "cpu", "available": True, "reason": "ready"}],
    )

    assert doctor.main([]) == 0
    assert capsys.readouterr().out == "fake: available (ready)\n"
