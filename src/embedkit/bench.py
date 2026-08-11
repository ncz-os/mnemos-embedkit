# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jason Perlow
"""Canonical command-line embedding benchmark."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .engine import Engine


def _texts_from_corpus(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as corpus_file:
        corpus = json.load(corpus_file)
    records = corpus.get("records") if isinstance(corpus, dict) else corpus
    if not isinstance(records, list):
        raise ValueError("corpus must be a JSON list or an object containing a 'records' list")

    texts: list[str] = []
    for record in records:
        text: object
        if isinstance(record, str):
            text = record
        elif isinstance(record, dict):
            payload = record.get("payload")
            text = payload.get("content") if isinstance(payload, dict) else record.get("content")
        else:
            text = None
        if isinstance(text, str) and text:
            texts.append(text)
    return texts


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(int(len(values) * fraction), len(values) - 1)]


def _bench(engine_name: str, model: str | None, texts: list[str]) -> dict[str, Any]:
    engine = Engine.auto(model=model) if engine_name == "auto" else Engine(engine_name, model=model)
    elapsed_ms: list[float] = []
    try:
        for text in texts:
            started = time.perf_counter()
            engine.embed(text)
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        info = engine.info()
    finally:
        engine.close()

    total_s = sum(elapsed_ms) / 1000.0
    return {
        **info,
        "records_embedded": len(elapsed_ms),
        "total_wall_s": round(total_s, 6),
        "rec_per_sec": round(len(elapsed_ms) / total_s, 3) if total_s else 0.0,
        "mean_ms": round(statistics.mean(elapsed_ms), 3),
        "p50_ms": round(_percentile(elapsed_ms, 0.50), 3),
        "p95_ms": round(_percentile(elapsed_ms, 0.95), 3),
        "p99_ms": round(_percentile(elapsed_ms, 0.99), 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark embedkit adapters against a corpus")
    parser.add_argument("--corpus", type=Path, required=True, help="JSON corpus path")
    parser.add_argument(
        "--engines",
        "--engine",
        default="auto",
        help="auto, all, or a comma-separated list of adapter names",
    )
    parser.add_argument("--model", help="model name or path")
    parser.add_argument("--output", type=Path, help="also write summary JSON to this path")
    args = parser.parse_args(argv)

    texts = _texts_from_corpus(args.corpus)
    if not texts:
        parser.error("corpus contains no non-empty text records")

    requested = [name.strip() for name in args.engines.split(",") if name.strip()]
    if requested == ["all"]:
        requested = [
            str(adapter["name"])
            for adapter in Engine.list_adapters()
            if adapter["available"]
        ]
    if not requested:
        parser.error("no engines selected")
    results = [_bench(engine_name, args.model, texts) for engine_name in requested]
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
