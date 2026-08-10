# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jason Perlow
"""Command-line adapter availability report."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .engine import Engine


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report embedkit adapter availability")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    adapters = Engine.list_adapters()
    if args.json:
        print(json.dumps(adapters, indent=2))
    else:
        for adapter in adapters:
            state = "available" if adapter["available"] else "unavailable"
            print(f"{adapter['name']}: {state} ({adapter['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
