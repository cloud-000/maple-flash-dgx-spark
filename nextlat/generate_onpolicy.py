#!/usr/bin/env python3
"""Dump Maple completions for NextLat training.

    uv run python nextlat/generate_onpolicy.py --model checkpoints/maple-2bit \
      --prompts tmp/mt_bench.jsonl -o nextlat/data/onpolicy.jsonl
"""

from __future__ import annotations

import sys

from maple_run.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["nextlat-synth", *sys.argv[1:]]))
