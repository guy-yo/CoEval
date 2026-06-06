#!/usr/bin/env python3
"""Summarise every Runs/EXP-guy-* experiment into a Markdown results table.

Usage:
    python scripts/summarize_runs.py            # print to stdout
    python scripts/summarize_runs.py > EXPERIMENTS_table.md

Scans each run folder for Phase-3 datapoints (attribute coverage) and Phase-5
evaluations (per-student accuracy), so the experiment log can be regenerated
after every new run instead of being updated by hand.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

RUNS_GLOB = "Runs/EXP-guy-*"


def _load_jsonl(pattern: str) -> list[dict]:
    out: list[dict] = []
    for f in glob.glob(pattern):
        with open(f, encoding="utf-8") as fh:
            out += [json.loads(line) for line in fh if line.strip()]
    return out


def summarise(run_dir: str) -> dict:
    meta = {}
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))

    dps = _load_jsonl(os.path.join(run_dir, "phase3_datapoints", "*.jsonl"))
    combos, full = set(), 0
    for r in dps:
        a = r.get("sampled_target_attributes", {})
        combos.add((a.get("label"), a.get("sender_type"), a.get("quality")))
        if len(a) == 3:
            full += 1

    tot: dict[str, float] = defaultdict(float)
    n: dict[str, int] = defaultdict(int)
    for e in _load_jsonl(os.path.join(run_dir, "phase5_evaluations", "*.jsonl")):
        sid = e["response_id"].split("__")[3]
        score = float(list(e["scores"].values())[0])
        tot[sid] += score
        n[sid] += 1
    acc = {s: (tot[s] / n[s], int(tot[s]), n[s]) for s in sorted(tot)}

    return {
        "id": os.path.basename(run_dir),
        "status": meta.get("status", "?"),
        "datapoints": len(dps),
        "full_attrs": full,
        "distinct_combos": len(combos),
        "accuracy": acc,
    }


def main() -> None:
    runs = sorted(glob.glob(RUNS_GLOB))
    if not runs:
        print("No Runs/EXP-guy-* folders found.")
        return

    print("| Run | Status | Datapoints | Full attrs (3/3) | Distinct combos |")
    print("|-----|--------|-----------:|-----------------:|----------------:|")
    summaries = [summarise(r) for r in runs]
    for s in summaries:
        print(
            f"| `{s['id']}` | {s['status']} | {s['datapoints']} | "
            f"{s['full_attrs']} | {s['distinct_combos']} |"
        )

    print("\n### Per-student accuracy (exact_match)\n")
    for s in summaries:
        if not s["accuracy"]:
            continue
        cells = ", ".join(
            f"{name} {a:.2f} ({c}/{t})" for name, (a, c, t) in s["accuracy"].items()
        )
        print(f"- **{s['id']}**: {cells}")


if __name__ == "__main__":
    main()
