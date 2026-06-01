"""Contamination check: verbatim 13-gram overlap of CoEval-generated items
against public benchmarks that are present in current models' pretraining data.

Near-zero overlap corroborates (does not by itself prove) the structural
freshness guarantee: items synthesized fresh per run are not verbatim copies of
the sampled public test sets. Writes Runs/medium-benchmark/reports/contamination_check.json.

Usage:  python scripts/contamination_check.py [--n 13]
"""
from __future__ import annotations
import argparse, json, glob, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (t or "").lower())


def ngrams(t: str, n: int) -> set:
    w = words(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)} if len(w) >= n else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=13, help="n-gram size")
    args = ap.parse_args()
    n = args.n

    # CoEval-generated items (teacher-synthesized prompts + references)
    gen = []
    for f in glob.glob(str(ROOT / "Runs/medium-benchmark/phase3_datapoints/*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            gen.append(r.get("prompt", "") + " " + r.get("reference_response", ""))

    # Public benchmark items present in pretraining corpora (XSum, CNN/DM,
    # CodeSearchNet, SciQ, ARC-Challenge), as staged in the EXP-001* runs.
    pub_ng: set = set()
    n_public = 0
    for f in glob.glob(str(ROOT / "Runs/EXP001*/phase3_datapoints/*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            n_public += 1
            pub_ng |= ngrams(r.get("prompt", "") + " " + r.get("reference_response", ""), n)

    overlaps = []
    for g in gen:
        gg = ngrams(g, n)
        if gg:
            overlaps.append(len(gg & pub_ng) / len(gg))
    ov = np.array(overlaps)

    out = {
        "metric": f"{n}-gram verbatim overlap, CoEval-generated vs public benchmarks",
        "public_benchmarks": ["xsum", "cnn_dailymail", "codesearchnet", "sciq", "arc_challenge"],
        "n_generated": len(gen),
        "n_public": n_public,
        "n_public_distinct_ngrams": len(pub_ng),
        "mean_overlap": round(float(ov.mean()), 6),
        "max_overlap": round(float(ov.max()), 6),
        "frac_items_any_overlap": round(float((ov > 0).mean()), 6),
    }
    outp = ROOT / "Runs/medium-benchmark/reports/contamination_check.json"
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
