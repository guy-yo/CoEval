"""EXP-V2 item A: ranking stability of the cross-family ensemble vs single judges.

On the exact-match QA experiment (EXP001b) the ground-truth student ranking is
gpt-4o-mini > gpt-3.5-turbo > llama-3.2-3b (true accuracies 0.969 > 0.942 > 0.832).
We bootstrap the 191 datapoints B times; for each resample we rank the students by
(a) each single judge and (b) the cross-family ensemble mean, and measure how often
each method recovers the TRUE ranking and the true top-2 order. A more reliable
evaluator recovers the true ranking on a larger fraction of resamples.

Run:  python scripts/v2_ranking_stability.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from analyzer.loader import load_ees  # noqa: E402

RUN = ROOT / "Runs" / "EXP001b-exactmatch-qa"
ACCURACY_ASPECTS = {"accuracy"}
ENSEMBLE = {"gpt-4o", "claude-sonnet-4", "gemini-flash"}  # frontier cross-family panel
GT = {"gpt-4o-mini": 0.969, "gpt-3.5-turbo": 0.942, "llama-3.2-3b": 0.832}
TRUE_ORDER = ["gpt-4o-mini", "gpt-3.5-turbo", "llama-3.2-3b"]
B = 2000
SEED = 0


def _norm_judge(j: str) -> str:
    j = j.lower()
    for k in ("gpt-4o", "claude-sonnet-4", "gemini", "gpt-3.5-turbo", "claude-haiku", "gpt-4o-mini"):
        if k in j:
            return "gemini-flash" if k == "gemini" else k
    return j


def main():
    model = load_ees(RUN, partial_ok=True)
    # score[judge][student][datapoint] -> list of score_norm (accuracy aspect)
    cells: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    judges, students, dps = set(), set(), set()
    for u in model.units:
        if u.rubric_aspect not in ACCURACY_ASPECTS:
            continue
        j = _norm_judge(u.judge_model_id)
        cells[j][u.student_model_id][u.datapoint_id].append(u.score_norm)
        judges.add(j); students.add(u.student_model_id); dps.add(u.datapoint_id)
    dps = sorted(dps)
    print(f"judges={sorted(judges)}\nstudents={sorted(students)}\nn_datapoints={len(dps)}")

    def student_means(judge_set, sampled_dps):
        """per-student mean over the given judges and sampled datapoints."""
        out = {}
        for s in TRUE_ORDER:
            vals = []
            for dp in sampled_dps:
                jv = []
                for j in judge_set:
                    v = cells[j][s].get(dp)
                    if v:
                        jv.append(np.mean(v))
                if jv:
                    vals.append(np.mean(jv))  # mean over judges for this datapoint
            out[s] = float(np.mean(vals)) if vals else float("nan")
        return out

    methods = {f"single:{j}": {j} for j in sorted(judges)}
    methods["ENSEMBLE (cross-family)"] = ENSEMBLE & judges

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(dps))
    recover_full = defaultdict(int)
    recover_top2 = defaultdict(int)
    for _ in range(B):
        samp = [dps[i] for i in rng.choice(idx, size=len(idx), replace=True)]
        for name, jset in methods.items():
            if not jset:
                continue
            m = student_means(jset, samp)
            order = sorted(TRUE_ORDER, key=lambda s: -m[s])
            if order == TRUE_ORDER:
                recover_full[name] += 1
            if order[:2] == TRUE_ORDER[:2]:
                recover_top2[name] += 1

    print(f"\nBootstrap B={B}: fraction of resamples recovering the TRUE ranking")
    rows = []
    for name in sorted(methods, key=lambda n: -recover_full[n] / B):
        if not methods[name]:
            continue
        rows.append({
            "method": name,
            "p_full_ranking": round(recover_full[name] / B, 4),
            "p_top2_order": round(recover_top2[name] / B, 4),
        })
        print(f"  {name:28} full={recover_full[name]/B:.3f}  top2={recover_top2[name]/B:.3f}")

    out = {"experiment": "v2_ranking_stability", "run": str(RUN), "B": B,
           "true_order": TRUE_ORDER, "ground_truth_acc": GT, "results": rows}
    (RUN / "reports" / "v2_ranking_stability.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", RUN / "reports" / "v2_ranking_stability.json")


if __name__ == "__main__":
    main()
