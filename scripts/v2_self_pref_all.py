"""EXP-V2: self-preference across ALL verticals, with bootstrap CIs (addresses the
selective-reporting concern). The panel includes gpt-4o-mini as both judge and
candidate. Per item we form a difference-in-differences that controls for a judge's
overall harshness:

  per-item d = [gpt-4o-mini JUDGE: s(gpt-4o-mini) - s(gpt-3.5-turbo)]
             - [cross-family mean: s(gpt-4o-mini) - s(gpt-3.5-turbo)]
  self_preference = mean_items(d);  95% CI by resampling items.

Run: python scripts/v2_self_pref_all.py
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from analyzer.loader import load_ees  # noqa: E402

RUNS = {"drug_interaction_reasoning": "Runs/EXP007-ddi-vertical",
        "clinical_reasoning": "Runs/EXP006-vertical-case-studies",
        "legal_analysis": "Runs/EXP006-vertical-case-studies"}
INFAM, SELF, RIVAL = "gpt-4o-mini", "gpt-4o-mini", "gpt-3.5-turbo"
B, SEED = 5000, 0


def per_item_d(run, task):
    m = load_ees(run, partial_ok=True)
    # cell[item][judge][student] = mean score over aspects
    cell = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for u in m.units:
        if u.task_id != task:
            continue
        cell[u.datapoint_id][u.judge_model_id][u.student_model_id].append(u.score_norm)
    judges = sorted({j for it in cell.values() for j in it})
    cross = [j for j in judges if j != INFAM]
    ds = []
    for it, jd in cell.items():
        def s(judge, stu):
            v = jd.get(judge, {}).get(stu)
            return float(np.mean(v)) if v else None
        infam = (s(INFAM, SELF), s(INFAM, RIVAL))
        if None in infam:
            continue
        cr = [(s(c, SELF), s(c, RIVAL)) for c in cross]
        cr = [(a, b) for a, b in cr if a is not None and b is not None]
        if not cr:
            continue
        d_same = infam[0] - infam[1]
        d_cross = np.mean([a - b for a, b in cr])
        ds.append(d_same - d_cross)
    return np.array(ds)


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for task, run in RUNS.items():
        d = per_item_d(run, task)
        point = float(np.mean(d))
        boot = [float(np.mean(rng.choice(d, size=len(d), replace=True))) for _ in range(B)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_gt0 = float(np.mean(np.array(boot) <= 0))  # one-sided p that effect > 0
        out[task] = {"n_items": len(d), "self_preference": round(point, 4),
                     "ci95": [round(lo, 4), round(hi, 4)],
                     "p_one_sided_gt0": round(p_gt0, 4),
                     "significant_flip_direction": bool(lo > 0)}
        print(f"{task:28} self-pref={point:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  "
              f"p(>0)={p_gt0:.3f}  CI>0={lo>0}")
    (ROOT / "Runs/EXP006-vertical-case-studies/reports/v2_self_pref_all.json").write_text(json.dumps(out, indent=2))
    print("\nwrote v2_self_pref_all.json")


if __name__ == "__main__":
    main()
