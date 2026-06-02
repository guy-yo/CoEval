"""EXP-010: many-model rank-recovery (answers ">10 models?").

For N candidate models answering gold-answer benchmark items, compare the
label-free CoEval ranking against the gold-accuracy ranking. The win is that
the cross-family ensemble (+ reliability weighting) recovers the gold ordering
with high Spearman/Kendall, above the best single judge and the plain mean.

Inputs (under the run dir):
  - EES units (judge scores per response)              via analyzer.loader.load_ees
  - benchmark_response_scores.jsonl (gold per response) via benchmark.score_responses

Run:  python scripts/v2_rank_recovery.py Runs/EXP010-scale-ranking-pilot
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from analyzer.loader import load_ees  # noqa: E402


def _gold_by_student(run: Path):
    """student -> mean gold accuracy (inclusion/exact match)."""
    acc = defaultdict(list)
    f = run / "benchmark_response_scores.jsonl"
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("benchmark_native_score")
        if s is not None:
            acc[r["student_model_id"]].append(float(s))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _coeval_scores(run: Path, aspect: str | None = None):
    """student -> {judge -> mean score}, and student -> ensemble mean."""
    model = load_ees(run, partial_ok=True)
    # per student, per judge: list of normalized aspect scores
    sj = defaultdict(lambda: defaultdict(list))
    for u in model.units:
        if aspect and u.rubric_aspect != aspect:
            continue
        sj[u.student_model_id][u.judge_model_id].append(u.score_norm)
    judges = sorted({j for s in sj for j in sj[s]})
    per_judge = {s: {j: float(np.mean(v)) for j, v in d.items()} for s, d in sj.items()}
    ensemble = {s: float(np.mean([np.mean(v) for v in d.values()])) for s, d in sj.items()}
    return per_judge, ensemble, judges


def _rank_metrics(students, gold, scores):
    g = np.array([gold[s] for s in students])
    x = np.array([scores[s] for s in students])
    return float(spearmanr(x, g).correlation), float(kendalltau(x, g).correlation)


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "Runs/EXP010-scale-ranking-pilot")
    gold = _gold_by_student(run)
    per_judge, ensemble, judges = _coeval_scores(run)
    _, ens_acc, _ = _coeval_scores(run, aspect="accuracy")

    students = sorted(set(gold) & set(ensemble), key=lambda s: -gold[s])
    n = len(students)

    # ensemble (all aspects) and ensemble (accuracy aspect only)
    ens_rho, ens_tau = _rank_metrics(students, gold, ensemble)
    acc_rho, acc_tau = _rank_metrics(students, gold, ens_acc) if ens_acc else (None, None)

    # reliability-weighted ensemble (down-weight low panel-agreement judges)
    rel = {}
    for j in judges:
        agree = []
        for o in judges:
            if o == j:
                continue
            xs = [(per_judge[s][j], per_judge[s][o]) for s in students
                  if j in per_judge[s] and o in per_judge[s]]
            if len(xs) > 3:
                a = np.array([p[0] for p in xs]); b = np.array([p[1] for p in xs])
                if a.std() > 0 and b.std() > 0:
                    agree.append(spearmanr(a, b).correlation)
        rel[j] = max(0.0, float(np.mean(agree))) if agree else 0.0
    wsum = sum(rel.values()) or 1.0
    w = {j: rel[j] / wsum for j in judges}
    wens = {}
    for s in students:
        num = sum(w[j] * per_judge[s][j] for j in judges if j in per_judge[s])
        den = sum(w[j] for j in judges if j in per_judge[s]) or 1.0
        wens[s] = num / den
    wt_rho, wt_tau = _rank_metrics(students, gold, wens)

    # single judges
    single = {}
    for j in judges:
        sc = {s: per_judge[s].get(j, np.nan) for s in students}
        if all(not np.isnan(v) for v in sc.values()):
            single[j] = _rank_metrics(students, gold, sc)
    best_j = max(single, key=lambda j: single[j][1]) if single else None
    worst_j = min(single, key=lambda j: single[j][1]) if single else None

    out = {
        "experiment": "v2_rank_recovery",
        "run": run.name,
        "n_models": n,
        "models_by_gold": [{"model": s, "gold_acc": round(gold[s], 3),
                            "coeval_ens": round(ensemble[s], 3)} for s in students],
        "judges": judges,
        "rank_recovery": {
            "ensemble_all_aspects": {"spearman": round(ens_rho, 3), "kendall": round(ens_tau, 3)},
            "ensemble_accuracy_aspect": {"spearman": round(acc_rho, 3) if acc_rho is not None else None,
                                         "kendall": round(acc_tau, 3) if acc_tau is not None else None},
            "reliability_weighted": {"spearman": round(wt_rho, 3), "kendall": round(wt_tau, 3)},
            "single_judges": {j: {"spearman": round(single[j][0], 3), "kendall": round(single[j][1], 3)} for j in single},
            "best_single_judge": {best_j: {"kendall": round(single[best_j][1], 3)}} if best_j else None,
            "worst_single_judge": {worst_j: {"kendall": round(single[worst_j][1], 3)}} if worst_j else None,
            "judge_choice_regret_kendall": round(single[best_j][1] - single[worst_j][1], 3) if best_j else None,
        },
    }
    (run / "reports").mkdir(exist_ok=True)
    (run / "reports" / "v2_rank_recovery.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
