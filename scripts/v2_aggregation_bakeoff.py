"""EXP-V2 item B (revised): judge-choice regret + label-free aggregation bake-off.

Computed through the SAME analyzer code path as Section 5.1
(`analyzer.benchmark_correlation`), on the SAME canonical benchmark-grounded set
(all three benchmark-scored tasks, four-judge panel, pooled). The plain-mean
aggregate therefore reproduces the Section-5.1 ensemble number and the best
single judge reproduces the Section-5.1 best-single number exactly, so Section
5.2 is consistent with Section 5.1 by construction.

Aggregators compared, all unsupervised (no ground truth used to fit them):
  mean (Sec 5.1 ensemble), median, trimmed-mean, reliability-weighted
  (panel-agreement weighting), Dawid-Skene (EM over discretized levels).
Metric: Spearman rho of the aggregate score vs the benchmark-native score,
pooled over the canonical set, via `analyzer.stats.correlation_ci`.

Run:  python scripts/v2_aggregation_bakeoff.py
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from analyzer.loader import load_ees  # noqa: E402
from analyzer.stats import correlation_ci  # noqa: E402
from analyzer.benchmark_correlation import (  # noqa: E402
    _load_response_benchmark_scores,
    _coeval_scores_by_response,
)

RUN = ROOT / "Runs" / "EXP001-benchmark-grounded-comparison"


def _rho(agg: dict, bench: dict) -> float:
    xs, ys = [], []
    for rid, b in bench.items():
        if rid in agg:
            xs.append(agg[rid])
            ys.append(b["score"])
    return float(correlation_ci(xs, ys, method="spearman").point)


def _dawid_skene(labels, n_levels=3, n_iter=80):
    rids = list(labels.keys())
    judges = sorted({j for r in rids for j in labels[r]})
    L, J = n_levels, len(judges)
    jidx = {j: k for k, j in enumerate(judges)}
    T = np.zeros((len(rids), L))
    for i, r in enumerate(rids):
        for j, lv in labels[r].items():
            T[i, lv] += 1.0
        T[i] = T[i] / T[i].sum() if T[i].sum() else np.ones(L) / L
    for _ in range(n_iter):
        p = T.sum(0) / len(rids)
        pi = np.ones((J, L, L)) * 1e-6
        for i, r in enumerate(rids):
            for j, lv in labels[r].items():
                pi[jidx[j], :, lv] += T[i]
        pi = pi / pi.sum(2, keepdims=True)
        newT = np.zeros_like(T)
        for i, r in enumerate(rids):
            logp = np.log(p + 1e-12)
            for j, lv in labels[r].items():
                logp = logp + np.log(pi[jidx[j], :, lv] + 1e-12)
            logp -= logp.max()
            w = np.exp(logp)
            newT[i] = w / w.sum()
        if np.abs(newT - T).max() < 1e-6:
            T = newT
            break
        T = newT
    levels = np.arange(L)
    return {r: float(T[i] @ levels) for i, r in enumerate(rids)}


def main():
    model = load_ees(RUN, partial_ok=True)
    bench = _load_response_benchmark_scores(RUN)
    judges = sorted(model.judges)

    # Plain-mean ensemble and per-judge, via the canonical Sec 5.1 functions.
    ens = _coeval_scores_by_response(model)
    mean_rho = _rho(ens, bench)
    single = {}
    for j in judges:
        jc = _coeval_scores_by_response(model, judge_filter={j})
        single[j] = _rho(jc, bench)
    best_j = max(single, key=single.get)
    worst_j = min(single, key=single.get)

    # Per-response, per-judge mean score (for the weighted/robust aggregators).
    perj = defaultdict(dict)
    raw = defaultdict(lambda: defaultdict(list))
    for u in model.units:
        raw[u.response_id][u.judge_model_id].append(u.score_norm)
    for rid, d in raw.items():
        for j, vs in d.items():
            perj[rid][j] = float(np.mean(vs))
    rids = [r for r in perj if r in bench]

    def agg_dict(fn):
        out = {}
        for r in rids:
            vals = [perj[r][j] for j in judges if j in perj[r]]
            if vals:
                out[r] = float(fn(vals))
        return out

    median_rho = _rho(agg_dict(np.median), bench)

    def trimmed(vals):
        vals = sorted(vals)
        return np.mean(vals[1:-1]) if len(vals) > 2 else np.mean(vals)
    trim_rho = _rho(agg_dict(trimmed), bench)

    # Unsupervised reliability weights: each judge's mean pairwise agreement with
    # the rest of the panel (leave-one-out), no ground truth used.
    rel = {}
    for j in judges:
        agree = []
        for o in judges:
            if o == j:
                continue
            xs = [(perj[r][j], perj[r][o]) for r in rids if j in perj[r] and o in perj[r]]
            if len(xs) > 5:
                a = np.array([p[0] for p in xs]); b = np.array([p[1] for p in xs])
                if a.std() > 0 and b.std() > 0:
                    agree.append(spearmanr(a, b).correlation)
        rel[j] = max(0.0, float(np.mean(agree))) if agree else 0.0
    wsum = sum(rel.values()) or 1.0
    w = {j: rel[j] / wsum for j in judges}
    wagg = {}
    for r in rids:
        num = sum(w[j] * perj[r][j] for j in judges if j in perj[r])
        den = sum(w[j] for j in judges if j in perj[r]) or 1.0
        wagg[r] = num / den
    weighted_rho = _rho(wagg, bench)

    # Dawid-Skene over discretized 0/1/2 levels.
    def level(v):
        return 0 if v < 1 / 3 else (1 if v < 2 / 3 else 2)
    labels = {r: {j: level(perj[r][j]) for j in judges if j in perj[r]} for r in rids}
    ds_rho = _rho(_dawid_skene(labels), bench)

    out = {
        "experiment": "v2_aggregation_bakeoff",
        "task": "benchmark-grounded (3 tasks: code BLEU + news/text BERTScore), pooled, via analyzer.benchmark_correlation",
        "n_responses": len(bench),
        "judges": judges,
        "single_judge_rho": {j: round(single[j], 4) for j in judges},
        "best_single": {best_j: round(single[best_j], 4)},
        "worst_single": {worst_j: round(single[worst_j], 4)},
        "judge_choice_regret_range": round(single[best_j] - single[worst_j], 4),
        "reliability_weights": {j: round(w[j], 3) for j in judges},
        "aggregations": {
            "mean (Sec 5.1 ensemble)": round(mean_rho, 4),
            "median": round(median_rho, 4),
            "trimmed_mean": round(trim_rho, 4),
            "reliability_weighted": round(weighted_rho, 4),
            "dawid_skene": round(ds_rho, 4),
        },
    }
    (RUN / "reports" / "v2_aggregation_bakeoff.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
