"""EXP-V2: how many bad judges can the panel tolerate? (robustness vs independence)

On EXP001 (4 real judges, gold = benchmark-native score) we add k bad judges and
track plain-mean vs reliability-weighted accuracy. Two regimes:

  INDEPENDENT bad judges (each uniform-random, uncorrelated): they never agree with
  each other or the panel, so reliability weighting keeps their weight ~0 for ANY k.

  CORRELATED coalition (all anti-correlated with quality the same way, so they agree
  with EACH OTHER): once the coalition outnumbers the competent judges it forms a
  false consensus and hijacks reliability weighting. This is precisely why CoEval
  requires VENDOR DIVERSITY: independent vendors are unlikely to share a bias coalition.

Run:  python scripts/v2_tolerance_curve.py
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
from analyzer.benchmark_correlation import _load_response_benchmark_scores  # noqa: E402

RUN = ROOT / "Runs" / "EXP001-benchmark-grounded-comparison"


def _rho(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)


def main():
    model = load_ees(RUN, partial_ok=True)
    bench = _load_response_benchmark_scores(RUN)
    raw = defaultdict(lambda: defaultdict(list))
    for u in model.units:
        if u.response_id in bench:
            raw[u.response_id][u.judge_model_id].append(u.score_norm)
    rids = [r for r in raw if r in bench]
    real = sorted({j for r in rids for j in raw[r]})
    gold = np.array([bench[r]["score"] for r in rids])
    R = {j: np.array([np.mean(raw[r][j]) if raw[r].get(j) else np.nan for r in rids]) for j in real}
    consensus = np.nanmean(np.vstack([R[j] for j in real]), axis=0)
    rng = np.random.default_rng(0)

    def agg(cols, weighted):
        if weighted:
            w = {}
            for j in cols:
                ags = []
                for o in cols:
                    if o == j:
                        continue
                    a = _rho(cols[j], cols[o])
                    if not np.isnan(a):
                        ags.append(a)
                w[j] = max(0.0, float(np.mean(ags))) if ags else 0.0
            tot = sum(w.values()) or 1.0
            w = {j: w[j] / tot for j in cols}
        else:
            w = {j: 1.0 / len(cols) for j in cols}
        num = np.zeros(len(rids)); den = np.zeros(len(rids))
        for j in cols:
            v = cols[j]; ok = ~np.isnan(v)
            num[ok] += w[j] * v[ok]; den[ok] += w[j]
        return _rho(np.where(den > 0, num / np.where(den == 0, 1, den), np.nan), gold), w

    def run_regime(kind, kmax=6):
        curve = []
        for k in range(0, kmax + 1):
            cols = dict(R)
            for b in range(k):
                if kind == "independent":
                    cols[f"BAD{b}"] = rng.uniform(0, 1, size=len(rids))
                else:  # correlated coalition: all ~ (1 - consensus) + small noise
                    cols[f"BAD{b}"] = np.clip(1.0 - consensus + rng.normal(0, 0.05, len(rids)), 0, 1)
            pm, _ = agg(cols, weighted=False)
            rw, w = agg(cols, weighted=True)
            badw = sum(w[j] for j in w if j.startswith("BAD"))
            curve.append({"k_bad": k, "plain_mean": round(pm, 3),
                          "reliability_weighted": round(rw, 3),
                          "bad_total_weight": round(badw, 3)})
        return curve

    out = {
        "experiment": "v2_tolerance_curve", "run": RUN.name,
        "n_real_judges": len(real), "n_responses": len(rids),
        "clean_accuracy": round(_rho(consensus, gold), 3),
        "independent_bad_judges": run_regime("independent"),
        "correlated_coalition": run_regime("coalition"),
    }
    (RUN / "reports" / "v2_tolerance_curve.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
