# EXP-V2 — bad-judge tolerance: independence is the safeguard (WIN)

**Status:** completed · **Date:** 2026-06-02 · **Cost:** $0 (EXP001 logged data)
**Script:** `scripts/v2_tolerance_curve.py` · **Artifact:** `Runs/EXP001-.../reports/v2_tolerance_curve.json`

## Hypothesis
How many bad judges can reliability weighting tolerate? Prediction: any number of
INDEPENDENT bad judges (they never agree with each other -> stay down-weighted), but a
CORRELATED coalition hijacks once it outnumbers the competent judges. If so, vendor
diversity (independence) is the precondition for the panel's robustness.

## Results (WIN, clean numbers; n=4 real judges, gold = benchmark-native, n=900)
INDEPENDENT random bad judges (k added):
  reliability-weighted accuracy stays ~0.25 for ALL k (k=6 -> 0.249) while plain mean
  degrades and gets noisy (k=6 -> 0.162). Bad-judge total weight stays 0.0-0.10.
  => robust to ANY number of independent bad judges.

CORRELATED coalition (all anti-correlated, mutually agreeing):
  k<=2 (minority): reliability-weighted holds (0.246, 0.216), coalition weight 0.0.
  k=3-4 (parity): unstable transition.
  k>=5 (coalition outnumbers the 4 good judges): HIJACK -- coalition weight 1.0, good
  judges 0.0, accuracy inverts to -0.23. The coalition becomes the false consensus.

## The lesson (unifies robustness + cross-family)
Reliability weighting suppresses bad judges only when their errors are INDEPENDENT.
A correlated majority breaks it. Cross-family / vendor-disjoint composition is exactly
what keeps judge errors independent (independent vendors are unlikely to share a bias
coalition), so the robustness mechanism (Sec 5.2) and the composition principle (the
central thesis) are two sides of one coin.

## Integrated
Tight extension to Sec 5.2 (self-validating-panel discussion): robust to any number of
independent broken judges; fails only to a correlated majority, which cross-family
composition prevents.
