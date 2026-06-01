# CoEval Experiment Registry

One row per experiment. New entries append; superseded runs are marked, not deleted.

| ID | Title | Date | Status | Cost | One-line result |
|----|-------|------|--------|------|-----------------|
| [EXP-002](2026-06-01_exp002_004_005_free_reanalysis.md) | Ensemble-size ablation | 2026-06-01 | completed | $0 | ICC(3,k) is **non-monotone**: peaks at k=2 (0.70) then falls to 0.40 as weak judges dilute the panel; Spearman-to-full rises 0.67→1.0. Motivates judge selection. |
| [EXP-004](2026-06-01_exp002_004_005_free_reanalysis.md) | Verbosity bias | 2026-06-01 | completed | $0 | Strong judges show **negative** length bias (GPT-3.5 r=−0.18), weak judges positive (SmolLM2 +0.23); ensemble r=+0.01 (CI straddles 0). mean\|indiv\|=0.153 → ensemble 0.010 (93% magnitude reduction). |
| [EXP-005](2026-06-01_exp002_004_005_free_reanalysis.md) | Rubric generalization | 2026-06-01 | completed | $0 | 22 criteria / 4 tasks; within-task sim 0.342 > cross-task 0.294; one universal cluster (`completeness`, 3 tasks), 11 task-specific singletons. |
| EXP-001 | Benchmark-grounded comparison + frontier judge + G-Eval | — | planned | ~$8-15 | Pending. Unblocked by `score_responses.py` integrity fix. |
| EXP-003 | Positional-bias / flip-rate | — | planned | ~$1.30 | Pending `positional_bias_analysis.py`. |

## Code artifacts produced (2026-06-01)
- `Code/analyzer/stats.py` — bootstrap CIs, ICC(3,1)/ICC(3,k), Benjamini-Hochberg (validated vs Shrout & Fleiss reference).
- `Code/analyzer/ensemble_ablation.py`, `verbosity_bias.py`, `rubric_generalization.py` — EXP-002/004/005 compute + self-contained HTML.
- `Code/analyzer/experiment_reports.py` — shared no-CDN Plotly page helper.
- `Public/benchmark/score_responses.py` — per-response benchmark-native scorer (fixes the reference-vs-itself integrity bug for EXP-001).
- CLI: `coeval analyze ensemble-ablation | verbosity-bias | rubric-overlap`.
- Tests: `Tests/analyzer/test_experiments.py` (23), `Tests/benchmark/test_score_responses.py` (9). Zero regressions (1024 prior tests still pass).

## Known pre-existing issue (not introduced here)
- `Tests/analyzer/test_analyze_reports.py::TestExportBenchmark` (7 tests) fail on a clean tree: robust-filter `theta` default drift (test expects `theta=0.0` behavior, code default is 0.05 producing 0 robust datapoints). Orthogonal to this work.

## Cycle-3 additions (2026-06-01)
- **G-Eval baseline (R9):** single gpt-4o+CoT on 600 summarization responses → rho=0.259 [0.18,0.33] vs BERTScore (CoEval ensemble 0.244; best single 0.354). `Runs/EXP001-.../reports/geval_baseline.json`, `scripts/geval_baseline.py`.
- **Model-ranking demo:** CoEval frontier ensemble reproduces exact ground-truth ranking (gpt-4o-mini 0.963 > gpt-3.5 0.921 > llama-3b 0.843; GT identical). `Runs/EXP001b-.../reports/model_ranking.json`. Paper Table 2.
