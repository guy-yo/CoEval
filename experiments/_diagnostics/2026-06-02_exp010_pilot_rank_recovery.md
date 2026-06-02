# DIAGNOSTIC — EXP-010 pilot (many-model rank-recovery)

**Status:** diagnostic (NOT paper material) · **Date:** 2026-06-02 · **Cost:** ~$0.6 (pilot)
**Purpose:** de-risk the scaled rank-recovery run before spending on 12-20 models.

## What ran
8-model student roster (gpt-4o, gpt-4o-mini, claude-3.5-sonnet, claude-3.5-haiku,
gemini-2.5-flash, llama-3.3-70b, llama-3.1-8b, llama-3.2-3b) + 3-judge cross-family panel
(gpt-4o-mini, claude-3.5-haiku, gemini-2.5-flash), on 40 reused gold sciq science-QA items.
Reused EXP001b ingested items; ran phases 4-5 via the real framework.

## What the pilot PROVED (mechanics)
- The multi-model pipeline runs end-to-end through the actual framework: 7/8 students
  collected, all judged, gold-scored, rank-recovery analysis runs. (`scripts/v2_rank_recovery.py`)

## Issues surfaced and triaged (Research-Honesty bug-hunt)
1. **Bad model slug:** `anthropic/claude-3.5-sonnet` 404s on OpenRouter ("No endpoints").
   FIX for scaled run: use `anthropic/claude-sonnet-4` (verified in EXP001b) or current slug.
2. **OpenRouter rate-limit truncation:** a single `coeval run` left phase-5 judging at ~118/280
   responses (exit 0 but incomplete). A second `--continue` (Extend mode) completed coverage
   to 7x3x~240 units. FIX for scaled run: route judges through native **Batch APIs**
   (openai/anthropic/mistral, 50% off + no real-time rate cap), per the scale plan. This is
   the concrete reason the batch interface matters operationally, not just for cost.
3. **Gold metric is output-format sensitive (FIXED):** raw `inclusion_match` scored
   claude-3.5-haiku at 0.05 because it answered with the option LETTER ("C") while the gold
   reference is the answer TEXT ("fossilization"). Built a format-robust MCQ scorer
   (`scripts/v2_gold_rescorer.py`: parses the option block, resolves the correct letter, credits
   a response that names the correct option by EITHER letter or text). After the fix, haiku
   gold = 0.95 (sensible). This is the canonical exact-match/format-mismatch trap.
4. **Task saturation (DESIGN BLOCKER for the result):** after the format fix, all 7 models
   score 0.95-1.00 on sciq (gold range 0.05). sciq science-QA is too easy to discriminate
   modern models, so rank-recovery is ill-defined (near-ties). NO rank-recovery number from
   this pilot is reportable.

## Refined design for the clean scaled run (carry forward)
- **Harder, discriminating task** so gold spreads models: ARC-Challenge is borderline; prefer
  logiqa / bigbench_hard / mathqa / math (or MMLU-Pro / GPQA via ingest) where small models
  fall well below frontier. Mix 2-3 tasks for robustness.
- **Format-robust gold scorer** (done) for every MCQ task.
- **Batch-routed judging** (openai/anthropic/mistral native Batch) to avoid the rate-limit
  truncation and halve cost; pre-flight with `coeval plan`.
- **12-20 models** spanning frontier -> tiny for rank-order resolution; fix the sonnet slug.
- Metric: Spearman/Kendall(CoEval ensemble ranking, gold accuracy ranking); compare to
  best/worst single judge (judge-choice regret in ranking terms) and plain mean.

## Artifacts
`Runs/EXP010-scale-ranking-pilot/` (config, phases, reports/v2_rank_recovery.json),
`scripts/v2_rank_recovery.py`, `scripts/v2_gold_rescorer.py`.
