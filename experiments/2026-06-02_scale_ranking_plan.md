# Plan — large-panel / many-model scaling (answers "should we go >10 models?")

**Status:** planned · **Date:** 2026-06-02 · **Cost est:** ~$15-20 (OpenRouter cheap tier, no GPU)

## The question, split into two axes
"More than 10 models" means two different things; the paper's own thesis answers them differently.

### Axis 1 — JUDGE panel size (>10 judges)
The central finding is **composition > size**: ICC(3,k) is non-monotone, peaking at a
well-chosen pair and FALLING as low-agreement judges are added. So a >10-judge panel is
NOT motivated as an operating point. It IS worth running once, as a **selection-at-scale**
demonstration: from a pool of 12-15 candidate judges spanning many vendors, show CoEval's
consensus selection picks the 2-3 that maximize ICC, and that the full pool is strictly
worse. This maps the full ICC(k) curve at high k and validates the selection mechanism,
turning the composition claim from "observed on a small panel" into "holds across a large
candidate pool."
- Metric: ICC(3,k) vs k (full curve to k=12+); ICC of selected subset vs full pool.
- Win: selected small panel ICC > full-pool ICC; curve matches Spearman-Brown prediction.

### Axis 2 — CANDIDATE (student) models being ranked (>10 students)
**This is the clearly valuable one and directly retires a known reviewer residual**
("ranking demo is 3 models / 1 task family"). Ranking 3 models makes rank-recovery trivial
(any monotone scorer passes). Ranking 12-20 models gives real rank-order resolution.
- Protocol: pick a verifiable task family where a gold accuracy ranking exists
  (SciQ/ARC exact-match, or an MMLU subset). Generate fresh CoEval items. Run 12-20
  students spanning the capability range (e.g. gpt-4o, gpt-4o-mini, gpt-3.5-turbo,
  claude-3.5-haiku, claude-3.5-sonnet, gemini-2.5-flash, llama-3.3-70b, llama-3.1-8b,
  llama-3.2-3b, qwen2.5-7b, qwen2.5-1.5b, mistral-7b, ...). Score with the vendor-disjoint
  panel + reliability-weighted aggregation.
- Metric: Spearman / Kendall-tau between CoEval label-free ranking and gold accuracy
  ranking over all N models; bootstrap CI on tau; compare to best single judge + plain mean.
- Win: CoEval rank-recovery tau >= 0.8 over 12-20 models, strictly above the best single
  judge and plain mean. This is the scaled version of scout Exp A and converts the
  "only 3 models" scope note into a headline scaling result.

## Cost equation
N_students(15) x N_items(100) = 1500 responses (~$1-2 cheap tier)
1500 x N_judges(6) x aspects(4) ~= 36k judge calls (~$7); 12 judges ~= $14.
Generation one-time ~$0.5. Total ~$10-20. Fully parallelizable (AsyncOpenAI / OpenRouter),
no GPU. Use OpenAI Batch-equivalent where available to halve cost.

## Batch-interface cost reduction (use the framework's native batch support)
The dominant cost is the judging step (judges x responses x rubric aspects), which is an
embarrassingly batchable, latency-tolerant offline workload, exactly what provider Batch
APIs are for (~50% discount, <=24 h turnaround). CoEval already supports batch per provider
(`interface: openai|anthropic|gemini|vertex|azure_openai|bedrock|mistral`; batch discounts
live in `Config/provider_pricing.yaml`, surfaced by `coeval plan`).

Routing for this experiment:
- **Judging (bulk cost): route the panel through batch-capable native interfaces.** Put the
  OpenAI judge on `interface: openai` (OpenAI Batch, 50% off), the Anthropic judge on
  `interface: anthropic` (Message Batches, 50% off), optionally a Mistral judge on
  `interface: mistral` (Batch, 50% off). NOTE: the `gemini` interface is a concurrent
  pseudo-batch (thread pool, NO async discount per `provider_pricing.yaml`); for a true 50%
  discount on a Google judge use `interface: vertex` (Vertex Batch Prediction) instead.
- **Student generation (smaller cost): also batch the OpenAI/Anthropic/Gemini students;**
  the open-weight students (llama/qwen/mistral-small) can stay on a cheap real-time route
  (OpenRouter/Groq) since those have no batch tier and the per-call price is already tiny.
- **Non-batch providers** (openrouter, groq, deepseek) stay real-time at cheap-tier price;
  do NOT force them through a batch path they do not have.
- **Pre-flight with `coeval plan --config scale.yaml`** to read the batch-vs-realtime cost
  table BEFORE launch (budget-aware planning), then `coeval run` writes the batch jobs and
  `coeval status` fetches results.

Revised cost equation:
real-time ~= $15-20  ->  batch-routed judging+generation ~= **$7-11** (judging halved is the
big lever; the open-weight real-time tail is already cheap). Turnaround <= 24 h, acceptable
for an offline strengthening experiment; if a same-day result is needed, run real-time and
pay the ~2x. Arm a spend watchdog and `coeval status` polling rather than blocking.

## Recommendation
- **Axis 2 (rank 12-20 candidate models on a verifiable task): YES, run it.** Highest-value
  next experiment; retires the 3-model residual and answers the scaling question with a
  rank-recovery number.
- **Axis 1 (12+ judge pool): run as a selection-at-scale demo,** reusing the same job's
  judge logs; do NOT propose a large judge panel as the operating point (the thesis says a
  selected small panel is more reliable).
- Both run from one config through the actual framework (real pipeline, not paper-only code).
- **Route batch-capable judges/students through native Batch interfaces (~50% off the bulk
  judging cost); pre-flight with `coeval plan`.** Batch-routed cost ~$7-11 vs ~$15-20 real-time.
Awaiting go to launch.
