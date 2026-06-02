# Human validation study — plan (<= 3 labelers)

**Goal (addresses reviewer W1):** show that CoEval's *label-free* model ranking on a
custom, no-ground-truth vertical agrees with a human-expert ranking, and that the
cross-family ensemble agrees with humans **better than any single judge**. This turns
the paper's weakest point (verticals are unvalidated) into a direct validation.

**Status:** planned · **Tool:** `human-labeling` skill (per-rater task files, blinding,
shuffling, agreement stats) · **Constraint:** at most 3 labelers.

## Hypotheses (pre-registered)
- **H1 (ranking recovery):** the human-consensus ranking of the 3 candidate models
  equals CoEval's ensemble ranking on the chosen vertical(s).
- **H2 (ensemble > single judge):** CoEval's ensemble agrees with the human consensus
  at least as well as the *best* single judge, and strictly better than the *mean*
  single judge (Kendall tau to human ranking; per-item win-rate agreement).
- **H3 (reliability):** inter-rater agreement is acceptable (Krippendorff alpha >= 0.4
  ordinal), so the human ranking is a usable reference.

## Design
- **Verticals:** two of the three, chosen to span the regimes the paper shows:
  1. **drug-interaction (DDI)** — judges were unanimous; a clean *positive control*
     (humans should agree strongly with the ensemble).
  2. **clinical reasoning** — judges *disagreed* on the top two; the *discriminating*
     case where the ensemble-vs-single-judge question actually bites.
  (Legal can be added later; DDI+clinical is the minimal informative pair.)
- **Items per vertical:** 24 of the 40 generated items, stratified across the
  attribute cells (severity x mechanism x patient-context for DDI), to keep the rater
  load tractable: 24 items x 3 model responses = 72 responses/vertical.
- **Task format — per-item full ranking (not absolute scoring).** For each item the
  rater sees the prompt + the 3 candidate responses in **shuffled, anonymized** order
  ("Response A/B/C") and ranks them 1-2-3 against the vertical's auto-generated rubric.
  Ranking is more reliable for humans than 1-5 scoring and yields a model ordering
  directly. (Ties allowed; recorded as shared rank.)
- **Raters:** 3. For DDI/clinical, raters need basic biomedical literacy; the rubric +
  a 1-page codebook (below) guide non-specialists. Domain-expertise level is recorded
  and reported as a scope condition.
- **Blinding:** model identities hidden; response order shuffled per (item, rater) with
  a fixed seed recorded; raters never see CoEval scores.

## Rater load (feasibility)
- 2 verticals x 24 items = 48 ranking decisions per rater (each over 3 responses).
- ~1-2 min/item => ~60-90 min per rater. Comfortable for 3 raters.

## Codebook (per vertical, 1 page)
- The vertical's task description + the auto-generated rubric factors (e.g. DDI:
  interaction_accuracy, severity_correct, safety, completeness).
- Ranking rule: order the three responses best-to-worst by overall rubric fit; tie only
  when genuinely indistinguishable.
- 2 worked examples (one clear, one borderline) with the intended ranking + rationale.

## Provisioning (human-labeling skill)
1. Build the item pool: pull the 24 sampled items + the 3 model responses from
   `Runs/EXP007-ddi-vertical` and `Runs/EXP006-vertical-case-studies` phase4 files.
2. Generate 3 per-rater task files (CSV or Label Studio), each with independently
   shuffled response order and anonymized labels; store the (item -> A/B/C -> model)
   key separately.
3. Collect responses; map back to model identities.

## Analysis
- **Human consensus ranking** per vertical: rank-aggregate the 3 raters (Borda / median
  rank); bootstrap CI over items on the aggregate model scores.
- **H1:** Kendall tau and exact-match between human-consensus and CoEval-ensemble
  rankings; report per vertical.
- **H2:** for each single judge and the ensemble, compute agreement with the human
  consensus (Kendall tau on the model ranking; per-item Spearman of the response
  ranking). Tabulate ensemble vs best/mean single judge. The target result:
  ensemble >= best single judge, > mean single judge.
- **H3:** Krippendorff alpha (ordinal) and Fleiss kappa across the 3 raters per item.
- **Adjudication:** items with full 3-way rater disagreement are flagged and discussed,
  not dropped.

## Success criterion (what makes the paper shine)
A table: "CoEval ensemble ranking == human-consensus ranking on both verticals
(Kendall tau = 1.0 on DDI, >= 0.x on clinical), and the ensemble aligns with humans at
tau = A vs the mean single judge at tau = B < A." That sentence directly answers W1 and
upgrades §5.8 from "unvalidated case studies" to "human-validated rankings".

## Artifacts
- `Runs/EXP009-human-validation/` : task files, raw rater responses, key, analysis JSON,
  agreement stats. Registered in `experiments/INDEX.md`.
