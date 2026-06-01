# Review Cycle 3 — TMLR-style (2026-06-01)

**Recommendation: Major Revision, borderline Minor.** Keystone integrity fully resolved & verified (R1/R2/W1 etc.). Remaining gaps "specific and cheaply closable" — addressed this round:

## P0 (addressed)
- **R9 external baseline** — ran **G-Eval** (single gpt-4o + CoT) on the 600 summarization responses: rho=0.259 [0.18,0.33] vs BERTScore, comparable to CoEval ensemble (0.244); integrated in 5.1 with the honest framing (CoEval adds bias-robustness + item generation, not point-correlation). Artifact: reports/geval_baseline.json.
- **W2 generality overreach** — scoped "any task/domain"→"target task; across the task families studied"; ADDED a concrete ranking demonstration (Table 2): CoEval ensemble reproduces the EXACT ground-truth model ranking (gpt-4o-mini>gpt-3.5>llama-3b), scores within 0.02 of truth. Artifact: reports/model_ranking.json.
- **W3 verbosity family/capability confound** — 5.3 now states the exact panel (gpt-4o-mini+gpt-3.5-turbo OpenAI; qwen2.5-1.5b+smollm2-1.7b open-weight) and re-attributes cancellation to bias-SIGN diversity (vendor diversity as a proxy), with the symmetric caveat matching 5.4.

## P1 (addressed)
- **W4 numeric drift** — 5.1 now 0.244 (was 0.241); code subtask rho=-0.031 (was "≈0").
- **W6 BH** — §4 now states Benjamini-Hochberg FDR control + datapoint-clustered bootstrap.
- **W5 stratification** — §3.2 marks per-stratum floor as a sampler guarantee / system feature (coverage in A3), not separately validated.

## P2 (open, non-blocking)
- R7 empirical contamination check on generated items; W7 broaden objective anchor beyond 2 MCQ-science sets; abstract↔§5 register polish.

## Verified
G-Eval rho=0.259 (n=580); ranking identical to ground truth; tables 1/2/3 consistent; 26 refs resolve; docx regenerated (0 unconverted $).
