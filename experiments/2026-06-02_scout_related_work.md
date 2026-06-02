# Scout — related work + new-experiment ideas (2026-06-02)

Web-researcher sweep for 2024-2026 work to strengthen the CoEval paper. **All
2026 preprint IDs below are UNVERIFIED and must pass `bibtest` (Crossref/OpenAlex)
before entering `references.bib`.** The 2024-2025 IDs are higher-confidence but
still bibtest-gated.

## Must-cite related work (verified-confidence 2024-2025)
| Paper | arXiv | Role in our story |
|-------|-------|-------------------|
| PoLL — Replacing Judges with Juries (Cohere, 2024) | 2404.18796 | canonical cross-family panel prior; baseline our vendor-disjoint panel extends |
| GSM-Symbolic (Apple, 2024, ICLR25) | 2410.05229 | closest sibling: regenerate items to defeat memorization; we generalize beyond templates |
| LiveBench (2024, ICLR25) | livebench.ai | monthly-refreshed items; we need no public stream |
| LiveCodeBench (2024) | 2403.07974 | post-cutoff windows; same camp-(ii) framing |
| Min-K%++ (EMNLP 2024) | 2404.02936 | membership-inference detection = the detect camp we complement |
| AutoBencher (2024) | 2407.08351 | difficulty/novelty/separability objectives for generated items; metric set to adopt |
| JudgeBench (2024, ICLR25) | 2410.12784 | verifiable-by-construction judge meta-eval; validation-protocol source |

## Positioning (no new experiment needed)
Two camps for "evaluate without trusting a static benchmark": (i) **detect** leakage
(Min-K%++, watermarking); (ii) **outrun** it by generating/refreshing items
(LiveBench, LiveCodeBench, GSM-Symbolic, ArenaBencher). CoEval is camp (ii) but
generalizes to **arbitrary custom task descriptions with no seed dataset**:
GSM-Symbolic perturbs an existing dataset, LiveBench harvests new public data,
CoEval generates de novo from a task spec. State this gap explicitly; our §5.6
memorizer ranking-flip is the camp-(ii) empirical payoff (the analog of
GSM-Symbolic's numbers-changed drop).

## Competitor watch (2026 preprints — UNVERIFIED, bibtest before any cite)
- BT-σ judge-aware Bradley-Terry jury (2602.16610?) — unsupervised judge-reliability
  inference; overlaps our reliability-weighted aggregation.
- CARE confounder-aware aggregation (2603.00039?) — models correlated judge errors.
If real, do NOT claim reliability-weighting as novel; frame it as a simple label-free
option and run the bake-off (below) to position against them.

## New experiments (cheap, OpenRouter, no GPU)
- **Exp B — aggregation bake-off ($0, logged data):** on EXP001 logged per-judge
  scores add Dawid-Skene + Bradley-Terry to {mean, median, trimmed, reliability-weighted}
  and compare rho-to-BERTScore + regret-vs-oracle. Defends win #2 against BT-σ/CARE.
  HIGHEST priority: zero cost, fortifies the just-added §5.2 win.
- **Exp A — known-ranking recovery (<$20):** generate fresh CoEval items for verifiable
  tasks (arithmetic/MCQ/code) where benchmark_native_score is gold; show panel +
  reliability-weighted ranking recovers gold accuracy ordering (Spearman >= 0.8) above
  best single judge + plain mean. Rebuts "no gold labels = untrustworthy" (No Free Labels).
- **Exp C — generated-item quality triptych (<$20):** on one custom task report
  AutoBencher's difficulty/novelty/separability + Min-K%++ AUROC (near chance on our
  items vs high on the static benchmark). Answers "are the generated items non-trivial?"

## Caveats from the scout
PiCO id (2402.01830) inferred, not opened. Several 2026 ids unverified. Vendor-blog
hits (futureagi) non-citable. bibtest is mandatory on every new entry.

## Scout round 2 (2026-06-02) — competitors + positioning (bibtest-verified, INTEGRATED)
Added to the paper (all bibtest VALID): label-free judge-aggregation competitors
BT-σ (2602.16610), CARE (2603.00039), Xu et al. judge-aware ranking (2601.21817);
de-novo generators CHASE (2502.14678), DataMorgana (2501.12789); contamination
survey (2502.17521, EMNLP 2025). New §2.1 paragraph + §2.2 additions; refs r27-r32.

KEY POSITIONING (now in §2.1): contribution #3 (label-free reliability-weighted
aggregation) is NO LONGER novel in isolation (BT-σ/CARE/Xu all do unsupervised
judge-reliability aggregation). Reframed as INTEGRATION: those methods assume a
fixed externally-supplied item set scored as pairwise comparisons by a single-vendor
pool; CoEval supplies the items (de-novo, contamination-free) and scores rubric-anchored
absolute scores from a vendor-disjoint panel. Owning the whole loop is what surfaces the
judge-choice-regret phenomenon and the contamination ranking-flip.

UNVERIFIED candidates held back (need bibtest if ever used): 2605.09702, 2601.22336,
2601.22548, 2512.16041, 2602.10367, 2605.19999, 2603.00077, 2603.20882, 2605.23454.
Self-preference-is-justified papers (2504.03846 already cited as r24; 2506.02592,
2601.22548 NICE-TO-HAVE) could pre-empt the "your fix removes signal" objection.
