# Review Cycle 8 — appendix streamlining (TMLR/COLM)

**Verdict: ACCEPT.**

## Changes
- Moved §5.5 Rubric generalization (prose + Figure 5 heatmap + finding) to **Appendix B**.
- Moved the §5.7 cost throughput Figure 6 to **Appendix C**, keeping the headline cost
  (USD 5.89, ~1,350 evals/dollar) in the main §5.6 Cost text with a pointer.
- Renumbered §5.6->5.5 (contamination), §5.7->5.6 (cost), §5.8->5.7 (case studies). Figures 5
  and 6 keep their numbers, now physically in the appendix (continuous numbering), so main
  Figures 1-4 and their references are untouched.
- Main results chain is now uninterrupted: 5.1 trust -> 5.2 reliability/regret -> 5.3 bias
  -> 5.4 self-preference -> 5.5 contamination -> 5.6 cost -> 5.7 case studies.

## Audits (clean)
- Subsections 5.1-5.7 contiguous (no gap); Appendices A/B/C anchors all defined + referenced.
- No lingering "Section 5.8" / stale Table-1 row labels; "USD 5.89" intact (4x).
- 6 figures present (5,6 in appendices); 6 imgs; Tables 1-5 unchanged.
- 0 em-dashes, 0 tone hits, $ balanced (344). docx rebuilt clean (164 OMML, App A/B/C present).
- Citations unchanged (32/32 bibtest-valid from cycle 7).

## Open (planned, non-blocking)
- Large-model rank-recovery experiment, now with batch-API cost routing (~$7-11 batch vs
  ~$15-20 real-time): `experiments/2026-06-02_scale_ranking_plan.md`.
