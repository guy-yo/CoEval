# Experiment Log — Results of All Runs

A living record of every experiment run for this project (phishing-detection
benchmark). Regenerate the metrics table at any time with:

```bash
python scripts/summarize_runs.py
```

The narrative analysis (root cause, fix, before/after) is in
[REPORT.md](REPORT.md); this file is the **results ledger**.

---

## Master results table

| Run | Sampling | Purpose | Status | Datapoints | Full attrs (3/3) | Distinct combos |
|-----|----------|---------|--------|-----------:|-----------------:|----------------:|
| `EXP-guy-01a-phishing-buggy` | `target: [1,1]`, total 8 | First buggy run (5 students) | failed¹ | 1 | 0 | 1 |
| `EXP-guy-01b-phishing-buggy` | `target: [1,1]`, total 4 | Retry with patient backoff | killed² | 0 | 0 | 0 |
| `EXP-guy-01c-phishing-buggy` | `target: [1,1]`, total 4 | **BEFORE** — reproduce the bug | completed | 4 | **0** | 4 |
| `EXP-guy-02-phishing-fixed`  | `target: all`, total 12 | **AFTER** — the fix | failed³ | 12 | **12** | **12** |
| `EXP-guy-03-ranking`         | `target: all`, total 12 | Model ranking (clean students) | completed | 12 | **12** | **12** |
| `EXP-guy-04-improved-prompt` | `target: all`, total 12 | **Improved teacher prompt** (finding #2 fix) | completed | 12 | **12** | **12** |
| `EXP-guy-07-proper-free`     | `target: all`, total 12 | Broader 4-model free ranking | completed | 12 | **12** | **12** |
| `EXP-guy-08-big-mixed`       | `target: all`, total 24 | **Stable ranking, free + paid frontier** | completed | 24 | **24** | **24** |

¹ Free-tier 429 storm on `llama-3.3-70b:free` teacher — 7/8 datapoints skipped.
² `llama-3.3-70b:free` persistently rate-limited upstream; killed and switched teacher to `gpt-oss-20b:free`.
³ Phase-3 generation completed (the headline result); Phase-4 students partially failed because the OpenRouter **50 requests/day** free cap was exhausted. Does not affect the generation result.

### Headline result

The fix moves benchmark generation from **0/4 fully-specified emails** (buggy) to
**12/12 fully-specified emails covering all 12 attribute combinations** (fixed):

| | BEFORE (`[1,1]`) | AFTER (`all`) |
|---|---|---|
| Attributes controlled per email | 1 of 3 | **3 of 3** |
| Emails missing the `label` ground-truth | 50% | **0%** |
| Distinct (label × sender_type × quality) combos | 4 (random, repeating) | **12 / 12** |
| Class balance | uncontrolled | **4 Phishing / 4 Suspicious / 4 Legitimate** |

---

## Per-student classification accuracy (judge = `exact_match`)

> ⚠️ Caveat: `nemotron-9b` and `glm-4.5-air` are *reasoning* models; capped at
> `max_tokens: 16` they emit chain-of-thought instead of a bare label, which
> `exact_match` scores 0. This is a student-output artifact, not capability.
> See [REPORT.md §7](REPORT.md).

| Run | lfm-1.2b | nemotron-9b | glm-4.5-air |
|-----|---------:|------------:|------------:|
| `EXP-guy-01c` (buggy)  | 0.50 (2/4)  | 0.00 (0/4) | 0.00 (0/4) |
| `EXP-guy-02` (fixed)   | 0.33 (4/12) | 0.00 (0/5) | 0.00 (0/8) |

`nemotron-9b` and `glm-4.5-air` are reasoning models truncated at `max_tokens: 16`,
so their 0.00 is an output artifact, not capability (see caveat above).

---

## Model ranking on the fixed benchmark (`EXP-guy-03`)

This run swaps in **non-reasoning instruct students** (they emit a bare label),
so the scores are a real ranking. Gold labels are perfectly balanced by the fix:
**4 Phishing / 4 Suspicious / 4 Legitimate**.

| Rank | Student | Accuracy | Answer distribution (P / S / L) |
|-----:|---------|---------:|---------------------------------|
| 1 | `gemma-4-26b`  | **0.42** (5/12) | 10 / 1 / 1 |
| 2 | `gpt-oss-120b` | 0.33 (4/12) | **12 / 0 / 0** |
| 2 | `lfm-1.2b`     | 0.33 (4/12) | 7 / 0 / 5 |

**Findings:**

- **All three models are strongly biased toward "Phishing"** ("cry wolf") and
  cluster near the 0.33 majority-class baseline.
- **None reliably detects "Suspicious"** — `gpt-oss-120b` and `lfm-1.2b` never
  predict it at all.
- **Bigger is not better here:** `gpt-oss-120b` (120 B) is a *degenerate*
  classifier — it answers "Phishing" to all 12 items — while the smaller
  `gemma-4-26b` (26 B) is the best of the three.
- Because the benchmark is now correct (balanced, fully labeled, diverse), these
  low scores are **genuine model weakness, not a sampling bug** — exactly the
  distinction this project set out to make.

### Second finding: the email content doesn't match its label

A blind read of the 12 generated emails shows a deeper benchmark-generation
problem: a careful human reader would classify **all 12 as Phishing**, regardless
of the teacher's assigned label. The "Suspicious" items are prize/credential scams,
and even two "Legitimate" items are textbook phishing:

| # | Gold | Content | Honest read |
|---|------|---------|-------------|
| 9 | **Legitimate** | "account suspended … verify in 24h or permanent closure + link" | **Phishing** |
| 11 | **Legitimate** | "you won a $5,000 gift card … confirm your address" | Phishing (scam) |
| 7, 8 | Suspicious | "$5,000 gift card … submit bank details" | Phishing (scam) |

**Cause:** `prompt_library.sample` gives the teacher only one example, and it is
*Phishing*, so it writes phishing-style content for every label. The students that
answered "Phishing" to everything are arguably **more correct than the gold labels**
on items 5–9 and 11 — so the ~0.33 accuracy partly reflects **bad ground truth**,
not model weakness. (See [REPORT.md §6](REPORT.md).)

### Fixing finding #2 worked — and proves the point (`EXP-guy-04`)

The fix is a **YAML-only prompt change**: the class label becomes the primary
instruction, each class is defined, one example is given *per* class (not just a
Phishing one), and the scam-flavoured nuances are replaced with neutral ones.

Now the generated content matches the labels (Legitimate = real statements /
receipts with no links; Suspicious = mild red flags only), and **student
accuracy roughly doubled** on the same balanced 4/4/4 benchmark:

| Student | Run 03 (old prompt) | Run 04 (fixed prompt) | Answer mix (Run 04) |
|---------|--------------------:|----------------------:|---------------------|
| `gemma-4-26b`  | 0.42 | **0.83** | P 6 / S 2 / L 4 |
| `gpt-oss-120b` | 0.33 | **0.67** | P 8 / S 0 / L 4 |
| `lfm-1.2b`     | 0.33 | **0.67** | P 6 / S 0 / L 6 |

**Takeaway:** the models were not as weak as Run 03 implied — the ground truth was
broken. Fixing benchmark generation roughly doubled accuracy. The only class still
missed is "Suspicious" (the two larger models never predict it), which is now a
**genuine model limitation** on a correct benchmark — exactly the failure type the
assignment treats as acceptable.

### Broader free ranking + a note on variance (`EXP-guy-07`)

Four reliable free students (1.2B → 31B) on a fresh balanced benchmark:

| Rank | Student | Accuracy | Phishing | Suspicious | Legitimate |
|-----:|---------|---------:|---------:|-----------:|-----------:|
| 1 | `lfm-1.2b`    | 0.67 (8/12) | 4/4 | 0/4 | 4/4 |
| 2 | `gpt-oss-20b` | 0.58 (7/12) | 4/4 | 0/4 | 3/4 |
| 2 | `gemma-4-31b` | 0.58 (7/12) | 4/4 | 1/4 | 2/4 |
| 2 | `gemma-4-26b` | 0.58 (7/12) | 4/4 | 0/4 | 3/4 |

**Two honest conclusions:**

1. **The robust signal is per-class, not the fine ranking.** Every model solves
   Phishing (4/4) and essentially none solves Suspicious (~0/4); the scores cluster
   at 0.58–0.67. Model size does not predict accuracy here (the 1.2B model "wins"
   only because it never guesses Suspicious and happens to split Phishing/Legitimate
   well — luck, not skill).
2. **Small benchmarks are noisy.** `gemma-4-26b` scored 0.83 in Run 04 but 0.58
   here — each run regenerates a different 12-item set, so 12 items give high
   variance. A stable fine-grained ranking needs many more items (now feasible:
   the daily free cap rose to 1000/day after the account added credit). The
   trustworthy takeaway from these runs is the **per-class pattern**, not the exact
   order within the 0.58–0.67 cluster.

### Bigger, stable ranking: free vs paid frontier (`EXP-guy-08`)

24 items (8 per class) for stability; six students, three reliable free + three
cheap paid frontier models. **Actual cost: USD 0.0057** (free daily cap is 1000/day).

| Rank | Student | Tier | Accuracy | Phishing | Suspicious | Legitimate |
|-----:|---------|------|---------:|---------:|-----------:|-----------:|
| 1 | `gemini-2.5-flash-lite` | paid | **0.88** (21/24) | 8/8 | **5/8** | 8/8 |
| 2 | `gpt-4o-mini`   | paid | 0.79 (19/24) | 8/8 | 3/8 | 8/8 |
| 2 | `gemma-4-26b`   | free | 0.79 (19/24) | 8/8 | 3/8 | 8/8 |
| 4 | `gpt-oss-20b`   | free | 0.71 (17/24) | 8/8 | 1/8 | 8/8 |
| 5 | `lfm-1.2b`      | free | 0.67 (16/24) | 8/8 | 0/8 | 8/8 |
| 5 | `claude-3.5-haiku` | paid | 0.67 (16/24) | 8/8 | 0/8 | 8/8 |

**Findings (now stable, 24 items):**

1. **Every model solves Phishing (8/8) and Legitimate (8/8).** The benchmark's clear
   classes are a sanity floor that all models pass — confirming the items are
   well-formed — so **the entire ranking is decided by the "Suspicious" class.**
2. **`gemini-2.5-flash-lite` wins (0.88)** — the only model that handles the
   ambiguous Suspicious class reasonably (5/8).
3. **Bigger / paid is not automatically better.** `claude-3.5-haiku` (paid frontier)
   landed **last** (0/8 Suspicious), tied with the tiny free `lfm-1.2b`; and the
   **free** `gemma-4-26b` matched paid `gpt-4o-mini`. Capability on *this* task is
   not predicted by price or size.
4. Going from 12 → 24 items removed the earlier ranking noise: Legitimate is now a
   clean 8/8 for everyone, and the Suspicious gradient (5/3/3/1/0/0) is crisp.

---

## How to add a new run

1. Create a config under `Runs/_guy_configs/` with a fresh `experiment.id`.
2. Run it (free models, low concurrency):
   ```bash
   COEVAL_MAX_WORKERS=2 python -m runner.cli run --config Runs/_guy_configs/<your_config>.yaml
   ```
3. Re-run `python scripts/summarize_runs.py` and paste the refreshed table above.

> Budget reminder: OpenRouter free models are capped at **50 requests/day** unless
> the account has purchased credit (then 1000/day). Plan run sizes accordingly.
