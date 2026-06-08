# Course Project Report — Diagnosing and Fixing a Benchmark-Generation Diversity Bug

**Domain:** Cybersecurity · **Task:** Phishing-email classification (Phishing / Suspicious / Legitimate)
**Base system:** [CoEval](https://github.com/ApartsinProjects/CoEval) by Alexander Apartsin & Yehudit Aperstein. This report covers **only my own contributions** (experiments, debugging, and fixes) on top of that system.

---

## 1. Goal

Use CoEval to build a small **model-evaluation benchmark** for a custom domain (phishing detection) and rank several LLMs on it. A *teacher* model generates email descriptions with controlled attributes; *student* models classify each email; a deterministic *judge* (`exact_match`) scores them.

The assignment's central question is to separate two very different failure modes:

| Failure | Acceptable? |
|---|---|
| A weak student answers wrong because it is not capable | ✅ Yes — expected |
| The **benchmark itself is generated incorrectly** (bad / low-diversity items) | ❌ No — a bug to find and fix |

The hypothesis given was that the generated benchmark suffers from **low diversity**, that this is **not** the LLM's fault, and that the root cause is a **sampling bug** somewhere in the pipeline. This report confirms that hypothesis, locates the root cause in code, fixes it, and shows a clean before/after.

All models were run through **OpenRouter** using only `:free` endpoints, so every experiment cost **USD 0**.

---

## 2. Setup

| Role | Model (OpenRouter `:free`) |
|---|---|
| Teacher (generates emails) | `openai/gpt-oss-20b:free` |
| Students (classify) | `liquid/lfm-2.5-1.2b-instruct:free`, `nvidia/nemotron-nano-9b-v2:free`, `z-ai/glm-4.5-air:free` |
| Judge | `exact_match` (deterministic metric, no LLM, no bias) |

**Task attribute space** (`docs/examples/phishing_detection.yaml` → my run configs in `Runs/_guy_configs/`):

```yaml
target_attributes:
  label:        [Phishing, Suspicious, Legitimate]   # the ground-truth class
  sender_type:  [bank, unknown]
  quality:      [professional, typos]
# 3 x 2 x 2 = 12 possible combinations
```

---

## 3. Root cause of the diversity bug

The bug is in the Phase-3 (data-generation) sampler, `Code/runner/phases/phase3.py`:

```python
# phase3.py ~236
if task.sampling.target == 'all' and target_attrs:
    target_sequence = _make_target_cycle(...)   # full Cartesian coverage of all combos
else:
    target_sequence = None
    ...
    sampled_target = _sample_attrs(target_attrs, task.sampling.target)  # per-item random subset
```

The **diversity-preserving path** (`_make_target_cycle`, which guarantees every attribute combination is covered) only runs when `sampling.target == 'all'`. For any numeric spec it falls through to `_sample_attrs`, which selects a *random subset* of attribute **names** per item (`phase3.py:573`):

```python
n = randint(lo, hi)
selected_names = random.sample(attr_names, n)   # picks n of the attribute NAMES
```

My original task config used `sampling.target: [1, 1]`, i.e. **pick exactly 1 of the 3 target attributes per email**. The consequences:

1. **The `label` ground-truth is dropped on most items** — when the single sampled attribute happens to be `sender_type` or `quality`, the email is generated with *no controlled class*, so the teacher invents one.
2. **No coverage guarantee** — combinations repeat and many are never produced → low diversity.

This matches the hypothesis exactly: the problem is in **sampling**, not the LLM, and it corrupts **benchmark generation** rather than reflecting student weakness.

---

## 4. Empirical confirmation — BEFORE the fix

Run `EXP-guy-01c-phishing-buggy` (`sampling.target: [1,1]`, 4 emails). The `sampled_target_attributes` of the generated emails:

| # | sampled attributes | has `label`? |
|---|---|---|
| 1 | `{sender_type: unknown}` | ❌ no |
| 2 | `{label: Phishing}` | ✅ |
| 3 | `{label: Suspicious}` | ✅ |
| 4 | `{sender_type: bank}` | ❌ no |

- **Every** email received exactly **1 of 3** attributes.
- **2 of 4 (50%)** had **no controlled label** — the teacher had to invent the ground-truth class.
- Secondary generation defect: a raw attribute token leaked verbatim into an email body
  (*"you've been selected as a **prize_winner**"*), showing the teacher echoing internal attribute names.

Student accuracy in this run (reported with a caveat, see §7):

| Student | exact_match accuracy |
|---|---|
| `lfm-1.2b` | 0.50 (2/4 — but it answered "Phishing" to *everything*) |
| `nemotron-9b` | 0.00 |
| `glm-4.5-air` | 0.00 |

---

## 5. The fix and AFTER results

The fix is to drive Phase-3 sampling through the full-coverage path. Setting:

```yaml
sampling:
  target: all     # full Cartesian coverage (was: [1, 1])
  total: 12       # = 3 x 2 x 2 -> every combination produced exactly once
```

Run `EXP-guy-02-phishing-fixed` then generated **all 12 distinct combinations, each exactly once, every email fully specified**:

| # | label | sender_type | quality | attrs |
|---|---|---|---|---|
| 1 | Phishing | bank | professional | 3/3 |
| 2 | Phishing | bank | typos | 3/3 |
| 3 | Phishing | unknown | professional | 3/3 |
| 4 | Phishing | unknown | typos | 3/3 |
| 5 | Suspicious | bank | professional | 3/3 |
| 6 | Suspicious | bank | typos | 3/3 |
| 7 | Suspicious | unknown | professional | 3/3 |
| 8 | Suspicious | unknown | typos | 3/3 |
| 9 | Legitimate | bank | professional | 3/3 |
| 10 | Legitimate | bank | typos | 3/3 |
| 11 | Legitimate | unknown | professional | 3/3 |
| 12 | Legitimate | unknown | typos | 3/3 |

### Before vs After

| Metric | Before (`[1,1]`) | After (`all`) |
|---|---|---|
| Attributes controlled per email | 1 / 3 | **3 / 3** |
| Emails with no `label` ground-truth | 50% | **0%** |
| Distinct combinations covered | random, repeating | **12 / 12** |
| Class balance | uncontrolled | **4 Phishing / 4 Suspicious / 4 Legitimate** |

The diversity bug is fully resolved: the benchmark is now balanced, fully labeled, and maximally diverse.

---

## 6. Fixing the root cause in the system code (not just the config)

§5 fixes the bug at the *config* level (`sampling.target: all`). But the defect
lives in the **system code**, and a config workaround leaves the same trap in
place for every other task and user. So I also fixed it in `phase3.py` itself.

**What I saw.** With `sampling.target: [1, 1]`, generated items were missing the
`label` ground truth ~50% of the time, and the attribute combinations were not
covered (§4).

**What I found.** In `Code/runner/phases/phase3.py`, target-attribute assignment
used the full-coverage routine `_make_target_cycle` **only** when
`sampling.target == 'all'`. For any numeric spec it fell through to
`_sample_attrs`, which selects a random *subset of target attribute names* per
item. Per-attribute subsetting is correct for *nuanced* attributes, but for
*target* attributes it silently drops the structured ground truth (including the
class label) and destroys diversity. The two generation paths (batch and
streaming) both had this branch.

**What I fixed.** I routed all target-attribute assignment through a new helper
`_build_target_sequence`, which always uses `_make_target_cycle` (full
combination coverage) for target attributes regardless of the spec, and logs a
clear warning when a legacy numeric `sampling.target` is encountered.
`_sample_attrs` now serves nuanced attributes only. Both call sites updated.

**How I verified.**
- *Regression:* the full runner test suite still passes — **778 passed**; the only
  2 failures are pre-existing and unrelated (a missing optional `google-genai`
  package), identical before and after my change.
- *Direct proof:* calling the sampler with the **old buggy `[1, 1]` spec** now
  yields **12/12 items with all 3 attributes, 0 missing the `label`, and 12/12
  distinct combinations** — the bug can no longer occur, even from a config that
  was previously broken.

---

## 7. Second finding — the generated content does not match its label

Fixing the sampler fixed *coverage* (every label is now represented). But a blind
read of the 12 generated emails from `EXP-guy-03` reveals a **second, deeper
benchmark-generation problem**: the email *content* does not reflect its assigned
label. A careful human reader classifies essentially **all 12 emails as Phishing**,
regardless of whether the teacher labeled them Phishing, Suspicious, or Legitimate.

Examples where the gold label is simply wrong on its face:

| # | Gold label | Email content | Honest read |
|---|------------|---------------|-------------|
| 7, 8 | Suspicious | "Congratulations! claim your $5,000 gift card … submit your bank details" | Phishing (prize scam) |
| 9 | **Legitimate** | "account suspended … verify within 24 hours or permanent closure: https://secure.bankoftrust.com/verify" | **Phishing** (textbook) |
| 11 | **Legitimate** | "Congratulations! winner of our contest … claim your $5,000 gift card" | Phishing (prize scam) |

**Why this happens:** the teacher is given the class as just one attribute among
several, plus a single in-prompt example that is itself **Phishing**
(`prompt_library.sample` shows only `{"response": "Phishing"}`). The teacher anchors
on that exemplar and writes phishing-style content (urgency, fake verify links,
prize bait) no matter which label it was asked for. So the "Suspicious" and
"Legitimate" items are indistinguishable from the "Phishing" ones.

**Consequence for the evaluation:** the students that answered "Phishing" to
everything are arguably **more correct than the gold labels** on items 5–9 and 11.
Their ~0.33 accuracy therefore reflects **unreliable ground truth**, not only model
weakness. This is the assignment's core distinction in its sharpest form: the
failure is in **benchmark generation**, not in the students.

**The fix (applied in `EXP-guy-04`, a YAML-only prompt change):** rewrite
`prompt_library.sample` to (a) make the class label the primary instruction,
(b) define each class, (c) give one example *per* label instead of a single
Phishing exemplar, and (d) replace the scam-flavoured nuances
(`prize_winner` / `threatening`) with neutral ones (`order_confirmation` /
`account_update`). No code change is needed — the prompt lives entirely in the
task YAML.

**Result — the fix works, and it proves the point.** With label-matching content,
a blind reader can now tell the classes apart (Legitimate items are real
statements / receipts with no links; Suspicious items have mild red flags only),
and **student accuracy roughly doubled**:

| Student | Run 03 (old prompt) | Run 04 (fixed prompt) |
|---------|--------------------:|----------------------:|
| `gemma-4-26b`  | 0.42 | **0.83** |
| `gpt-oss-120b` | 0.33 (all-Phishing) | **0.67** |
| `lfm-1.2b`     | 0.33 | **0.67** |

The models were never as weak as Run 03 suggested: the ground truth was broken.
Once generation is correct, the same models score far higher, confirming the low
Run 03 scores were a **benchmark-generation** failure, not model weakness. (The
one genuinely hard class that remains is "Suspicious" — that residual is real
model limitation on a now-correct benchmark.)

---

## 8. Fix log (changelog of my changes)

Two kinds of change: **(A) genuine system-code bug fixes**, and **(B)** task
config + infrastructure.

**(A) System-code bug fixes**

| # | File | Change | Why |
|---|---|---|---|
| 1 | `Code/runner/phases/phase3.py` | **The diversity bug, fixed in code.** Target attributes now always go through full-coverage cycling (new `_build_target_sequence`); a numeric `sampling.target` no longer drops attributes. `_sample_attrs` is now nuanced-only. | The full-coverage path ran only for `target: all`; a numeric spec silently dropped the ground-truth `label` and killed diversity (§3, §6). Verified: 778 tests pass; `[1,1]` now gives 12/12 full, labeled, distinct items |
| 2 | `Code/runner/interfaces/probe.py` | Register the virtual `metric` interface (return early) | Without it a `metric` judge fell through to the HuggingFace probe and was wrongly loaded as a HF model → probe failure |

**(B) Task config + infrastructure (to run on free models)**

| # | File | Change | Why |
|---|---|---|---|
| 3 | `Runs/_guy_configs/*.yaml` | `sampling.target: all`; rewritten teacher prompt (per-class definitions + one example per class); neutral nuances | Config-level fixes for finding #1 (coverage) and finding #2 (content matches label) |
| 4 | `Code/runner/interfaces/openrouter_iface.py` | Recognise HTTP `429` / `rate-limited`; honour `retry_after`; 6 patient retries; fail fast on the daily free-quota cap | Free `:free` models share a pool that 429s with ~30 s cooldowns; the old 1–2 s backoff skipped most datapoints |
| 5 | `Code/runner/phases/phase4.py`, `phase5.py` | `_MAX_WORKERS` reads `COEVAL_MAX_WORKERS` env var (default 10) | Drop concurrency to 2 so parallel calls don't instantly trip the free-tier rate limit |

---

## 9. Limitations and honest notes

- **Free-tier rate limits dominated the engineering effort.** OpenRouter `:free` endpoints share an upstream pool. The popular `meta-llama/llama-3.3-70b-instruct:free` was so saturated that a teacher run there failed 7/8 items; switching the teacher to the responsive `openai/gpt-oss-20b:free` fixed it. There is also a hard **50 free requests/day** account cap, which truncated the student phase of the fixed run. None of this affects the headline result, which lives entirely in Phase-3 generation.
- **The student ranking is a confound, not a capability measure.** Two of the three students (`nemotron-9b`, `glm-4.5-air`) are *reasoning* models; at `max_tokens: 16` they emit chain-of-thought ("Hmm, the user has given me…") instead of a bare label, and `exact_match` (which requires full string equality, `metric_judge.py:136`) scores that 0. This is a student-side evaluation artifact, **not** the benchmark-generation bug, and per the assignment's framing the focus stayed on generation.

---

## 10. How to reproduce

```bash
# 1. Put an OpenRouter key in keys.yaml (git-ignored):
#    providers:
#      openrouter: sk-or-v1-...

# 2. Buggy run (before):
COEVAL_MAX_WORKERS=2 python -m runner.cli run --config Runs/_guy_configs/01a_buggy.yaml

# 3. Fixed run (after):
COEVAL_MAX_WORKERS=2 python -m runner.cli run --config Runs/_guy_configs/02_fixed.yaml

# Inspect the generated benchmark items:
#   Runs/EXP-guy-01c-phishing-buggy/phase3_datapoints/*.jsonl   (buggy: 1 attr each)
#   Runs/EXP-guy-02-phishing-fixed/phase3_datapoints/*.jsonl    (fixed: all 12 combos)
```

> Note: with the 50-requests/day free cap, run one experiment at a time, or add credit to OpenRouter to raise the cap to 1000/day.
