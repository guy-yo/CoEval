# 📧 MailRank — Ranking LLMs on Email-Classification Tasks

**Author:** Guy Yogev ([@guy-yo](https://github.com/guy-yo))

MailRank builds a small, contamination-free benchmark for an **email security task**
(classifying emails as **Phishing / Suspicious / Legitimate**) and uses it to **rank
several LLMs** without any hand-labeled data. A teacher model generates labeled emails
with controlled attributes; student models classify them; a deterministic judge scores
the answers.

Along the way I found and fixed **two bugs in how the benchmark was generated** — the
more interesting result than the ranking itself: once generation was correct, model
accuracy roughly **doubled**, showing the low scores were a *benchmark* problem, not a
*model* one.

> Built on the third-party library **[CoEval](https://github.com/ApartsinProjects/CoEval)**
> (the teacher→student→judge engine). MailRank is my project *on top of* CoEval — the
> task design, the benchmark, the experiments, the bug-hunt and the fixes are mine. See
> the CoEval repo for what the underlying framework does; this README does not repeat it.

📄 **Full write-up: [REPORT.md](REPORT.md)** &nbsp;·&nbsp; 📊 **All runs & results: [EXPERIMENTS.md](EXPERIMENTS.md)**

---

## Goal

Pick a custom domain (email security) and answer two questions:

1. **Can I build a clean, diverse, *correctly labeled* benchmark** for phishing
   detection from just a task description — no scraped or hand-labeled data?
2. **Which LLMs are actually best at this task**, and is "bigger/more expensive"
   really better?

A key requirement from the brief: distinguish a model that is simply *weak* (acceptable)
from a **benchmark that is generated incorrectly** (a bug to fix).

## The task

| Role | Model(s) | Job |
|------|----------|-----|
| 👨‍🏫 Teacher | a capable LLM | generate emails with controlled attributes (label, sender type, writing quality, tone, context) |
| 🎓 Students | several LLMs | classify each email: Phishing / Suspicious / Legitimate |
| ⚖️ Judge | `exact_match` (deterministic) | score each answer against the ground-truth label |

Attribute space: `label × sender_type × quality = 3 × 2 × 2 = 12` combinations.

## What I found and fixed

**Bug #1 — diversity / sampling (fixed in code).** The generator only produced the full,
balanced set of attribute combinations when `sampling.target == 'all'`; with any other
setting each email got just **one random attribute**, and the ground-truth `label` was
**dropped on ~50% of items**. I fixed this in the system code (`phase3.py`) so target
attributes are always fully assigned — verified by 778 passing tests and a direct check.

**Bug #2 — content didn't match the label (fixed via the task prompt).** Even after #1,
the teacher wrote phishing-style content for *every* label (the prompt gave only a
Phishing example), so "Legitimate" emails still looked like scams. I rewrote the prompt
with a definition and an example *per class*. Result: the content finally matched the
labels — and **student accuracy roughly doubled.**

| | Before fixes | After fixes |
|---|---|---|
| Attributes controlled per email | 1 of 3 | **3 of 3** |
| Emails with no ground-truth label | 50% | **0%** |
| Combinations covered | random, repeating | **all 12** |
| Top model accuracy | ~0.42 | **~0.88** |

## Results — model ranking

Largest run: 24 balanced emails (8 per class), 6 students (free + cheap paid frontier),
total cost **USD 0.0057**. Judge = `exact_match`.

| Rank | Model | Tier | Accuracy | Phishing | Suspicious | Legitimate |
|-----:|-------|------|---------:|---------:|-----------:|-----------:|
| 1 | gemini-2.5-flash-lite | paid | **0.88** | 8/8 | **5/8** | 8/8 |
| 2 | gpt-4o-mini | paid | 0.79 | 8/8 | 3/8 | 8/8 |
| 2 | gemma-4-26b | free | 0.79 | 8/8 | 3/8 | 8/8 |
| 4 | gpt-oss-20b | free | 0.71 | 8/8 | 1/8 | 8/8 |
| 5 | lfm-1.2b | free | 0.67 | 8/8 | 0/8 | 8/8 |
| 5 | claude-3.5-haiku | paid | 0.67 | 8/8 | 0/8 | 8/8 |

**Takeaways:**

- **Every model nails Phishing and Legitimate (8/8)** — the whole ranking is decided by
  the genuinely ambiguous **Suspicious** class.
- **Price and size do not predict quality here:** paid `claude-3.5-haiku` tied *last*
  with the tiny free `lfm-1.2b`, and free `gemma-4-26b` matched paid `gpt-4o-mini`.

## Example generated emails

Real items the teacher produced (after the fixes), one per class:

- **Phishing** — *"We have noticed some unusuall activity on your bank account … click the link below and confirm your login details: http://bank-secure-verify-login…"* (urgency + fake link + credential request)
- **Suspicious** — *"I just got a note about a small file attached to a recent email … could you take a quick look at the attachment? No rush."* (mild red flags, no overt malice)
- **Legitimate** — *"Just wanted to let you knwo that your order #A12345 has been shipped … tracking number 1Z999AA10123456784."* (genuine, no links/credentials)

## Repository layout (my work)

| Path | What |
|------|------|
| [`REPORT.md`](REPORT.md) | Full report: root-cause analysis, before/after, model ranking, fix log |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Results ledger for every run |
| `Runs/_guy_configs/*.yaml` | My experiment configs (`01a`–`08`) |
| `Runs/EXP-guy-*/` | Run artifacts: generated emails, student answers, scores |
| `scripts/summarize_runs.py` | Regenerates the results table from disk |

## How to run

```bash
# 1. Add an OpenRouter key to keys.yaml (git-ignored):
#    providers:
#      openrouter: sk-or-v1-...

# 2. The model ranking (24 emails, 6 models):
COEVAL_MAX_WORKERS=2 python -m runner.cli run --config Runs/_guy_configs/08_big_mixed.yaml

# 3. Refresh the results table:
python scripts/summarize_runs.py
```

## Credits

MailRank is built on **[CoEval](https://github.com/ApartsinProjects/CoEval)**, an
open-source teacher/student/judge evaluation framework, used here as a third-party
library. All MailRank-specific design, experiments, findings and fixes are my own.
