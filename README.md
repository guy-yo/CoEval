# CoEval: Ensemble-Based Self-Evaluation for LLMs

📄 **[Read the CoEval paper online](https://apartsinprojects.github.io/CoEval/)** &nbsp;·&nbsp; [Download the Word version](https://github.com/ApartsinProjects/CoEval/raw/master/docs/paper/CoEval.docx)

[![Paper](https://img.shields.io/badge/%F0%9F%93%84%20paper-online-success)](https://apartsinprojects.github.io/CoEval/)
[![Status WIP](https://img.shields.io/badge/status-WIP-yellow)](CHANGELOG.md)
[![Python ≥3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Version 0.3.0](https://img.shields.io/badge/version-0.3.0-informational)](CHANGELOG.md)
[![Tests 622 passing](https://img.shields.io/badge/tests-622%20passing-brightgreen)](docs/README/11-testing.md)
[![© 2026 Alexander Apartsin](https://img.shields.io/badge/%C2%A9%202026-Alexander%20Apartsin-red)](README.md)

<p align="center">
  <img src="docs/coeval_banner.jpg" alt="CoEval — Teacher · Student · Judge evaluation ensemble" width="860"/>
</p>

---

## 📄 Published Paper

**[CoEval: Ranking Language Models for Custom Tasks Without Labeled Data or Trustworthy Benchmarks](https://apartsinprojects.github.io/CoEval/)** — read it online (HTML with rendered math) or download the [Word version](https://github.com/ApartsinProjects/CoEval/raw/master/docs/paper/CoEval.docx).

*Alexander Apartsin (Holon Institute of Technology) · Yehudit Aperstein (Afeka Tel Aviv Academic College of Engineering)*

CoEval ranks models for a custom task or domain in the hardest setting: when **no task-specific labeled data** exists and **public benchmarks cannot be trusted** because their items have likely leaked into pretraining. From only a task description, a teacher model synthesizes a fresh, contamination-free benchmark and a cross-family judge ensemble ranks the candidates, with no human labels or raters.

| Result | Evidence |
|--------|----------|
| Recovers the **true model ranking** with no labeled data | Spearman ρ = 0.86 vs ground-truth correctness, 95% CI [0.77, 0.94] |
| **Doubly-robust ranking** recovers the true ordering and resists rogue judges | reliability and discrimination weighting lift rank-recovery to Spearman 0.95 across 13 models; an injected random judge receives weight 0.00 |
| **Rankings are domain-specific**, so a generic leaderboard misleads | three different models top four CoEval-generated domains; the pooled-best model is domain-best in only 1 of 4 |
| Cancels a **verbosity bias no single judge avoids** | ensemble *r* = +0.010 (CI spans zero), a 93% reduction |
| **Composition over size**: panel diversity, not panel size, drives reliability | ICC(3,*k*) peaks at two well-chosen judges, falls as low-agreement judges are added |
| Structurally precludes **same-family self-preference** | vendor-disjoint panel; aggregation shifts every score ≤ 0.015 |
| **Contamination-free** generated items | 0.0000 verbatim 13-gram overlap with five major public benchmarks |
| **Inexpensive** enough to re-run per model release | 7,978 evaluations for USD 5.89, fully automated |

---

## 🚨 The Challenge

**Evaluating and selecting off-the-shelf or fine-tuned models for a specific use case is difficult.**

Choosing the right LLM means navigating a minefield of hidden pitfalls:

|     | Challenge                                                      | Why It Hurts                                                                                                          |
|:---:|:-------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------- |
| 🎯  | **Generic benchmarks don't transfer**                          | Public data and metrics often miss the nuances of *your* real-world requirements.                                     |
| 🧩  | **Custom benchmarks are hard to design**                       | Defining representative tasks, building rubrics, and choosing robustness variations is non-trivial.                   |
| 💸  | **Multi-model multi-task benchmarks are expensive to execute** | Running every candidate model across every task and rubric quickly multiplies cost and compute.                       |
| 🕳️ | **Leakage biases results**                                     | Public and private benchmark items (or near-duplicates) may lurk in training data, inflating scores via memorization. |
| ⚙️  | **Ops and cost are complex**                                   | Running evaluations across providers, inference modes, and scoring criteria demands careful orchestration.            |

> **Bottom line:** You can't trust a leaderboard number, and building your own eval is a project in itself.

---

## 💡 The Concept

**Ensemble-based synthetic self-evaluation benchmarking** — let the models evaluate *each other*.

CoEval generates a synthetic evaluation suite spanning multiple domain-specific tasks and scoring rubrics, then assembles an **ensemble of models** that rotate through three roles:

```
┌─────────────────────────────────────────────────────────────┐
│                     MODEL  ENSEMBLE                         │
│                                                             │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│   │  Model A   │    │  Model B   │    │  Model C   │  ...   │
│   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘          │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓     │
│   ┃          ROTATING  ROLE  ASSIGNMENT               ┃     │
│   ┗━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━┛     │
│              ▼                ▼                  ▼           │
│      🎓 TEACHER        📝 STUDENT          ⚖️ JUDGE        │
│   Generate synthetic   Models under       Score outputs     │
│   challenges &         evaluation take    against the       │
│   reference answers    the challenges     rubric            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Reliability through selection

Not all teachers and judges are created equal. CoEval improves signal quality by identifying:

| Role           | Selection Criterion                                                         | Intuition                                    |
|:--------------:|:--------------------------------------------------------------------------- |:-------------------------------------------- |
| 🎓 **Teacher** | **Differentiating** — produces challenges that separate student performance | A good exam question reveals who studied.    |
| ⚖️ **Judge**   | **Consensus** — high agreement with ensemble majority                       | A reliable judge aligns with peer consensus. |

These two label-free signals (item discrimination and judge agreement) combine into CoEval's **doubly-robust ranking**, which recovers the true model ordering at Spearman 0.95 across 13 candidate models and assigns an injected rogue judge zero weight. Because the best model is domain-specific, CoEval ranks on *your* domain rather than trusting a pooled leaderboard.

### Flexible provisioning

```
  Fully Automatic          Semi-Automatic               Manual
  ┌────────────┐          ┌────────────────┐        ┌──────────────┐
  │ Tasks       │          │ Tasks ✏️       │        │ Tasks ✏️      │
  │ Rubrics     │  ──►     │ Rubrics        │  ──►   │ Rubrics ✏️    │
  │ Attr. Space │          │ Attr. Space ✏️ │        │ Attr. Space ✏️│
  └────────────┘          └────────────────┘        └──────────────┘
   AI-generated            Human-guided               Human-defined
```

Tasks, rubrics, and diversity/attribute spaces can be provisioned **fully automatically**, **semi-automatically** (human-in-the-loop), or **manually** — choose the level of control that fits your workflow.

---

## 🏗️ The Framework

**CoEval is an end-to-end system** — from benchmark design to interactive reporting.

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                        C o E v a l                          ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║   📦 Multi-Vendor Support                                   ║
  ║   ├── Multiple LLM providers & interfaces out of the box    ║
  ║   └── Plug in proprietary / self-hosted models              ║
  ║                                                              ║
  ║   🗺️ Benchmark Design & Planning                            ║
  ║   ├── Automated task & rubric provisioning                  ║
  ║   └── Run orchestration with cost optimization              ║
  ║                                                              ║
  ║   📊 Interactive Visual Reports                             ║
  ║   ├── Side-by-side model comparison                         ║
  ║   └── Drill-down into tasks, rubrics & scores               ║
  ║                                                              ║
  ║   🔄 Experiment Tracking                                    ║
  ║   ├── Easy reruns & parameter sweeps                        ║
  ║   └── Repair & resume after interruptions                   ║
  ║                                                              ║
  ║   📚 Complete Documentation                                 ║
  ║   ├── User guides & tutorials                               ║
  ║   └── Developer API reference                               ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
```

### At a glance

| Feature                | Description                                                              |
|:---------------------- |:------------------------------------------------------------------------ |
| **Multi-vendor**       | Swap providers without changing your eval pipeline.                      |
| **Auto-provisioning**  | Generate tasks, rubrics, and attribute spaces from a domain description. |
| **Orchestration**      | Schedule and parallelize runs; optimize for cost and latency.            |
| **Visual reports**     | Interactive dashboards for deep-dive analysis.                           |
| **Resilient tracking** | Resume interrupted experiments; repair partial results.                  |
| **Docs-first**         | Comprehensive guides for users and contributors alike.                   |

---

## Supported Model APIs

OpenAI, Anthropic, Google Gemini, Azure OpenAI, Azure AI Inference, AWS Bedrock, Google Vertex AI, OpenRouter, Groq, DeepSeek, Mistral, DeepInfra, Cerebras, Cohere, HuggingFace API, HuggingFace (local), Ollama

→ [Providers & Pricing](docs/README/05-providers.md) — auth setup, batch discounts, pricing tables for all 18 interfaces.

---

## Quick Start

```bash
# 1. Install
pip install coeval

# 2. Add your API keys  (see: docs/tutorial.md § 2)
cp keys.yaml.template keys.yaml   # then fill in your provider keys

# 3. Probe all models — no tokens consumed  (runnable example included in the repo)
coeval probe --config examples/quickstart.yaml

# 4. Estimate cost before spending anything
coeval plan --config examples/quickstart.yaml

# 5. Run the experiment (phases 1-5: infer attributes + rubric, generate, respond, judge)
coeval run --config examples/quickstart.yaml

# 6. Generate analysis reports
coeval analyze all --run ./Runs/quickstart --out ./Runs/quickstart/reports
```

### Levels of specification

CoEval accepts your intent at whichever level of detail you have, from a single
sentence to a fully hand-written config:

| Level | You provide | CoEval infers | How |
|-------|-------------|---------------|-----|
| **Objective** | one-line goal | everything: tasks, attributes, rubric, model roles | `coeval wizard --objective "..."` |
| **Most-automatic** | task description + models | target attributes + rubric (Phases 1-2) | hand-write a minimal YAML (below) |
| **Semi-automatic** | description + some attributes/rubric | the rest | partial YAML, human-in-the-loop wizard |
| **Manual** | full config | nothing | complete YAML |

Generate a complete, runnable config from a single high-level objective (no
questions asked) and run it:

```bash
# One sentence in, a validated config out
coeval wizard \
  --objective "rank LLMs on classifying customer-support tickets into urgency levels" \
  --models "gpt-4o-mini, claude-3-5-haiku" \
  --items 8 \
  --out ticket_urgency.yaml

coeval run --config ticket_urgency.yaml
```

The LLM proposes the tasks, target attributes, scoring rubric, and a
cross-family judge panel; the config is auto-validated (and auto-repaired on any
validation error) before it is written. Omit `--objective` for the interactive,
question-by-question wizard instead.

### Minimal experiment config (most-automatic level)

You give only a task description and the models; CoEval infers the target attributes
and the scoring rubric. See the complete runnable file at `examples/quickstart.yaml`.

```yaml
models:
  - name: gpt-4o-mini
    interface: openrouter
    parameters: { model: openai/gpt-4o-mini, temperature: 0.7, max_tokens: 512 }
    roles: [teacher, student, judge]
  - name: claude-haiku
    interface: openrouter
    parameters: { model: anthropic/claude-3.5-haiku, temperature: 0.0, max_tokens: 128 }
    roles: [judge]            # cross-family judge

tasks:
  - name: regex_explanation
    description: Explain in plain English what a given regular expression matches.
    output_description: A clear one-to-three sentence plain-English explanation.
    sampling: { total: 6 }    # target_attributes + rubric are inferred (Phases 1-2)
    evaluation_mode: single

experiment:
  id: quickstart
  storage_folder: ./eval_runs
```

---

## Examples

Interactive HTML examples — click to open rendered in browser:

### Experiment Planning

| Example | Description |
|---------|-------------|
| [Education Benchmark — Planning View](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/education/education_description.html) | Full experiment plan: 3 real-dataset tasks + 10 synthetic tasks, 6 models, per-phase call budget, cost table, and attribute maps |
| [Mixed Benchmark — Planning View](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/mixed/mixed_description.html) | Mixed benchmark plan: real benchmark datasets + OpenAI models |
| [Paper Dual-Track — Planning View](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/paper/paper_dual_track_description.html) | Paper evaluation: dual-track design with benchmark + generative teachers |

> **Generate your own planning view:**
> ```bash
> coeval describe --config my_experiment.yaml --out my_experiment_plan.html
> ```

### Example of Reports

| Report | Description |
|--------|-------------|
| [Dashboard](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/medium-benchmark/reports/index.html) | Overview dashboard — all reports in one place with top-line rankings and navigation |
| [Student Performance Report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/medium-benchmark/reports/student_report/index.html) | Per-student score breakdowns, task rankings, rubric factor heatmaps |
| [Judge Consistency Report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/medium-benchmark/reports/judge_consistency/index.html) | Inter-judge ICC agreement, calibration drift, flagged uncertain items |
| [Robust Summary Report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/medium-benchmark/reports/summary/index.html) | Final model rankings with confidence intervals and robust ensemble weights |
| [Score Distribution Report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/medium-benchmark/reports/score_distribution/index.html) | High / Medium / Low histograms filterable by task, teacher, student, and judge |
| [Teacher Report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/medium-benchmark/reports/teacher_report/index.html) | Per-teacher source quality, attribute stratum coverage, data consistency |
| [Interaction Matrix](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/medium-benchmark/reports/interaction_matrix/index.html) | Teacher × Student pair quality heatmap — spot which combinations succeed or fail |
| [Coverage Summary](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/medium-benchmark/reports/coverage_summary/index.html) | Attribute Coverage Ratio (ACR) and rare-attribute recall per task |
| [Judge Report](https://htmlpreview.github.io/?https://raw.githubusercontent.com/ApartsinProjects/CoEval/master/Runs/medium-benchmark/reports/judge_report/index.html) | Judge-level bias rates, score calibration, inter-rater reliability |
| [Annotated Report Guide](Docs/paperv2/report_samples.md) | Detailed annotated screenshots of every CoEval report with explanations of every visualization and metric |

> **Generate all reports from a completed run:**
> ```bash
> coeval analyze all --run ./Runs/my-experiment-v1 --out ./reports
> ```

---

## Related documents

| Guide | What it covers |
|-------|----------------|
| [Concepts Glossary](docs/concepts.md) | Every first-class concept explained: teacher, student, judge, attributes, rubric, datapoint, slot, phases, wizard, probing, planning, resume, repair, auto interface, batch API, and more |
| [Evaluation Experiment Planning and Preparation Guide](docs/tutorial.md) | End-to-end walkthrough: installation, config design, probing, running, analysis, and benchmark export |
| [Command Line Option Reference](docs/cli_reference.md) | Every `coeval` subcommand, flag, and exit code — `run`, `probe`, `plan`, `generate`, `status`, `models`, `analyze`, `describe`, `wizard`, `ingest`, `repair` |
| [Running Experiments](docs/README/06-running.md) | Phase modes, `--continue`, batch API, quota control, cost estimation, fault recovery, use-case examples |
| [Providers & Pricing](docs/README/05-providers.md) | All 18 interfaces with auth, batch support, code examples, and pricing tables |
| [Analytics & Reports](docs/README/08-reports.md) | 11 interactive HTML dashboards, paper-quality result tables, programmatic API, Excel workbook export |
| [Configuration Guide](docs/README/04-configuration.md) | YAML config schema: models, tasks, attributes, rubric, sampling, prompt overrides, experiment settings |
| [Benchmark Datasets](docs/README/07-benchmarks.md) | Pre-ingested datasets, `coeval ingest`, `interface: benchmark` virtual teacher, reproducing published results |
| [Testing Guide](docs/testing.md) | All 20 test files, how to run each suite, interpreting failures, CI/CD setup |
| [System Feature Wishlist](Docs/paperv2/system_todo.md) | 35-item prioritized roadmap: 10 benchmark additions, 12 system features, 13 new report types |

---

## Pipeline at a Glance

```
YAML Config  →  Phase 1: Attribute Mapping   (teachers infer task dimensions)
             →  Phase 2: Rubric Mapping       (teachers build evaluation criteria)
             →  Phase 3: Data Generation      (teachers produce benchmark items)
             →  Phase 4: Response Collection  (students answer benchmark prompts)
             →  Phase 5: Evaluation           (judges score student responses)
             →  coeval analyze all            (8 HTML reports + Excel workbook)
```

### 16 Model Interfaces

| Cloud — Async Batch ✅ | Cloud — Real-time | OpenAI-Compatible | Local / Virtual |
|:---:|:---:|:---:|:---:|
| `openai` | `azure_openai`¹ | `groq` | `huggingface` |
| `anthropic` | `azure_ai` | `deepseek` | `ollama` |
| `gemini`² | `bedrock` | `mistral` | `benchmark` |
| | `vertex` | `deepinfra` | |
| | `openrouter` | `cerebras` | |

> ¹ `azure_openai` supports Azure Global Batch API (50% discount) — enable via `batch: azure_openai:` in config.
> ² `gemini` uses concurrent requests (pseudo-batch) — no async discount.

### Key Capabilities

| Capability | Detail |
|-----------|--------|
| **Cost estimation** | Itemised call budget and cost table before any phases run; Batch API discounts modelled |
| **Batch API** | 50% async discount for OpenAI, Anthropic, and Azure OpenAI; Gemini uses concurrent mode (no discount) |
| **Resume** | `--continue` resumes at exact JSONL record; no duplicate API calls |
| **Auto attributes** | Teachers infer task dimensions from a description; no hand-labelling required |
| **Auto rubric** | Teachers propose rubric factors; merge-and-deduplicate across N teachers |
| **Multi-judge ensemble** | N judges → bias-resistant aggregate scores; outlier judges down-weighted |
| **8 HTML reports** | Interactive charts, filterable tables, CSV export, fully self-contained (no CDN) |
| **Model probe** | Verify all 16 interfaces are reachable before spending a dollar |
| **Virtual teachers** | Pre-ingested public datasets supply zero-cost Phase 3 ground truth |
| **Label accuracy** | Judge-free exact-match for classification tasks (`label_attributes`) |

### Project Statistics · System v1.3

| Component | Files | LoC |
|-----------|------:|----:|
| `Code/runner` — pipeline engine | 59 `.py` | 15,087 |
| `Code/analyzer` — analysis & reports | 21 `.py` | 9,554 |
| `Public/benchmark` — dataset utilities | 34 `.py` | 5,211 |
| `Tests` — test suites | 41 `.py` | 16,845 |
| `docs` — documentation | 35 `.md` | 12,521 |

---

<div align="center">

**CoEval** · Multi-Model LLM Evaluation Framework

*Designed for LLM developers, integrators, and evaluation practitioners who require robust model evaluation and ranking using custom use-case data and metrics.*

Copyright (c) 2026 Alexander Apartsin. All rights reserved.

</div>
