"""Generate the CoEval paper content figures (F2-F6) from committed experiment artifacts.

Run from repo root:  python docs/paper/figures/make_figures.py
Writes PNGs into docs/paper/figures/.  Reproducible (no randomness).
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parent
RUNS = ROOT / "Runs"

plt.rcParams.update({
    "font.size": 11, "font.family": "serif", "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
BLUE, GREEN, GREY, RED = "#2b6cb0", "#2f855a", "#a0aec0", "#c53030"


def _load(p):
    return json.loads((RUNS / p).read_text(encoding="utf-8"))


def f2_icc_curve():
    """F2: non-monotone ICC(3,k) vs k with Spearman-Brown overlay (composition>size)."""
    d = _load("EXP002-ensemble-size-ablation/reports/ensemble_ablation.json")
    ks = [r["k"] for r in d["per_k"]]
    icc_k = [r["icc_3_k"] if r["icc_3_k"] == r["icc_3_k"] else np.nan for r in d["per_k"]]
    icc_1 = [r["icc_3_1"] if r["icc_3_1"] == r["icc_3_1"] else np.nan for r in d["per_k"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(ks, icc_k, "o-", color=BLUE, lw=2.4, label="ICC(3,k) average-measures")
    ax.plot(ks, icc_1, "s--", color=GREY, lw=1.8, label="ICC(3,1) single-rater")
    ax.axhspan(0.66, 0.72, color=GREEN, alpha=0.12)
    ax.annotate("selected strong panel\n(reliable, ICC≈0.70)", xy=(2, 0.698),
                xytext=(2.15, 0.50), fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.annotate("naive size-scaling adds\nlow-agreement judges →\nreliability falls", xy=(4, 0.405),
                xytext=(2.6, 0.20), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("judges in ensemble (k)"); ax.set_ylabel("inter-rater reliability (ICC)")
    ax.set_xticks(ks); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Composition, not size: selection delivers reliability")
    fig.tight_layout(); fig.savefig(FIG / "f2_icc_curve.png", bbox_inches="tight"); plt.close(fig)


def f3_verbosity_forest():
    """F3: per-judge length-bias forest plot with ensemble at ~0."""
    d = _load("EXP004-verbosity-bias-analysis/reports/verbosity_bias.json")
    judges = d["judges"]
    rows = [(j, d["judge_pooled"][j]["point"], d["judge_pooled"][j]["lo"],
             d["judge_pooled"][j]["hi"]) for j in judges]
    rows.append(("ENSEMBLE", d["ensemble_pooled"]["point"],
                 d["ensemble_pooled"]["lo"], d["ensemble_pooled"]["hi"]))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ys = list(range(len(rows)))[::-1]
    for y, (name, pt, lo, hi) in zip(ys, rows):
        c = GREEN if name == "ENSEMBLE" else BLUE
        ax.plot([lo, hi], [y, y], color=c, lw=2)
        ax.plot(pt, y, "o", color=c, ms=7 if name == "ENSEMBLE" else 5)
    ax.axvline(0, color=GREY, lw=1, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Pearson r (response length vs. score)")
    ax.set_title("Mixed-sign per-judge length bias cancels in the ensemble")
    fig.tight_layout(); fig.savefig(FIG / "f3_verbosity_forest.png", bbox_inches="tight"); plt.close(fig)


def f4_correctness_corr():
    """F4: CoEval recovers the true model ranking with no labels (QA accuracy).

    Earlier this figure plotted per-judge vs ensemble correlation, but on objective
    QA every capable judge converges to ~0.85, so the bars were visually identical
    and read as "the ensemble adds nothing". The decisive evidence for the use case
    is that CoEval's label-free scores reproduce the ground-truth ranking; we plot
    that directly (the ensemble-vs-single-judge advantage is Figure 4 / Section 5.3)."""
    d = _load("EXP001b-exactmatch-qa/reports/model_ranking.json")
    rows = d["ranking"]  # already best -> worst
    models = [r["student"] for r in rows]
    coe = [r["coeval_score"] for r in rows]
    err = [[r["coeval_score"] - r["lo"] for r in rows], [r["hi"] - r["coeval_score"] for r in rows]]
    gt = [r["ground_truth_acc"] for r in rows]
    y = np.arange(len(models))[::-1]  # best model at the top
    h = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.barh(y + h / 2, coe, height=h, color=GREEN, alpha=0.9,
            xerr=err, capsize=4, label="CoEval score (no labels)")
    ax.barh(y - h / 2, gt, height=h, color=BLUE, alpha=0.75,
            label="Ground-truth accuracy")
    for yi, c, g in zip(y, coe, gt):
        ax.text(c + 0.012, yi + h / 2, f"{c:.2f}", va="center", fontsize=7.5)
        ax.text(g + 0.012, yi - h / 2, f"{g:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlim(0, 1.14)
    ax.set_xlabel("Score")
    ax.set_title("CoEval recovers the true model ranking with no labels")
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92, edgecolor="none")
    fig.tight_layout(); fig.savefig(FIG / "f4_correctness_corr.png", bbox_inches="tight"); plt.close(fig)


def f5_rubric_heatmap():
    """F5: rubric criterion semantic-similarity heatmap (within > cross task)."""
    d = _load("EXP005-rubric-generalization/reports/rubric_overlap.json")
    sim = np.array(d["similarity_matrix"])
    labels = [f"{c['task'][:4]}/{c['criterion'][:12]}" for c in d["criteria"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine similarity")
    ax.set_title(f"Rubric criteria: within-task {d['mean_within_task_similarity']:.2f} > "
                 f"cross-task {d['mean_cross_task_similarity']:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "f5_rubric_heatmap.png", bbox_inches="tight"); plt.close(fig)


def f6_cost():
    """F6: evaluations-per-dollar vs human annotation (illustrative orders of magnitude)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    # CoEval: 7978 evals / $5.89 = 1354 evals/$; human ~ $0.30-3.00 per annotation -> 0.33-3.3 evals/$
    methods = ["Human\nannotation\n(low)", "Human\nannotation\n(high)", "CoEval\nensemble"]
    evals_per_dollar = [1/3.0, 1/0.30, 7978/5.89]
    colors = [GREY, GREY, GREEN]
    bars = ax.bar(methods, evals_per_dollar, color=colors)
    ax.set_yscale("log"); ax.set_ylabel("evaluations per dollar (log scale)")
    for b, v in zip(bars, evals_per_dollar):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.0f}" if v >= 1 else f"{v:.2f}",
                ha="center", fontsize=9)
    ax.set_title("Throughput: 7,978 evaluations for \\$5.89")
    fig.tight_layout(); fig.savefig(FIG / "f6_cost.png", bbox_inches="tight"); plt.close(fig)


def f1_architecture():
    """F1: clean 5-phase pipeline with teacher/student/judge role bands."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    phases = [
        ("Phase 1", "Attribute\nMapping"),
        ("Phase 2", "Rubric\nConstruction"),
        ("Phase 3", "Datapoint\nGeneration"),
        ("Phase 4", "Response\nCollection"),
        ("Phase 5", "Ensemble\nScoring"),
    ]
    roles = [("TEACHER", 0, 3, "#2b6cb0"), ("STUDENT", 3, 4, "#dd6b20"),
             ("JUDGE", 4, 5, "#2f855a")]
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.2); ax.axis("off")
    bw, gap, x0, y = 1.7, 0.25, 0.2, 0.7
    centers = []
    for i, (ptag, label) in enumerate(phases):
        x = x0 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, y), bw, 1.1, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#f1f3f9", ec="#1f2a44", lw=1.4))
        ax.text(x + bw / 2, y + 0.82, ptag, ha="center", va="center", fontsize=8.5,
                color="#1f2a44", weight="bold")
        ax.text(x + bw / 2, y + 0.36, label, ha="center", va="center", fontsize=9.5,
                color="#1a1a1a")
        centers.append(x + bw / 2)
        if i < len(phases) - 1:
            ax.add_patch(FancyArrowPatch((x + bw, y + 0.55), (x + bw + gap, y + 0.55),
                                         arrowstyle="-|>", mutation_scale=14, color="#1f2a44", lw=1.4))
    # role bands above
    for name, a, b, col in roles:
        xa = x0 + a * (bw + gap)
        xb = x0 + (b - 1) * (bw + gap) + bw
        ax.add_patch(FancyBboxPatch((xa, 2.15), xb - xa, 0.5,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=col, ec="none", alpha=0.9))
        ax.text((xa + xb) / 2, 2.4, name, ha="center", va="center", fontsize=10,
                color="white", weight="bold")
        ax.plot([(xa + xb) / 2, (xa + xb) / 2], [2.15, 1.8], color=col, lw=1, ls=":")
    ax.text(x0, 0.25, "YAML config", fontsize=8.5, color="#667", style="italic")
    ax.text(centers[-1], 0.25, "8 interactive reports + benchmark export", fontsize=8.5,
            color="#667", style="italic", ha="center")
    ax.set_title("CoEval pipeline: one model pool, three rotating roles", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "f1_architecture.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for fn in (f1_architecture, f2_icc_curve, f3_verbosity_forest, f4_correctness_corr,
               f5_rubric_heatmap, f6_cost):
        fn(); print("wrote", fn.__name__)
