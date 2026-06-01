"""Family-aware (vendor-disjoint) aggregation to control systematic same-family bias.

The literature (Preference Leakage; Play Favorites) shows judges can favor outputs
from their own model family. CoEval's cross-family ensemble mitigates this in
expectation, but we can make the control *explicit*: when scoring a model under
test, exclude judges that share its vendor family, so no model ever contributes
to its own (or its family's) score. This module computes both the naive ensemble
ranking and the family-disjoint ranking and reports whether they differ&mdash;a
direct, auditable correction for systematic same-family bias.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .loader import EESDataModel
from .self_eval_control import model_family
from .stats import mean_ci


def _student_scores(model: EESDataModel, task: str | None,
                    family_aware: bool, aspect: str | None = None) -> dict[str, list[float]]:
    """Per-student normalised scores. If family_aware, drop judges sharing the
    student's vendor family (vendor-disjoint aggregation)."""
    by_resp: dict[tuple[str, str], list[float]] = defaultdict(list)
    for u in model.units:
        if task and u.task_id != task:
            continue
        if aspect and u.rubric_aspect != aspect:
            continue
        if family_aware and model_family(u.judge_model_id) == model_family(u.student_model_id):
            continue
        by_resp[(u.student_model_id, u.response_id)].append(u.score_norm)
    out: dict[str, list[float]] = defaultdict(list)
    for (s, _), v in by_resp.items():
        out[s].append(float(np.mean(v)))
    return out


def compute_family_aware_ranking(model: EESDataModel, tasks: list[str] | None = None,
                                 aspect: str | None = None, seed: int = 0) -> dict[str, Any]:
    """Compare naive vs vendor-disjoint rankings per task."""
    tasks = tasks or sorted(model.tasks)
    out: dict[str, Any] = {"run": str(model.run_path),
                           "families": {s: model_family(s) for s in sorted(model.students)},
                           "per_task": {}}
    for task in tasks:
        naive = _student_scores(model, task, family_aware=False, aspect=aspect)
        disj = _student_scores(model, task, family_aware=True, aspect=aspect)
        if not naive:
            continue

        def rank(d):
            order = sorted(d, key=lambda s: -np.mean(d[s]))
            return order, {s: mean_ci(d[s], seed=seed) for s in order}

        n_order, n_ci = rank(naive)
        d_order, d_ci = rank(disj)
        out["per_task"][task] = {
            "naive_ranking": n_order,
            "vendor_disjoint_ranking": d_order,
            "ranking_changed": n_order != d_order,
            "naive_scores": {s: round(n_ci[s].point, 3) for s in n_order},
            "vendor_disjoint_scores": {s: round(d_ci[s].point, 3) for s in d_order},
        }
    out["any_ranking_changed"] = any(v["ranking_changed"] for v in out["per_task"].values())
    return out
