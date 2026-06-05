"""Phase 5 — Evaluation (REQ-7.1, REQ-7.5, REQ-7.6, §4, §6.2.6, §9)."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from ..config import CoEvalConfig, TaskConfig, ModelConfig
from ..exceptions import PartialPhaseFailure
from ..interfaces import ModelPool, create_batch_runner
from ..logger import RunLogger
from ..metric_judge import split_rubric, score_metric_factors
from ..prompts import get_prompt
from ..storage import ExperimentStorage
from .utils import call_llm_json, call_llm_word, QuotaTracker, parse_json_text, parse_word_text

# Maximum concurrent workers for non-batch, network-bound judge calls.
# Override with COEVAL_MAX_WORKERS (set to 1-3 for rate-limited free-tier models).
_MAX_WORKERS = int(os.environ.get('COEVAL_MAX_WORKERS', '10'))
# Separator for batch keys — ASCII NUL never appears in task/model/response IDs
_KEY_SEP = "\x00"
# Suffix used in batch_key to distinguish single-mode evaluation entries
_SINGLE_SUFFIX = "\x01"


def run_phase5(
    cfg: CoEvalConfig,
    storage: ExperimentStorage,
    logger: RunLogger,
    pool: ModelPool,
    quota: QuotaTracker,
    phase_mode: str,
    only_models: set[str] | None = None,
) -> None:
    """Execute Phase 5 for all (task, teacher, judge) triples.

    Judges are grouped by interface:
      - Network interfaces (OpenAI, Anthropic, Gemini): when
        ``experiment.batch.<interface>.evaluation: true`` is set, all
        requests are submitted in one batch job per interface.  Otherwise they
        run concurrently (ThreadPoolExecutor, up to _MAX_WORKERS).
      - HuggingFace: always sequential (GPU-bound).

    Individual judge failures are logged but do not abort the phase as long as
    at least one evaluation was recorded.

    If only_models is provided, only judges in that set are processed.
    """
    teachers = cfg.get_models_by_role('teacher')
    all_judges = cfg.get_models_by_role('judge')
    judges = (
        [j for j in all_judges if j.name in only_models]
        if only_models is not None else all_judges
    )

    # Split judges: metric vs network vs HuggingFace
    metric_judges: list[ModelConfig] = []
    by_iface: dict[str, list[ModelConfig]] = defaultdict(list)
    hf_judges: list[ModelConfig] = []
    for j in judges:
        if j.interface == 'metric':
            metric_judges.append(j)
        elif j.interface == 'huggingface':
            hf_judges.append(j)
        else:
            by_iface[j.interface].append(j)

    errors: list[str] = []

    # --- Metric judges: deterministic computation, no LLM call ---
    for task in cfg.tasks:
        for teacher in teachers:
            for judge in metric_judges:
                try:
                    _evaluate_metric(
                        task, teacher, judge, storage, logger, phase_mode,
                    )
                except Exception as exc:
                    msg = (
                        f"Phase 5: metric evaluation failed for "
                        f"(task='{task.name}', teacher='{teacher.name}', "
                        f"judge='{judge.name}'): {exc}"
                    )
                    logger.error(msg)
                    errors.append(msg)

    # --- Network-based judges: batch or concurrent sequential ---
    for iface_name, iface_judges in by_iface.items():
        if cfg.use_batch(iface_name, 'evaluation'):
            try:
                _evaluate_batch(
                    iface_name, iface_judges, cfg.tasks, teachers,
                    storage, logger, quota, phase_mode,
                )
            except Exception as exc:
                msg = f"Phase 5: {iface_name} batch evaluation failed: {exc}"
                logger.error(msg)
                errors.append(msg)
        else:
            triples = [
                (task, teacher, judge)
                for task in cfg.tasks
                for teacher in teachers
                for judge in iface_judges
            ]
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _evaluate,
                        task, teacher, judge, storage, logger, pool, quota, phase_mode,
                    ): (task, teacher, judge)
                    for task, teacher, judge in triples
                }
                for future in as_completed(futures):
                    task, teacher, judge = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        msg = (
                            f"Phase 5: evaluation failed for "
                            f"(task='{task.name}', teacher='{teacher.name}', "
                            f"judge='{judge.name}'): {exc}"
                        )
                        logger.error(msg)
                        errors.append(msg)

    # --- HuggingFace judges: sequential (GPU-bound) ---
    for task in cfg.tasks:
        for teacher in teachers:
            for judge in hf_judges:
                try:
                    _evaluate(task, teacher, judge, storage, logger, pool, quota, phase_mode)
                except Exception as exc:
                    msg = (
                        f"Phase 5: evaluation failed for "
                        f"(task='{task.name}', teacher='{teacher.name}', "
                        f"judge='{judge.name}'): {exc}"
                    )
                    logger.error(msg)
                    errors.append(msg)

    # Count total evaluations produced across all active judge combinations.
    # Skip the zero-guard when a model filter is active (partial run is expected).
    total_evals = sum(
        len(storage.read_evaluations(task.name, teacher.name, judge.name))
        for task in cfg.tasks
        for teacher in teachers
        for judge in judges
    )

    if total_evals == 0 and only_models is None:
        raise RuntimeError(
            "Phase 5: no evaluations were generated — all judge/teacher combinations failed"
        )

    if errors:
        n_total = sum(
            1
            for task in cfg.tasks
            for teacher in teachers
            for judge in judges
        )
        raise PartialPhaseFailure(
            n_failures=len(errors),
            n_successes=n_total - len(errors),
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Metric judge path (deterministic computation, no LLM)
# ---------------------------------------------------------------------------


def _evaluate_metric(
    task: TaskConfig,
    teacher: ModelConfig,
    judge: ModelConfig,
    storage: ExperimentStorage,
    logger: RunLogger,
    phase_mode: str,
) -> None:
    """Evaluate responses using deterministic metric computation (no LLM call).

    Metric judges only score rubric factors that have a ``"metric"`` key in their
    definition.  LLM-evaluated factors are ignored.
    """
    task_id = task.name
    teacher_id = teacher.name
    judge_id = judge.name
    label = f"(task='{task_id}', teacher='{teacher_id}', judge='{judge_id}')"

    if phase_mode == 'Keep':
        logger.info(f"Phase 5: {label} — Keep mode, skipping (metric)")
        return

    if phase_mode == 'Model' and storage.evaluation_file_exists(
        task_id, teacher_id, judge_id
    ):
        logger.info(f"Phase 5: {label} — Model mode, file exists, skipping (metric)")
        return

    rubric = storage.read_rubric(task_id)
    _, metric_factors = split_rubric(rubric)

    if not metric_factors:
        logger.info(f"Phase 5: {label} — no metric factors in rubric, skipping")
        return

    datapoints_index = storage.index_datapoints(task_id, teacher_id)

    # Collect all responses
    all_responses: list[dict] = []
    for resp_path in storage.iter_response_files(task_id, teacher_id):
        for line in resp_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                all_responses.append(json.loads(line))

    if not all_responses:
        logger.warning(f"Phase 5: {label} — no responses found, skipping (metric)")
        return

    # Already-evaluated response IDs (for Extend mode)
    evaluated_resp_ids = (
        storage.get_evaluated_response_ids(task_id, teacher_id, judge_id)
        if phase_mode == 'Extend' else set()
    )

    if phase_mode == 'Extend':
        new_responses = [r for r in all_responses if r['id'] not in evaluated_resp_ids]
        if not new_responses:
            logger.info(f"Phase 5: {label} — Extend mode, no new responses (metric)")
            return

    logger.info(
        f"Phase 5: {label} — scoring {len(all_responses)} responses with "
        f"{len(metric_factors)} metric factor(s)"
    )

    for resp in all_responses:
        if phase_mode == 'Extend' and resp['id'] in evaluated_resp_ids:
            continue

        dp_id = resp['datapoint_id']
        if dp_id not in datapoints_index:
            logger.error(
                f"Phase 5: datapoint_id '{dp_id}' not found in index; "
                f"skipping response '{resp['id']}' (metric)"
            )
            continue

        dp = datapoints_index[dp_id]
        reference_response = dp['reference_response']

        try:
            scores = score_metric_factors(
                metric_factors,
                hypothesis=resp['response'],
                reference=reference_response,
            )
        except Exception as exc:
            logger.error(
                f"Phase 5: metric scoring failed for response '{resp['id']}' "
                f"(judge='{judge_id}'): {exc}"
            )
            scores = {f: "0.0" for f in metric_factors}

        eval_id = f"{resp['id']}__{judge_id}"
        record: dict = {
            'id': eval_id,
            'response_id': resp['id'],
            'datapoint_id': dp_id,
            'task_id': task_id,
            'teacher_model_id': teacher_id,
            'judge_model_id': judge_id,
            'scores': scores,
            'evaluated_at': _now_iso(),
        }
        storage.append_evaluation(task_id, teacher_id, judge_id, record)

    logger.info(f"Phase 5: {label} — done (metric)")


# ---------------------------------------------------------------------------
# Batch path (network interfaces with batch enabled)
# ---------------------------------------------------------------------------


def _evaluate_batch(
    iface_name: str,
    iface_judges: list[ModelConfig],
    tasks: list[TaskConfig],
    teachers: list[ModelConfig],
    storage: ExperimentStorage,
    logger: RunLogger,
    quota: QuotaTracker,
    phase_mode: str,
) -> None:
    """Evaluate all responses for all judges of one interface via a single batch job.

    Handles both ``single`` and ``per_factor`` evaluation modes:
      - ``single``: one request per (judge, task, teacher, response) → JSON scores
      - ``per_factor``: one request per (judge, task, teacher, response, factor) → word

    Bug 1 fix: in Extend mode with new rubric factors, already-evaluated responses
    are NOT skipped — they need scoring for the new factors.  They are only skipped
    when there are no new factors (i.e. only new responses need evaluation).
    """
    access_key = iface_judges[0].access_key or None
    runner = create_batch_runner(iface_name, access_key=access_key)

    # For single-mode: batch_key → eval metadata
    pending_single: dict[str, dict] = {}
    # For per_factor mode: batch_key → eval metadata including factor name
    pending_factor: dict[str, dict] = {}

    for task in tasks:
        for teacher in teachers:
            for judge in iface_judges:
                task_id = task.name
                teacher_id = teacher.name
                judge_id = judge.name
                label = (
                    f"(task='{task_id}', teacher='{teacher_id}', judge='{judge_id}')"
                )

                if phase_mode == 'Keep':
                    logger.info(f"Phase 5: {label} — Keep mode, skipping")
                    continue

                if phase_mode == 'Model' and storage.evaluation_file_exists(
                    task_id, teacher_id, judge_id
                ):
                    logger.info(f"Phase 5: {label} — Model mode, file exists, skipping")
                    continue

                full_rubric = storage.read_rubric(task_id)
                # LLM judges only score non-metric factors
                rubric, _ = split_rubric(full_rubric)
                datapoints_index = storage.index_datapoints(task_id, teacher_id)

                all_responses: list[dict] = []
                for resp_path in storage.iter_response_files(task_id, teacher_id):
                    for line in resp_path.read_text(encoding='utf-8').splitlines():
                        line = line.strip()
                        if line:
                            all_responses.append(json.loads(line))

                if not all_responses:
                    logger.warning(f"Phase 5: {label} — no responses found, skipping")
                    continue

                evaluated_resp_ids = (
                    storage.get_evaluated_response_ids(task_id, teacher_id, judge_id)
                    if phase_mode == 'Extend' else set()
                )

                # Determine rubric_to_use for Extend mode (Bug 1 fix applied)
                new_factors: dict[str, str] = {}
                if phase_mode == 'Extend':
                    existing_evals = storage.read_evaluations(task_id, teacher_id, judge_id)
                    evaluated_factors: set[str] = set()
                    for ev in existing_evals:
                        evaluated_factors.update(ev.get('scores', {}).keys())
                    new_factors = {k: v for k, v in rubric.items() if k not in evaluated_factors}
                    new_responses = [r for r in all_responses if r['id'] not in evaluated_resp_ids]
                    if not new_factors and not new_responses:
                        logger.info(
                            f"Phase 5: {label} — Extend mode, no new rubric factors "
                            "and no new responses, skipping"
                        )
                        continue
                    if new_factors:
                        rubric_to_use = new_factors
                        logger.info(
                            f"Phase 5: {label} — Extend mode, evaluating "
                            f"{len(new_factors)} new factor(s) across "
                            f"{len(all_responses)} response(s)"
                        )
                    else:
                        rubric_to_use = rubric
                        logger.info(
                            f"Phase 5: {label} — Extend mode, {len(new_responses)} "
                            "new response(s) to evaluate (no new rubric factors)"
                        )
                else:
                    rubric_to_use = rubric

                if quota.is_exhausted(judge_id):
                    logger.warning(
                        f"Quota exhausted for model {judge_id} in phase evaluation; "
                        f"skipping {label}"
                    )
                    continue

                params = judge.get_parameters_for_role('judge')

                for resp in all_responses:
                    # Extend mode: always skip responses that already have evaluation
                    # records, regardless of whether new rubric factors exist.  New
                    # factors are applied only to responses that have never been
                    # evaluated at all.
                    if phase_mode == 'Extend' and resp['id'] in evaluated_resp_ids:
                        continue

                    if quota.is_exhausted(judge_id):
                        logger.warning(
                            f"Quota exhausted for model {judge_id} in phase evaluation"
                        )
                        break

                    dp_id = resp['datapoint_id']
                    if dp_id not in datapoints_index:
                        logger.error(
                            f"Phase 5: datapoint_id '{dp_id}' not found in index; "
                            f"skipping response '{resp['id']}'"
                        )
                        continue

                    dp = datapoints_index[dp_id]
                    common_vars = {
                        'task_description': task.description,
                        'output_description': task.output_description,
                        'input': resp['input'],
                        'target_attributes': json.dumps(dp.get('sampled_target_attributes', {})),
                        'reference_response': dp['reference_response'],
                        'response': resp['response'],
                    }

                    if task.evaluation_mode == 'single':
                        rubric_text = '\n'.join(
                            f'- {f}: {d}' for f, d in rubric_to_use.items()
                        )
                        prompt = get_prompt(
                            'evaluate_single',
                            task.prompt_library,
                            judge_id,
                            {**common_vars, 'rubric': rubric_text},
                        )
                        # Use _SINGLE_SUFFIX to distinguish single-mode keys from factor keys
                        batch_key = _KEY_SEP.join(
                            [task_id, teacher_id, judge_id, resp['id']]) + _SINGLE_SUFFIX
                        runner.add(batch_key, prompt, params)
                        quota.consume(judge_id)
                        pending_single[batch_key] = {
                            'task_id': task_id,
                            'teacher_id': teacher_id,
                            'judge_id': judge_id,
                            'resp': resp,
                            'dp_id': dp_id,
                            'rubric_factors': list(rubric_to_use.keys()),
                        }
                    else:  # per_factor
                        for factor, desc in rubric_to_use.items():
                            if quota.is_exhausted(judge_id):
                                break
                            prompt = get_prompt(
                                'evaluate_per_factor',
                                task.prompt_library,
                                judge_id,
                                {
                                    **common_vars,
                                    'rubric_factor_name': factor,
                                    'rubric_factor_description': desc,
                                },
                            )
                            batch_key = _KEY_SEP.join(
                                [task_id, teacher_id, judge_id, resp['id'], factor]
                            )
                            runner.add(batch_key, prompt, params)
                            quota.consume(judge_id)
                            pending_factor[batch_key] = {
                                'task_id': task_id,
                                'teacher_id': teacher_id,
                                'judge_id': judge_id,
                                'resp': resp,
                                'dp_id': dp_id,
                                'factor': factor,
                            }

    if not pending_single and not pending_factor:
        logger.info(
            f"Phase 5: no {iface_name} judge requests pending; "
            "skipping batch submission"
        )
        return

    total_requests = len(runner)
    logger.info(
        f"Phase 5: submitting {iface_name} batch of {total_requests} request(s)"
    )
    results = runner.run(
        description="Phase 5 evaluations",
        logger=logger,
        storage=storage,
        phase='evaluation',
    )

    # --- Process single-mode results ---
    n_single_ok = n_single_fail = 0
    for batch_key, response_text in results.items():
        if batch_key not in pending_single:
            continue
        info = pending_single[batch_key]
        task_id = info['task_id']
        teacher_id = info['teacher_id']
        judge_id = info['judge_id']
        resp = info['resp']
        rubric_factors = info['rubric_factors']

        scores: dict[str, str] = {}
        failed = not response_text
        if response_text:
            try:
                result_json = parse_json_text(response_text)
                for factor in rubric_factors:
                    val = str(result_json.get(factor, '')).strip()
                    if val not in ('High', 'Medium', 'Low'):
                        val = 'Low'
                    scores[factor] = val
                n_single_ok += 1
            except Exception:
                for factor in rubric_factors:
                    scores[factor] = 'Low'
                n_single_fail += 1
                failed = True
        else:
            for factor in rubric_factors:
                scores[factor] = 'Low'
            n_single_fail += 1

        eval_id = f"{resp['id']}__{judge_id}"
        record: dict = {
            'id': eval_id,
            'response_id': resp['id'],
            'datapoint_id': info['dp_id'],
            'task_id': task_id,
            'teacher_model_id': teacher_id,
            'judge_model_id': judge_id,
            'scores': scores,
            'evaluated_at': _now_iso(),
        }
        if failed:
            record['status'] = 'failed'
        storage.append_evaluation(task_id, teacher_id, judge_id, record)

    # --- Process per_factor results: aggregate scores per response ---
    factor_scores: dict[tuple, dict[str, str]] = defaultdict(dict)
    factor_meta: dict[tuple, dict] = {}

    for batch_key, response_text in results.items():
        if batch_key not in pending_factor:
            continue
        info = pending_factor[batch_key]
        resp_key = (info['task_id'], info['teacher_id'], info['judge_id'], info['resp']['id'])
        word = parse_word_text(response_text) if response_text else 'Low'
        factor_scores[resp_key][info['factor']] = word
        if resp_key not in factor_meta:
            factor_meta[resp_key] = info

    n_factor_ok = 0
    for resp_key, scores in factor_scores.items():
        info = factor_meta[resp_key]
        task_id = info['task_id']
        teacher_id = info['teacher_id']
        judge_id = info['judge_id']
        resp = info['resp']

        eval_id = f"{resp['id']}__{judge_id}"
        record = {
            'id': eval_id,
            'response_id': resp['id'],
            'datapoint_id': info['dp_id'],
            'task_id': task_id,
            'teacher_model_id': teacher_id,
            'judge_model_id': judge_id,
            'scores': scores,
            'evaluated_at': _now_iso(),
        }
        storage.append_evaluation(task_id, teacher_id, judge_id, record)
        n_factor_ok += 1

    n_ok = n_single_ok + n_factor_ok
    n_fail = n_single_fail  # per_factor failures map to 'Low', not logged separately
    logger.info(
        f"Phase 5: {iface_name} batch written — "
        f"{n_ok} eval records written, {n_fail} single-mode failures"
    )


# ---------------------------------------------------------------------------
# Sequential path (HF judges or non-batch network judges)
# ---------------------------------------------------------------------------


def _evaluate(
    task: TaskConfig,
    teacher: ModelConfig,
    judge: ModelConfig,
    storage: ExperimentStorage,
    logger: RunLogger,
    pool: ModelPool,
    quota: QuotaTracker,
    phase_mode: str,
) -> None:
    task_id = task.name
    teacher_id = teacher.name
    judge_id = judge.name
    label = f"(task='{task_id}', teacher='{teacher_id}', judge='{judge_id}')"

    # Keep mode: skip entirely
    if phase_mode == 'Keep':
        logger.info(f"Phase 5: {label} — Keep mode, skipping")
        return

    # Model mode: skip if evaluation file already exists for this judge
    if phase_mode == 'Model' and storage.evaluation_file_exists(task_id, teacher_id, judge_id):
        logger.info(f"Phase 5: {label} — Model mode, file exists, skipping")
        return

    # Load rubric and datapoints index (REQ-7.5)
    # LLM judges only score non-metric factors; metric factors are handled
    # by metric judges in _evaluate_metric().
    full_rubric = storage.read_rubric(task_id)
    rubric, _ = split_rubric(full_rubric)
    datapoints_index = storage.index_datapoints(task_id, teacher_id)

    # Collect all response files for this (task, teacher) pair
    all_responses: list[dict] = []
    for resp_path in storage.iter_response_files(task_id, teacher_id):
        for line in resp_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                all_responses.append(json.loads(line))

    if not all_responses:
        logger.warning(f"Phase 5: {label} — no responses found, skipping")
        return

    # Already-evaluated response IDs (for Extend mode)
    evaluated_resp_ids = (
        storage.get_evaluated_response_ids(task_id, teacher_id, judge_id)
        if phase_mode == 'Extend' else set()
    )

    # Extend mode: skip only if there are no new rubric factors AND no new responses.
    new_factors: dict[str, str] = {}
    if phase_mode == 'Extend':
        existing_evals = storage.read_evaluations(task_id, teacher_id, judge_id)
        evaluated_factors: set[str] = set()
        for ev in existing_evals:
            evaluated_factors.update(ev.get('scores', {}).keys())
        new_factors = {k: v for k, v in rubric.items() if k not in evaluated_factors}
        new_responses = [r for r in all_responses if r['id'] not in evaluated_resp_ids]
        if not new_factors and not new_responses:
            logger.info(
                f"Phase 5: {label} — Extend mode, no new rubric factors and "
                "no new responses, skipping"
            )
            return
        if new_factors:
            rubric_to_use = new_factors
            logger.info(
                f"Phase 5: {label} — Extend mode, evaluating {len(new_factors)} "
                f"new factor(s) across {len(all_responses)} response(s)"
            )
        else:
            rubric_to_use = rubric
            logger.info(
                f"Phase 5: {label} — Extend mode, {len(new_responses)} new "
                "response(s) to evaluate (no new rubric factors)"
            )
    else:
        rubric_to_use = rubric

    if quota.is_exhausted(judge_id):
        logger.warning(
            f"Quota exhausted for model {judge_id} in phase evaluation; skipping {label}"
        )
        return

    iface = pool.get(judge)
    params = judge.get_parameters_for_role('judge')

    logger.info(
        f"Phase 5: {label} — evaluating {len(all_responses)} responses "
        f"with {len(rubric_to_use)} factor(s) in '{task.evaluation_mode}' mode"
    )

    for resp in all_responses:
        # Extend mode: always skip responses that already have evaluation records.
        # New rubric factors are applied only to responses that have never been
        # evaluated at all.
        if phase_mode == 'Extend' and resp['id'] in evaluated_resp_ids:
            continue

        if quota.is_exhausted(judge_id):
            logger.warning(
                f"Quota exhausted for model {judge_id} in phase evaluation"
            )
            break

        # Resolve reference_response from datapoints index (REQ-7.5)
        dp_id = resp['datapoint_id']
        if dp_id not in datapoints_index:
            logger.error(
                f"Phase 5: datapoint_id '{dp_id}' not found in datapoints index; "
                f"skipping response '{resp['id']}'"
            )
            continue

        dp = datapoints_index[dp_id]
        reference_response = dp['reference_response']
        target_attrs_str = json.dumps(dp.get('sampled_target_attributes', {}))

        eval_id = f"{resp['id']}__{judge_id}"
        try:
            scores = _score_response(
                task=task,
                rubric=rubric_to_use,
                response=resp,
                reference_response=reference_response,
                target_attrs_str=target_attrs_str,
                judge_id=judge_id,
                iface=iface,
                params=params,
                pool=pool,
                quota=quota,
            )
            record: dict = {
                'id': eval_id,
                'response_id': resp['id'],
                'datapoint_id': dp_id,
                'task_id': task_id,
                'teacher_model_id': teacher_id,
                'judge_model_id': judge_id,
                'scores': scores,
                'evaluated_at': _now_iso(),
            }
        except Exception as exc:
            logger.error(
                f"Phase 5: scoring failed for response '{resp['id']}' "
                f"(judge='{judge_id}'): {exc}"
            )
            record = {
                'id': eval_id,
                'response_id': resp['id'],
                'datapoint_id': dp_id,
                'task_id': task_id,
                'teacher_model_id': teacher_id,
                'judge_model_id': judge_id,
                'scores': {},
                'status': 'failed',
                'evaluated_at': _now_iso(),
            }
        storage.append_evaluation(task_id, teacher_id, judge_id, record)

    logger.info(f"Phase 5: {label} — done")


def _score_response(
    task: TaskConfig,
    rubric: dict[str, str],
    response: dict,
    reference_response: str,
    target_attrs_str: str,
    judge_id: str,
    iface,
    params: dict,
    pool: ModelPool,
    quota: QuotaTracker,
) -> dict[str, str]:
    """Score a single response against the rubric using the configured evaluation_mode."""
    common_vars = {
        'task_description': task.description,
        'output_description': task.output_description,
        'input': response['input'],
        'target_attributes': target_attrs_str,
        'reference_response': reference_response,
        'response': response['response'],
    }

    if task.evaluation_mode == 'single':
        rubric_text = '\n'.join(
            f'- {factor}: {desc}' for factor, desc in rubric.items()
        )
        prompt = get_prompt(
            'evaluate_single',
            task.prompt_library,
            judge_id,
            {**common_vars, 'rubric': rubric_text},
        )
        result = call_llm_json(iface, prompt, params)
        quota.consume(judge_id)
        # Validate and normalise scores
        scores: dict[str, str] = {}
        for factor in rubric:
            val = str(result.get(factor, '')).strip()
            if val not in ('High', 'Medium', 'Low'):
                val = 'Low'  # fallback for malformed responses
            scores[factor] = val
        return scores

    else:  # per_factor
        scores = {}
        for factor, desc in rubric.items():
            if quota.is_exhausted(judge_id):
                break
            prompt = get_prompt(
                'evaluate_per_factor',
                task.prompt_library,
                judge_id,
                {
                    **common_vars,
                    'rubric_factor_name': factor,
                    'rubric_factor_description': desc,
                },
            )
            word = call_llm_word(iface, prompt, params)
            quota.consume(judge_id)
            scores[factor] = word
        return scores


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
