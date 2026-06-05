"""Phase 4 — Response Collection (REQ-7.1, §4, §6.2.5)."""
from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from ..config import CoEvalConfig, TaskConfig, ModelConfig
from ..exceptions import PartialPhaseFailure
from ..interfaces import ModelPool, create_batch_runner
from ..interfaces.cost_estimator import count_tokens_approx
from ..logger import RunLogger
from ..prompts import get_prompt
from ..storage import ExperimentStorage
from .utils import QuotaTracker

# Maximum concurrent workers for non-batch, network-bound student calls.
# Override with COEVAL_MAX_WORKERS (set to 1-3 for rate-limited free-tier models).
_MAX_WORKERS = int(os.environ.get('COEVAL_MAX_WORKERS', '10'))
# Separator for batch keys — ASCII NUL never appears in task/model/datapoint IDs
_KEY_SEP = "\x00"


def run_phase4(
    cfg: CoEvalConfig,
    storage: ExperimentStorage,
    logger: RunLogger,
    pool: ModelPool,
    quota: QuotaTracker,
    phase_mode: str,
    only_models: set[str] | None = None,
) -> None:
    """Execute Phase 4 for all (task, teacher, student) triples.

    Students are grouped by interface:
      - Network interfaces (OpenAI, Anthropic, Gemini): when
        ``experiment.batch.<interface>.response_collection: true`` is set, all
        requests are submitted in one batch job per interface.  Otherwise requests
        run concurrently (ThreadPoolExecutor, up to _MAX_WORKERS).
      - HuggingFace: always sequential (GPU-bound).

    If only_models is provided, only students in that set are processed.
    """
    teachers = cfg.get_models_by_role('teacher')
    all_students = cfg.get_models_by_role('student')
    students = (
        [s for s in all_students if s.name in only_models]
        if only_models is not None else all_students
    )

    # Split students: network by interface vs HuggingFace
    by_iface: dict[str, list[ModelConfig]] = defaultdict(list)
    hf_students: list[ModelConfig] = []
    for s in students:
        if s.interface == 'huggingface':
            hf_students.append(s)
        else:
            by_iface[s.interface].append(s)

    errors: list[str] = []

    # --- Network-based students: batch or concurrent sequential ---
    for iface_name, iface_students in by_iface.items():
        if cfg.use_batch(iface_name, 'response_collection'):
            try:
                _collect_batch_responses(
                    iface_name, iface_students, cfg.tasks, teachers,
                    storage, logger, quota, phase_mode,
                )
            except Exception as exc:
                msg = (
                    f"Phase 4: {iface_name} batch response collection failed: {exc}"
                )
                logger.error(msg)
                errors.append(msg)
        else:
            triples = [
                (task, teacher, student)
                for task in cfg.tasks
                for teacher in teachers
                for student in iface_students
            ]
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _collect_responses,
                        task, teacher, student, storage, logger, pool, quota, phase_mode,
                    ): (task, teacher, student)
                    for task, teacher, student in triples
                }
                for future in as_completed(futures):
                    task, teacher, student = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        msg = (
                            f"Phase 4: response collection failed for "
                            f"(task='{task.name}', teacher='{teacher.name}', "
                            f"student='{student.name}'): {exc}"
                        )
                        logger.error(msg)
                        errors.append(msg)

    # --- HuggingFace students: sequential (GPU-bound) ---
    for task in cfg.tasks:
        for teacher in teachers:
            for student in hf_students:
                try:
                    _collect_responses(
                        task, teacher, student, storage, logger, pool, quota, phase_mode,
                    )
                except Exception as exc:
                    msg = (
                        f"Phase 4: response collection failed for "
                        f"(task='{task.name}', teacher='{teacher.name}', "
                        f"student='{student.name}'): {exc}"
                    )
                    logger.error(msg)
                    errors.append(msg)

    if errors:
        n_total = sum(
            1
            for _task in cfg.tasks
            for _teacher in teachers
            for _student in students
        )
        raise PartialPhaseFailure(
            n_failures=len(errors),
            n_successes=n_total - len(errors),
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Batch path (network interfaces with batch enabled)
# ---------------------------------------------------------------------------


def _collect_batch_responses(
    iface_name: str,
    iface_students: list[ModelConfig],
    tasks: list[TaskConfig],
    teachers: list[ModelConfig],
    storage: ExperimentStorage,
    logger: RunLogger,
    quota: QuotaTracker,
    phase_mode: str,
) -> None:
    """Collect responses for all students of one interface via a single batch job.

    All students must share the same interface name (``iface_name``).  If
    students use different API keys, the first student's key is used; in
    practice all students of the same provider share one key.

    Quota is consumed at queue time to enforce limits before submission.
    Failed requests are written with ``status='failed'`` and will be retried
    automatically on the next ``--continue`` run.
    """
    access_key = iface_students[0].access_key or None
    runner = create_batch_runner(iface_name, access_key=access_key)

    # batch_key → (task_id, teacher_id, student_id, datapoint)
    pending: dict[str, tuple[str, str, str, dict]] = {}

    for task in tasks:
        for teacher in teachers:
            for student in iface_students:
                task_id = task.name
                teacher_id = teacher.name
                student_id = student.name
                label = (
                    f"(task='{task_id}', teacher='{teacher_id}', "
                    f"student='{student_id}')"
                )

                if phase_mode == 'Keep':
                    logger.info(f"Phase 4: {label} — Keep mode, skipping")
                    continue

                if phase_mode == 'Model' and storage.response_file_exists(
                    task_id, teacher_id, student_id
                ):
                    logger.info(f"Phase 4: {label} — Model mode, file exists, skipping")
                    continue

                datapoints = storage.read_datapoints(task_id, teacher_id)
                if not datapoints:
                    logger.warning(f"Phase 4: {label} — no datapoints found, skipping")
                    continue

                if phase_mode == 'Extend':
                    responded_ids = storage.get_responded_datapoint_ids(
                        task_id, teacher_id, student_id
                    )
                    datapoints = [
                        dp for dp in datapoints if dp['id'] not in responded_ids
                    ]
                    if not datapoints:
                        logger.info(
                            f"Phase 4: {label} — Extend mode, all datapoints "
                            "already responded, skipping"
                        )
                        continue
                    logger.info(
                        f"Phase 4: {label} — Extend mode, "
                        f"{len(datapoints)} remaining datapoint(s)"
                    )

                if quota.is_exhausted(student_id):
                    logger.warning(
                        f"Quota exhausted for model {student_id} in phase "
                        f"response_collection; skipping {label}"
                    )
                    continue

                params = student.get_parameters_for_role('student')
                logger.info(
                    f"Phase 4: {label} — queuing {len(datapoints)} response(s)"
                )

                for dp in datapoints:
                    if quota.is_exhausted(student_id):
                        logger.warning(
                            f"Quota exhausted for model {student_id} in phase "
                            "response_collection"
                        )
                        break

                    prompt = get_prompt(
                        'test',
                        task.prompt_library,
                        student_id,
                        {
                            'input': dp['prompt'],
                            'task_description': task.description,
                            'output_description': task.output_description,
                        },
                    )

                    batch_key = _KEY_SEP.join([task_id, teacher_id, student_id, dp['id']])
                    runner.add(batch_key, prompt, params)
                    quota.consume(student_id)
                    pending[batch_key] = (task_id, teacher_id, student_id, dp)

    if not pending:
        logger.info(
            f"Phase 4: no {iface_name} student requests pending; "
            "skipping batch submission"
        )
        return

    logger.info(
        f"Phase 4: submitting {iface_name} batch of {len(runner)} request(s)"
    )
    results = runner.run(
        description="Phase 4 student responses",
        logger=logger,
        storage=storage,
        phase='response_collection',
    )

    n_ok = n_fail = 0
    for batch_key, response_text in results.items():
        if batch_key not in pending:
            continue
        task_id, teacher_id, student_id, dp = pending[batch_key]
        response_id = f"{dp['id']}__{student_id}"
        record: dict = {
            'id': response_id,
            'datapoint_id': dp['id'],
            'task_id': task_id,
            'teacher_model_id': teacher_id,
            'student_model_id': student_id,
            'input': dp['prompt'],
            'response': response_text,
            'token_count': count_tokens_approx(response_text),
            'generated_at': _now_iso(),
        }
        if not response_text:
            record['status'] = 'failed'
            n_fail += 1
        else:
            n_ok += 1
        storage.append_response(task_id, teacher_id, student_id, record)

    logger.info(
        f"Phase 4: {iface_name} batch written — {n_ok} succeeded, {n_fail} failed"
    )
    if n_fail:
        logger.warning(
            f"Phase 4: {n_fail} failed response(s) written with status='failed'; "
            "they will be retried on the next --continue run"
        )


# ---------------------------------------------------------------------------
# Sequential path (HF students or non-batch network students)
# ---------------------------------------------------------------------------


def _collect_responses(
    task: TaskConfig,
    teacher: ModelConfig,
    student: ModelConfig,
    storage: ExperimentStorage,
    logger: RunLogger,
    pool: ModelPool,
    quota: QuotaTracker,
    phase_mode: str,
) -> None:
    task_id = task.name
    teacher_id = teacher.name
    student_id = student.name
    label = f"(task='{task_id}', teacher='{teacher_id}', student='{student_id}')"

    # Keep mode: skip entirely
    if phase_mode == 'Keep':
        logger.info(f"Phase 4: {label} — Keep mode, skipping")
        return

    # Model mode: skip if responses file already exists for this student
    if phase_mode == 'Model' and storage.response_file_exists(task_id, teacher_id, student_id):
        logger.info(f"Phase 4: {label} — Model mode, file exists, skipping")
        return

    datapoints = storage.read_datapoints(task_id, teacher_id)
    if not datapoints:
        logger.warning(f"Phase 4: {label} — no datapoints found, skipping")
        return

    # Extend mode: skip datapoints that already have a response
    if phase_mode == 'Extend':
        responded_ids = storage.get_responded_datapoint_ids(task_id, teacher_id, student_id)
        datapoints = [dp for dp in datapoints if dp['id'] not in responded_ids]
        if not datapoints:
            logger.info(
                f"Phase 4: {label} — Extend mode, all datapoints already responded, skipping"
            )
            return
        logger.info(
            f"Phase 4: {label} — Extend mode, {len(datapoints)} remaining datapoints"
        )

    if quota.is_exhausted(student_id):
        logger.warning(
            f"Quota exhausted for model {student_id} in phase response_collection; "
            f"skipping {label}"
        )
        return

    iface = pool.get(student)
    params = student.get_parameters_for_role('student')

    logger.info(f"Phase 4: {label} — collecting {len(datapoints)} responses")

    for dp in datapoints:
        if quota.is_exhausted(student_id):
            logger.warning(
                f"Quota exhausted for model {student_id} in phase response_collection"
            )
            break

        prompt = get_prompt(
            'test',
            task.prompt_library,
            student_id,
            {
                'input': dp['prompt'],
                'task_description': task.description,
                'output_description': task.output_description,
            },
        )

        response_text = iface.generate(prompt, params)
        quota.consume(student_id)

        response_id = f"{dp['id']}__{student_id}"
        record = {
            'id': response_id,
            'datapoint_id': dp['id'],
            'task_id': task_id,
            'teacher_model_id': teacher_id,
            'student_model_id': student_id,
            'input': dp['prompt'],
            'response': response_text,
            'token_count': count_tokens_approx(response_text),
            'generated_at': _now_iso(),
        }
        storage.append_response(task_id, teacher_id, student_id, record)

    logger.info(f"Phase 4: {label} — done")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
