"""Phase 3 — Data Generation (REQ-7.1, REQ-7.4, §4, §5.3.4, §6.2.4)."""
from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from ..config import CoEvalConfig, TaskConfig, ModelConfig, SamplingConfig
from ..interfaces import ModelPool, create_batch_runner
from ..logger import RunLogger
from ..prompts import get_prompt
from ..storage import ExperimentStorage
from .utils import call_llm_json, extract_prompt_response, QuotaTracker, parse_json_text

# Maximum concurrent workers for network-based (non-HF) non-batch execution
_MAX_WORKERS = 10
# Separator for batch keys — ASCII NUL never appears in task/model/datapoint IDs
_KEY_SEP = "\x00"


def run_phase3(
    cfg: CoEvalConfig,
    storage: ExperimentStorage,
    logger: RunLogger,
    pool: ModelPool,
    quota: QuotaTracker,
    phase_mode: str,
    only_models: set[str] | None = None,
) -> None:
    """Execute Phase 3 for all (task, teacher) pairs.

    Teachers are grouped by interface:
      - Network interfaces (OpenAI, Anthropic, Gemini): use batch runner when
        ``experiment.batch.<interface>.data_generation: true`` is set; otherwise
        fall back to concurrent sequential calls.
      - HuggingFace: always sequential (GPU-bound).

    Individual teacher failures are logged but do not abort the phase as long as
    at least one datapoint was generated per task.  Zero datapoints for any task
    raises ``RuntimeError`` to prevent downstream phases from receiving empty data.

    If only_models is provided, only teachers in that set are processed.
    """
    all_teachers = cfg.get_models_by_role('teacher')
    teachers = (
        [t for t in all_teachers if t.name in only_models]
        if only_models is not None else all_teachers
    )

    # Benchmark teachers have data pre-written by `coeval ingest`; skip them entirely
    benchmark_teachers = [t for t in teachers if t.interface == 'benchmark']
    teachers = [t for t in teachers if t.interface != 'benchmark']
    if benchmark_teachers:
        bnames = ', '.join(t.name for t in benchmark_teachers)
        logger.info(
            f"Phase 3: skipping {len(benchmark_teachers)} benchmark teacher(s) "
            f"(data already ingested): {bnames}"
        )
    generation_retries = cfg.experiment.generation_retries
    errors: list[str] = []

    # Split teachers: network (OAI/Anthropic/Gemini) by interface vs HuggingFace
    by_iface: dict[str, list[ModelConfig]] = defaultdict(list)
    hf_teachers: list[ModelConfig] = []
    for t in teachers:
        if t.interface == 'huggingface':
            hf_teachers.append(t)
        else:
            by_iface[t.interface].append(t)

    # --- Network-based teachers: batch or concurrent sequential ---
    for iface_name, iface_teachers in by_iface.items():
        if cfg.use_batch(iface_name, 'data_generation'):
            try:
                _generate_batch_datapoints(
                    iface_name, iface_teachers, cfg.tasks,
                    storage, logger, quota, phase_mode, generation_retries,
                )
            except Exception as exc:
                msg = (
                    f"Phase 3: {iface_name} batch data generation failed: {exc}"
                )
                logger.error(msg)
                errors.append(msg)
        else:
            # Concurrent sequential for network-bound teachers
            pairs = [
                (task, teacher)
                for task in cfg.tasks
                for teacher in iface_teachers
            ]
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _generate_datapoints,
                        task, teacher, storage, logger, pool, quota,
                        phase_mode, generation_retries,
                    ): (task, teacher)
                    for task, teacher in pairs
                }
                for future in as_completed(futures):
                    task, teacher = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        msg = (
                            f"Phase 3: data generation failed for "
                            f"(task='{task.name}', teacher='{teacher.name}'): {exc}"
                        )
                        logger.error(msg)
                        errors.append(msg)

    # --- HuggingFace teachers: sequential (GPU-bound) ---
    for task in cfg.tasks:
        for teacher in hf_teachers:
            try:
                _generate_datapoints(
                    task, teacher, storage, logger, pool, quota,
                    phase_mode, generation_retries,
                )
            except Exception as exc:
                msg = (
                    f"Phase 3: data generation failed for "
                    f"(task='{task.name}', teacher='{teacher.name}'): {exc}"
                )
                logger.error(msg)
                errors.append(msg)

    # All active teachers (including benchmark teachers whose data was pre-ingested)
    all_active = teachers + benchmark_teachers
    if not all_active:
        return  # nothing active — nothing to verify

    # Verify at least one datapoint per task across all active teachers
    for task in cfg.tasks:
        total_for_task = sum(
            storage.count_datapoints(task.name, t.name) for t in all_active
        )
        if total_for_task == 0:
            raise RuntimeError(
                f"Phase 3: no datapoints found for task '{task.name}' "
                f"— all {len(all_active)} teacher(s) failed or were not yet ingested"
            )

    if errors:
        logger.warning(
            f"Phase 3 completed with {len(errors)} partial failure(s) "
            f"(pipeline continues with available data):\n" + "\n".join(errors)
        )


# ---------------------------------------------------------------------------
# Batch path (network interfaces with batch enabled)
# ---------------------------------------------------------------------------


def _generate_batch_datapoints(
    iface_name: str,
    teachers: list[ModelConfig],
    tasks: list[TaskConfig],
    storage: ExperimentStorage,
    logger: RunLogger,
    quota: QuotaTracker,
    phase_mode: str,
    generation_retries: int,
) -> None:
    """Generate datapoints for all teachers of one interface via a single batch job.

    Attribute sampling is performed **per item** before submission, so each
    datapoint in the batch gets its own independently-sampled target and nuanced
    attributes.  The sampled attributes are stored in ``pending`` and correctly
    associated with results after the batch completes.

    Note: unlike the sequential path, this function does **not** retry individual
    items on JSON parse failure — failed items are logged and skipped.  Users can
    re-run with ``--continue`` (Extend mode) to regenerate missing items.
    """
    access_key = teachers[0].access_key or None
    runner = create_batch_runner(iface_name, access_key=access_key)

    # batch_key → (task, teacher, seq, sampled_target, sampled_nuanced)
    pending: dict[str, tuple] = {}

    for task in tasks:
        # Load attribute maps once per task (shared across teachers)
        target_attrs = storage.read_target_attrs(task.name)
        nuanced_attrs = storage.read_nuanced_attrs(task.name)

        for teacher in teachers:
            task_id = task.name
            teacher_id = teacher.name
            total = task.sampling.total
            label = f"(task='{task_id}', teacher='{teacher_id}')"

            if phase_mode == 'Keep':
                logger.info(f"Phase 3: {label} — Keep mode, skipping")
                continue

            if phase_mode == 'Model' and storage.datapoints_path(task_id, teacher_id).exists():
                logger.info(f"Phase 3: {label} — Model mode, file exists, skipping")
                continue

            existing_count = storage.count_datapoints(task_id, teacher_id)
            if phase_mode == 'Extend':
                if existing_count >= total:
                    logger.info(
                        f"Phase 3: {label} — Extend mode, already have "
                        f"{existing_count}/{total} items, skipping"
                    )
                    continue
                to_generate = total - existing_count
                seq_start = existing_count + 1
                logger.info(
                    f"Phase 3: {label} — Extend mode, generating "
                    f"{to_generate} more items (have {existing_count})"
                )
            else:
                to_generate = total
                seq_start = 1

            if quota.is_exhausted(teacher_id):
                logger.warning(
                    f"Quota exhausted for model {teacher_id} in phase "
                    f"data_generation; skipping {label}"
                )
                continue

            params = teacher.get_parameters_for_role('teacher')
            logger.info(f"Phase 3: {label} — queuing {to_generate} datapoint(s)")

            # Aligned with paper v2 methodology - Algorithm 1 (§3.5): use
            # stratified cycling for target attributes when spec is 'all'.
            if task.sampling.target == 'all' and target_attrs:
                target_sequence = _make_target_cycle(
                    target_attrs, to_generate, logger=logger, label=label
                )
            else:
                target_sequence = None

            for i in range(to_generate):
                if quota.is_exhausted(teacher_id):
                    logger.warning(
                        f"Quota exhausted for model {teacher_id} after {i} items "
                        "in phase data_generation"
                    )
                    break

                seq = seq_start + i
                # Use pre-built stratified sequence when available, else random
                if target_sequence is not None:
                    sampled_target = target_sequence[i]
                else:
                    sampled_target = _sample_attrs(target_attrs, task.sampling.target)
                sampled_nuanced = _sample_attrs(nuanced_attrs, task.sampling.nuance)

                prompt = get_prompt(
                    'sample',
                    task.prompt_library,
                    teacher_id,
                    {
                        'task_description': task.description,
                        'output_description': task.output_description,
                        'target_attributes': json.dumps(sampled_target),
                        'nuanced_attributes': json.dumps(sampled_nuanced),
                    },
                )

                batch_key = _KEY_SEP.join([task_id, teacher_id, str(seq)])
                runner.add(batch_key, prompt, params)
                quota.consume(teacher_id)
                pending[batch_key] = (task, teacher, seq, sampled_target, sampled_nuanced)

    if not pending:
        logger.info(
            f"Phase 3: no {iface_name} teacher requests pending; "
            "skipping batch submission"
        )
        return

    logger.info(
        f"Phase 3: submitting {iface_name} batch of {len(runner)} request(s)"
    )
    results = runner.run(
        description="Phase 3 teacher data generation",
        logger=logger,
        storage=storage,
        phase='data_generation',
    )

    n_ok = n_fail = 0
    for batch_key, response_text in results.items():
        if batch_key not in pending:
            continue
        task, teacher, seq, sampled_target, sampled_nuanced = pending[batch_key]
        task_id = task.name
        teacher_id = teacher.name

        if not response_text:
            logger.warning(
                f"Phase 3: request '{batch_key}' failed in batch; skipping"
            )
            n_fail += 1
            continue

        try:
            result = parse_json_text(response_text)
            prompt_text, ref_response = extract_prompt_response(result)
        except Exception as exc:
            logger.warning(
                f"Phase 3: request '{batch_key}' JSON parse failed — "
                f"skipping ({exc})"
            )
            n_fail += 1
            continue

        dp_id = f"{task_id}__{teacher_id}__{seq:05d}"
        record: dict = {
            'id': dp_id,
            'task_id': task_id,
            'teacher_model_id': teacher_id,
            'sampled_target_attributes': sampled_target,
            'prompt': prompt_text,
            'reference_response': ref_response,
            'generated_at': _now_iso(),
        }
        if task.store_nuanced:
            record['sampled_nuanced_attributes'] = sampled_nuanced

        storage.append_datapoint(task_id, teacher_id, record)
        n_ok += 1

    logger.info(
        f"Phase 3: {iface_name} batch written — "
        f"{n_ok} succeeded, {n_fail} failed/skipped"
    )


# ---------------------------------------------------------------------------
# Sequential path (HF teachers or non-batch network teachers)
# ---------------------------------------------------------------------------


def _generate_datapoints(
    task: TaskConfig,
    teacher: ModelConfig,
    storage: ExperimentStorage,
    logger: RunLogger,
    pool: ModelPool,
    quota: QuotaTracker,
    phase_mode: str,
    generation_retries: int = 2,
) -> None:
    task_id = task.name
    teacher_id = teacher.name
    total = task.sampling.total
    label = f"(task='{task_id}', teacher='{teacher_id}')"

    # Keep mode: skip entirely
    if phase_mode == 'Keep':
        logger.info(f"Phase 3: {label} — Keep mode, skipping")
        return

    # Model mode: skip if this teacher already has a datapoints file
    if phase_mode == 'Model' and storage.datapoints_path(task_id, teacher_id).exists():
        logger.info(f"Phase 3: {label} — Model mode, file exists, skipping")
        return

    # Extend mode: count existing items and only generate the gap
    existing_count = storage.count_datapoints(task_id, teacher_id)
    if phase_mode == 'Extend':
        if existing_count >= total:
            logger.info(
                f"Phase 3: {label} — Extend mode, already have "
                f"{existing_count}/{total} items, skipping"
            )
            return
        to_generate = total - existing_count
        seq_start = existing_count + 1
        logger.info(
            f"Phase 3: {label} — Extend mode, generating {to_generate} more "
            f"items (have {existing_count})"
        )
    else:
        # New mode
        to_generate = total
        seq_start = 1

    if quota.is_exhausted(teacher_id):
        logger.warning(
            f"Quota exhausted for model {teacher_id} in phase data_generation; "
            f"skipping {label}"
        )
        return

    # Load attribute maps from Phase 1
    target_attrs = storage.read_target_attrs(task_id)
    nuanced_attrs = storage.read_nuanced_attrs(task_id)

    # Aligned with paper v2 methodology - Algorithm 1 (§3.5): pre-build the
    # ordered sequence of target-attribute assignments using stratified cycling.
    # When target_spec == 'all', use full Cartesian cycling; otherwise fall back
    # to per-item random sampling for non-exhaustive target specs.
    if task.sampling.target == 'all' and target_attrs:
        target_sequence = _make_target_cycle(
            target_attrs, to_generate, logger=logger, label=label
        )
    else:
        target_sequence = None  # use per-item _sample_attrs below

    iface = pool.get(teacher)
    params = teacher.get_parameters_for_role('teacher')

    logger.info(f"Phase 3: {label} — generating {to_generate} datapoints")

    skipped = 0
    for i in range(to_generate):
        if quota.is_exhausted(teacher_id):
            logger.warning(
                f"Quota exhausted for model {teacher_id} after {i} items "
                "in phase data_generation"
            )
            break

        seq = seq_start + i
        # Use pre-built stratified sequence when available, else random sampling
        if target_sequence is not None:
            sampled_target = target_sequence[i]
        else:
            sampled_target = _sample_attrs(target_attrs, task.sampling.target)
        sampled_nuanced = _sample_attrs(nuanced_attrs, task.sampling.nuance)

        prompt = get_prompt(
            'sample',
            task.prompt_library,
            teacher_id,
            {
                'task_description': task.description,
                'output_description': task.output_description,
                'target_attributes': json.dumps(sampled_target),
                'nuanced_attributes': json.dumps(sampled_nuanced),
            },
        )

        # Per-datapoint retry loop
        max_attempts = 1 + generation_retries
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            quota_consumed = False
            try:
                result = call_llm_json(iface, prompt, params)
                quota.consume(teacher_id)
                quota_consumed = True
                prompt_text, response_text = extract_prompt_response(result)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if not quota_consumed:
                    quota.consume(teacher_id)
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Phase 3: {label} datapoint {seq} attempt {attempt + 1}/"
                        f"{max_attempts} failed ({exc}), retrying"
                    )

        if last_exc is not None:
            logger.warning(
                f"Phase 3: {label} datapoint {seq} failed after {max_attempts} "
                f"attempt(s) — skipping ({last_exc})"
            )
            skipped += 1
            continue

        dp_id = f"{task_id}__{teacher_id}__{seq:05d}"
        record: dict = {
            'id': dp_id,
            'task_id': task_id,
            'teacher_model_id': teacher_id,
            'sampled_target_attributes': sampled_target,
            'prompt': prompt_text,
            'reference_response': response_text,
            'generated_at': _now_iso(),
        }
        if task.store_nuanced:
            record['sampled_nuanced_attributes'] = sampled_nuanced

        storage.append_datapoint(task_id, teacher_id, record)

    final_count = storage.count_datapoints(task_id, teacher_id)
    if skipped:
        logger.warning(
            f"Phase 3: {label} — done with {skipped} skipped datapoint(s), "
            f"{final_count} total in file"
        )
    else:
        logger.info(
            f"Phase 3: {label} — done, {final_count} total datapoints in file"
        )


# ---------------------------------------------------------------------------
# Sampling algorithm (REQ-5.3.4)
# ---------------------------------------------------------------------------


def _make_target_cycle(
    target_attrs: dict[str, list],
    total: int,
    logger: 'RunLogger | None' = None,
    label: str = '',
) -> list[dict[str, str]]:
    """Build the ordered list of target-attribute assignments for Algorithm 1.

    Aligned with paper v2 methodology - Algorithm 1 (§3.5):
    - Compute perms = CartesianProduct(A_target.values()).
    - If N >= |perms|: use itertools.cycle(perms) — each combination appears
      at least floor(N/|perms|) times per teacher.
    - If N < |perms|: sample without replacement from the permutation space,
      maximising coverage by drawing distinct combinations first.

    Returns a list of length `total`, one target-attr dict per datapoint slot.
    """
    if not target_attrs:
        return [{} for _ in range(total)]

    keys = list(target_attrs.keys())
    value_lists = [target_attrs[k] for k in keys]
    all_perms = list(itertools.product(*value_lists))

    if not all_perms:
        return [{} for _ in range(total)]

    n_perms = len(all_perms)

    if total < n_perms:
        # N < |perms|: sample without replacement (Algorithm 1, line 2a note)
        if logger is not None:
            logger.warning(
                f"Phase 3: {label} N={total} < |perms|={n_perms}: "
                "not all attribute combinations will be sampled. "
                "Sampling without replacement for maximal coverage."
            )
        chosen = random.sample(all_perms, total)
        return [dict(zip(keys, combo)) for combo in chosen]
    else:
        # N >= |perms|: use circular iterator (Algorithm 1, line 4)
        cycle_iter = itertools.cycle(all_perms)
        return [dict(zip(keys, next(cycle_iter))) for _ in range(total)]


def _sample_attrs(attr_map: dict[str, list], target_spec) -> dict[str, str]:
    """Sample a subset of attribute key-value pairs per the spec algorithm.

    Used for nuanced attributes (random per-item sampling) and for cases
    where target_attrs is not provided as a full dict to _make_target_cycle.
    """
    if not attr_map:
        return {}

    attr_names = list(attr_map.keys())

    if target_spec == 'all':
        selected_names = attr_names
    else:
        # Accept [min, max], or a single-element [n] (treated as [n, n]), or a
        # bare int n. Tolerating the short forms keeps existing configs working.
        spec = target_spec if isinstance(target_spec, (list, tuple)) else [target_spec]
        lo = int(spec[0])
        hi = int(spec[1]) if len(spec) > 1 else lo
        n = max(0, min(random.randint(lo, hi), len(attr_names)))
        selected_names = random.sample(attr_names, n)

    return {
        name: random.choice(attr_map[name])
        for name in selected_names
        if attr_map[name]
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
