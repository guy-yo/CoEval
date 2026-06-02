"""Config loading, parsing, and validation (rules V-01 through V-17)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME_RE = re.compile(r'^[A-Za-z0-9._-]+$')
_TASK_NAME_RE = re.compile(r'^[A-Za-z0-9_-]+$')
_EXPERIMENT_ID_RE = re.compile(r'^[A-Za-z0-9._-]+$')
_RESERVED_SEP = '__'

VALID_ROLES = {'student', 'teacher', 'judge'}
VALID_INTERFACES = {
    'openai', 'anthropic', 'gemini', 'huggingface',
    'azure_openai', 'azure_ai', 'bedrock', 'vertex', 'openrouter',
    # OpenAI-compatible providers (all use the same REST API pattern)
    'groq', 'deepseek', 'mistral', 'deepinfra', 'cerebras', 'cohere', 'huggingface_api', 'ollama',
    # Virtual interface — benchmark data pre-written by `coeval ingest`; Phase 3 is skipped
    'benchmark',
    # Metric judge — computes deterministic metrics (BERTScore, BLEU, exact_match)
    # without LLM calls; Phase 5 dispatches to runner.metric_judge
    'metric',
}
VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR'}
VALID_EVAL_MODES = {'single', 'per_factor'}
VALID_PHASE_MODES = {'New', 'Keep', 'Extend', 'Model'}
# Phases that support native or pseudo-batch processing
VALID_BATCH_PHASES = {'data_generation', 'response_collection', 'evaluation'}
# Interfaces that support batch runners (HuggingFace is always sequential)
# gemini: concurrent thread-pool runner (pseudo-batch; no additional async discount)
# bedrock: AWS Model Invocation Jobs (true async; ~50% off; requires S3 + IAM role)
# vertex: Vertex AI Batch Prediction Jobs (true async; ~50% off; requires GCS bucket)
# mistral: Mistral Batch API (true async; ~50% off; OpenAI-compatible format)
BATCH_CAPABLE_INTERFACES = {
    'openai', 'anthropic', 'gemini', 'azure_openai', 'bedrock', 'vertex', 'mistral',
}
# Valid probe modes and fail behaviours
VALID_PROBE_MODES = {'disable', 'full', 'resume'}
VALID_PROBE_FAIL_MODES = {'abort', 'warn'}

PHASE_IDS = [
    'attribute_mapping',
    'rubric_mapping',
    'data_generation',
    'response_collection',
    'evaluation',
]
_PHASES_NO_MODEL_MODE = {'attribute_mapping', 'rubric_mapping'}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    target: list[int] | str   # [min, max] or "all"
    nuance: list[int]          # [min, max]
    total: int


@dataclass
class ModelConfig:
    name: str
    interface: str
    parameters: dict[str, Any]
    roles: list[str]
    access_key: str | None = None
    role_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_parameters_for_role(self, role: str) -> dict[str, Any]:
        merged = dict(self.parameters)
        merged.update(self.role_parameters.get(role, {}))
        return merged


@dataclass
class TaskConfig:
    name: str
    description: str
    output_description: str
    target_attributes: dict[str, list[str]] | str   # map | "auto" | "complete"
    nuanced_attributes: dict[str, list[str]] | str  # map | "auto" | "complete"
    sampling: SamplingConfig
    rubric: dict[str, str] | str                    # map | "auto" | "extend"
    target_attributes_seed: dict[str, list[str]] | None = None
    nuanced_attributes_seed: dict[str, list[str]] | None = None
    store_nuanced: bool = False
    evaluation_mode: str = 'single'
    prompt_library: dict[str, str] = field(default_factory=dict)
    # Label accuracy evaluation (classification / information-extraction tasks).
    # List the target attribute keys that serve as ground-truth labels.
    # When non-empty, experiments/label_eval.LabelEvaluator can be applied to
    # Phase 4 student responses after the pipeline without an LLM judge.
    # Example: ["sentiment"] for a sentiment-classification task,
    #          ["entity_type", "entity_value"] for an NER task.
    label_attributes: list[str] = field(default_factory=list)
    # Optional task category for grouping and visual distinction in reports.
    # Typical values: 'benchmark' (real dataset, pre-ingested) | 'synthetic' (LLM-generated)
    # Has no effect on pipeline behaviour; used only for display purposes.
    category: str | None = None


@dataclass
class ExperimentConfig:
    id: str
    storage_folder: str
    resume_from: str | None = None
    phases: dict[str, str] = field(default_factory=dict)
    log_level: str = 'INFO'
    quota: dict[str, dict[str, int]] = field(default_factory=dict)
    generation_retries: int = 2
    # Per-interface, per-phase batch flags.
    # Structure: {interface_name: {phase_id: bool}}
    # Example YAML:
    #   experiment:
    #     batch:
    #       openai:
    #         data_generation: true
    #         response_collection: true
    #         evaluation: true
    #       anthropic:
    #         response_collection: true
    #       gemini:
    #         evaluation: true
    batch: dict[str, dict[str, bool]] = field(default_factory=dict)

    # Model availability probe configuration.
    # probe_mode:
    #   "full"    — probe every model in the config (default).
    #   "resume"  — probe only models needed for phases not yet completed.
    #   "disable" — skip the probe entirely (equivalent to --skip-probe).
    # probe_on_fail:
    #   "abort"   — abort the run if any probed model is unavailable (default).
    #   "warn"    — log a warning and continue; unavailable models may cause
    #               individual phase failures later.
    # estimate_cost:
    #   When True, a cost & time estimate is computed and printed before the
    #   pipeline starts.  Set to False to skip (useful for quick test runs).
    # estimate_samples:
    #   Number of sample LLM calls used for cost/time calibration (default 2).
    #   Set to 0 to use heuristics only (no real API calls during estimation).
    probe_mode: str = 'full'
    probe_on_fail: str = 'abort'
    estimate_cost: bool = False
    estimate_samples: int = 2


@dataclass
class CoEvalConfig:
    models: list[ModelConfig]
    tasks: list[TaskConfig]
    experiment: ExperimentConfig
    _raw: dict = field(default_factory=dict, repr=False, compare=False)
    # Resolved provider credentials from key file (set by load_config)
    _provider_keys: dict = field(default_factory=dict, repr=False, compare=False)

    def get_models_by_role(self, role: str) -> list[ModelConfig]:
        return [m for m in self.models if role in m.roles]

    def get_phase_mode(self, phase_id: str) -> str:
        default = 'Keep' if self.experiment.resume_from else 'New'
        return self.experiment.phases.get(phase_id, default)

    def use_batch(self, interface: str, phase_id: str) -> bool:
        """Return True if the batch runner should be used for this interface+phase.

        Example YAML to enable OAI batch for all three phases::

            experiment:
              batch:
                openai:
                  data_generation: true
                  response_collection: true
                  evaluation: true
        """
        return bool(self.experiment.batch.get(interface, {}).get(phase_id, False))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(path: str, keys_file: str | None = None) -> CoEvalConfig:
    """Load and parse a YAML config file.

    Parameters
    ----------
    path:
        Path to the YAML experiment configuration file.
    keys_file:
        Optional path to a provider key file (YAML).  When provided, its
        ``providers`` block is loaded and stored on ``cfg._provider_keys``
        for later use by ``ModelPool``.  The resolution order is:
        ``keys_file`` arg → ``COEVAL_KEYS_FILE`` env var → ``~/.coeval/keys.yaml``.
    """
    from .interfaces.registry import resolve_provider_keys
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    cfg = _parse_config(raw)
    cfg._raw = raw
    cfg._provider_keys = resolve_provider_keys(keys_file=keys_file)
    # Resolve interface: auto -> cheapest available provider (must run after
    # _provider_keys are populated so we know which interfaces are configured).
    _resolve_auto_interfaces(cfg)
    return cfg


def _parse_config(raw: dict) -> CoEvalConfig:
    models = [_parse_model(m) for m in raw.get('models', [])]
    tasks = [_parse_task(t) for t in raw.get('tasks', [])]
    experiment = _parse_experiment(raw.get('experiment', {}))
    return CoEvalConfig(models=models, tasks=tasks, experiment=experiment)


def _parse_model(raw: dict) -> ModelConfig:
    return ModelConfig(
        name=raw['name'],
        interface=raw['interface'],
        parameters=dict(raw.get('parameters', {})),
        roles=list(raw.get('roles', [])),
        access_key=raw.get('access_key'),
        role_parameters={
            k: dict(v) for k, v in raw.get('role_parameters', {}).items()
        },
    )


def _parse_task(raw: dict) -> TaskConfig:
    s = raw.get('sampling', {})
    # Defaults enable the most-automatic specification level: a task may give only
    # name + description + output_description + sampling.total, and CoEval infers the
    # target attributes (Phase 1) and rubric (Phase 2). Explicit values override.
    sampling = SamplingConfig(
        target=s.get('target', [1]),
        nuance=s.get('nuance', [0]),
        total=int(s['total']),
    )
    return TaskConfig(
        name=raw['name'],
        description=raw['description'],
        output_description=raw['output_description'],
        target_attributes=raw.get('target_attributes', 'auto'),
        nuanced_attributes=raw.get('nuanced_attributes', {}),
        sampling=sampling,
        rubric=raw.get('rubric', 'auto'),
        target_attributes_seed=raw.get('target_attributes_seed'),
        nuanced_attributes_seed=raw.get('nuanced_attributes_seed'),
        store_nuanced=bool(raw.get('store_nuanced', False)),
        evaluation_mode=raw.get('evaluation_mode', 'single'),
        prompt_library=dict(raw.get('prompt_library', {})),
        label_attributes=list(raw.get('label_attributes', [])),
        category=raw.get('category'),
    )


def _parse_experiment(raw: dict) -> ExperimentConfig:
    # Parse batch config: {interface: {phase: bool}}
    batch_raw = raw.get('batch', {})
    batch: dict[str, dict[str, bool]] = {
        iface: {phase: bool(enabled) for phase, enabled in phases.items()}
        for iface, phases in batch_raw.items()
    }
    return ExperimentConfig(
        id=raw['id'],
        storage_folder=raw['storage_folder'],
        resume_from=raw.get('resume_from'),
        phases=dict(raw.get('phases', {})),
        log_level=raw.get('log_level', 'INFO'),
        quota={k: dict(v) for k, v in raw.get('quota', {}).items()},
        generation_retries=int(raw.get('generation_retries', 2)),
        batch=batch,
        probe_mode=str(raw.get('probe_mode', 'full')),
        probe_on_fail=str(raw.get('probe_on_fail', 'abort')),
        estimate_cost=bool(raw.get('estimate_cost', False)),
        estimate_samples=int(raw.get('estimate_samples', 2)),
    )



def _resolve_auto_interfaces(cfg: 'CoEvalConfig') -> None:
    """Replace ``interface: auto`` with the cheapest available provider.

    Scans ``benchmark/provider_pricing.yaml`` ``auto_routing`` table and
    selects the first matching entry whose interface has credentials
    available in ``cfg._provider_keys``.

    Modifies ``cfg.models`` in place.  Called from :func:`load_config`
    after ``_provider_keys`` are populated.

    Raises
    ------
    ValueError
        If ``interface: auto`` is used but no matching provider with
        available credentials can be found for the model.
    """
    from .interfaces.registry import resolve_auto_interface
    for model in cfg.models:
        if model.interface != 'auto':
            continue
        model_id = model.parameters.get('model', model.name)
        resolved = resolve_auto_interface(model_id, cfg._provider_keys)
        if resolved is None:
            raise ValueError(
                f"interface: auto — cannot resolve cheapest provider for "
                f"model '{model.name}' (parameters.model='{model_id}'). "
                f"Add credentials for the required provider in keys.yaml "
                f"or specify the interface explicitly."
            )
        model.interface = resolved

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(
    cfg: CoEvalConfig,
    continue_in_place: bool = False,
    _skip_folder_validation: bool = False,
) -> list[str]:
    """Apply all validation rules V-01 through V-17. Returns list of error strings.

    Parameters
    ----------
    cfg:
        Loaded experiment configuration.
    continue_in_place:
        When True: suppresses V-11 (folder must-not-exist) and activates
        V-14 (folder must already exist with meta.json).
    _skip_folder_validation:
        When True: suppresses both V-11 and V-14.  Used by standalone
        commands (``coeval probe``, ``coeval plan``) that operate on a
        config without caring about the experiment folder state.
    """
    errors: list[str] = []

    # V-01: all three top-level keys present and non-empty
    if not cfg.models:
        errors.append("Missing required top-level key or empty: 'models'")
    if not cfg.tasks:
        errors.append("Missing required top-level key or empty: 'tasks'")

    # V-02: model names unique
    seen: set[str] = set()
    for m in cfg.models:
        if m.name in seen:
            errors.append(f"Duplicate model name: '{m.name}'")
        seen.add(m.name)

    # V-03: task names unique
    seen = set()
    for t in cfg.tasks:
        if t.name in seen:
            errors.append(f"Duplicate task name: '{t.name}'")
        seen.add(t.name)

    # V-04: name character sets and reserved separator
    for m in cfg.models:
        if not _MODEL_NAME_RE.match(m.name):
            errors.append(
                f"Invalid model name '{m.name}': must match [A-Za-z0-9._-]"
            )
        elif _RESERVED_SEP in m.name:
            errors.append(
                f"Invalid model name '{m.name}': contains reserved separator '__'"
            )
    for t in cfg.tasks:
        if not _TASK_NAME_RE.match(t.name):
            errors.append(
                f"Invalid task name '{t.name}': must match [A-Za-z0-9_-]"
            )
        elif _RESERVED_SEP in t.name:
            errors.append(
                f"Invalid task name '{t.name}': contains reserved separator '__'"
            )
    if cfg.experiment.id and not _EXPERIMENT_ID_RE.match(cfg.experiment.id):
        errors.append(
            f"Invalid experiment id '{cfg.experiment.id}': must match [A-Za-z0-9._-]"
        )

    # V-05: roles valid and non-empty
    for m in cfg.models:
        if not m.roles:
            errors.append(f"Model '{m.name}' has no roles assigned")
        for role in m.roles:
            if role not in VALID_ROLES:
                errors.append(f"Unknown role '{role}' in model '{m.name}'")

    # V-06: interface valid
    for m in cfg.models:
        if m.interface not in VALID_INTERFACES:
            errors.append(f"Unknown interface '{m.interface}' in model '{m.name}'")

    # V-07: required roles present
    needs_teacher = any(
        isinstance(t.target_attributes, str)
        or isinstance(t.nuanced_attributes, str)
        or isinstance(t.rubric, str)
        for t in cfg.tasks
    )
    has_teacher = any('teacher' in m.roles for m in cfg.models)
    has_student = any('student' in m.roles for m in cfg.models)
    has_judge = any('judge' in m.roles for m in cfg.models)

    if needs_teacher and not has_teacher:
        errors.append(
            "No model assigned role 'teacher'; required for phases 1-3 with "
            "auto/complete attributes or rubric"
        )
    if not has_student:
        errors.append(
            "No model assigned role 'student'; required for phase 4"
        )
    if not has_judge:
        errors.append(
            "No model assigned role 'judge'; required for phase 5"
        )

    # V-08: Model mode not valid for phases 1 or 2
    for phase_id in _PHASES_NO_MODEL_MODE:
        mode = cfg.experiment.phases.get(phase_id)
        if mode == 'Model':
            errors.append(
                f"Phase '{phase_id}' does not support mode 'Model'"
            )

    # V-09: rubric 'extend' requires resume_from
    for t in cfg.tasks:
        if t.rubric == 'extend' and not cfg.experiment.resume_from:
            errors.append(
                f"Task '{t.name}': rubric: extend requires resume_from to be set in experiment"
            )

    # V-10: resume_from source folder must exist
    if cfg.experiment.resume_from:
        source_path = os.path.join(
            cfg.experiment.storage_folder, cfg.experiment.resume_from
        )
        if not os.path.isdir(source_path):
            errors.append(
                f"Source experiment '{cfg.experiment.resume_from}' not found in "
                f"{cfg.experiment.storage_folder}"
            )

    # V-11: for new experiments the target folder must NOT already exist
    # (skipped when continue_in_place is True — the folder is expected to exist)
    # (also skipped when _skip_folder_validation is True — standalone commands)
    if not cfg.experiment.resume_from and not continue_in_place and not _skip_folder_validation:
        target_path = os.path.join(
            cfg.experiment.storage_folder, cfg.experiment.id
        )
        if os.path.isdir(target_path):
            errors.append(
                f"Experiment folder '{cfg.experiment.id}' already exists in "
                f"'{cfg.experiment.storage_folder}'. Use --continue to resume it, "
                f"or choose a different experiment ID."
            )

    # V-12: generation_retries must be a non-negative integer
    if cfg.experiment.generation_retries < 0:
        errors.append(
            f"experiment.generation_retries must be >= 0, got {cfg.experiment.generation_retries}"
        )

    # V-13: batch config uses only known interfaces and valid batch phase IDs
    for iface, phases in cfg.experiment.batch.items():
        if iface not in BATCH_CAPABLE_INTERFACES:
            errors.append(
                f"experiment.batch: unknown or non-batchable interface '{iface}'. "
                f"Supported: {sorted(BATCH_CAPABLE_INTERFACES)}"
            )
        for phase_id in phases:
            if phase_id not in VALID_BATCH_PHASES:
                errors.append(
                    f"experiment.batch.{iface}: unknown phase '{phase_id}'. "
                    f"Batchable phases: {sorted(VALID_BATCH_PHASES)}"
                )

    # V-14: when continue_in_place is True the experiment folder must already exist
    # and contain a meta.json so the runner can read prior completion state
    # (skipped when _skip_folder_validation is True — standalone commands)
    if continue_in_place and not _skip_folder_validation:
        meta_path = os.path.join(
            cfg.experiment.storage_folder, cfg.experiment.id, 'meta.json'
        )
        if not os.path.isfile(meta_path):
            errors.append(
                f"--continue specified but no existing experiment found at "
                f"'{cfg.experiment.storage_folder}/{cfg.experiment.id}' "
                f"(meta.json is missing)"
            )

    # V-15: probe_mode must be one of the supported values
    if cfg.experiment.probe_mode not in VALID_PROBE_MODES:
        errors.append(
            f"experiment.probe_mode must be one of {sorted(VALID_PROBE_MODES)}, "
            f"got '{cfg.experiment.probe_mode}'"
        )

    # V-16: probe_on_fail must be one of the supported values
    if cfg.experiment.probe_on_fail not in VALID_PROBE_FAIL_MODES:
        errors.append(
            f"experiment.probe_on_fail must be one of "
            f"{sorted(VALID_PROBE_FAIL_MODES)}, got '{cfg.experiment.probe_on_fail}'"
        )

    # V-17: label_attributes must be a subset of target_attributes keys
    # (only enforceable when target_attributes is a static dict, not 'auto'/'complete')
    for t in cfg.tasks:
        if t.label_attributes and isinstance(t.target_attributes, dict):
            unknown_labels = [
                la for la in t.label_attributes
                if la not in t.target_attributes
            ]
            if unknown_labels:
                errors.append(
                    f"Task '{t.name}': label_attributes {unknown_labels} not found "
                    f"in target_attributes keys {list(t.target_attributes.keys())}"
                )

    # V-18: metric rubric factors must reference supported metrics
    from .metric_judge import SUPPORTED_METRICS, is_metric_factor
    for t in cfg.tasks:
        if isinstance(t.rubric, dict):
            for factor_name, factor_value in t.rubric.items():
                if is_metric_factor(factor_value):
                    metric_name = factor_value.get("metric", "")
                    if metric_name not in SUPPORTED_METRICS:
                        errors.append(
                            f"Task '{t.name}': metric rubric factor "
                            f"'{factor_name}' references unknown metric "
                            f"'{metric_name}'. Supported: "
                            f"{sorted(SUPPORTED_METRICS)}"
                        )

    # V-19: metric judges must have interface='metric' and specify a metric parameter
    for m in cfg.models:
        if m.interface == 'metric':
            if 'judge' not in m.roles:
                errors.append(
                    f"Model '{m.name}' uses 'metric' interface but does not "
                    f"have 'judge' role. Metric models can only be judges."
                )

    return errors
