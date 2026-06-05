"""Model availability probe: verifies each model is reachable before the pipeline starts.

For network interfaces (OpenAI, Anthropic, Gemini) a lightweight API call that does
NOT consume generation tokens is used (model listing endpoint or equivalent).
For HuggingFace the Hub metadata API is queried to confirm the model repository is
accessible with the supplied credentials.

Probe modes
-----------
``full``
    Probe every model in the config (default).
``resume``
    Probe only models whose phases have not yet been completed.  Phase-to-role
    mapping: attribute_mapping/rubric_mapping/data_generation → teacher,
    response_collection → student, evaluation → judge.
``disable``
    Skip the probe entirely (equivalent to the legacy ``--skip-probe`` flag).

Usage::

    from runner.interfaces.probe import run_probe
    results, needed = run_probe(cfg, logger, mode='full', on_fail='abort',
                                phases_completed=set())
    # results → {"gpt-4o-mini": "ok", "claude-3-haiku": "ok", "bad-model": "err msg"}
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import CoEvalConfig, ModelConfig
    from ..logger import RunLogger

# Phases that require each role
_PHASE_TO_ROLES: dict[str, list[str]] = {
    'attribute_mapping':  ['teacher'],
    'rubric_mapping':     ['teacher'],
    'data_generation':    ['teacher'],
    'response_collection': ['student'],
    'evaluation':         ['judge'],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_probe(
    cfg: 'CoEvalConfig',
    logger: 'RunLogger',
    mode: str = 'full',
    on_fail: str = 'abort',
    phases_completed: 'set[str] | None' = None,
    only_models: 'set[str] | None' = None,
    probe_results_path: 'Path | None' = None,
) -> tuple[dict[str, str], set[str]]:
    """Run the model availability probe and return results.

    Parameters
    ----------
    cfg:
        Loaded experiment configuration.
    logger:
        Run logger — results are written to both log file and console.
    mode:
        ``'full'`` — probe all models; ``'resume'`` — probe only models needed
        for remaining phases; ``'disable'`` — skip probe entirely.
    on_fail:
        ``'abort'`` — caller should abort if any model reports failure;
        ``'warn'``  — caller should log a warning but continue.
    phases_completed:
        Set of phase IDs already completed (used in ``resume`` mode).
        Pass an empty set or ``None`` for a fresh experiment.
    only_models:
        When set, restrict probing to models in this set regardless of mode.
    probe_results_path:
        Optional path to write a ``probe_results.json`` file.  Pass the
        experiment storage ``run_path / "probe_results.json"`` from the runner.

    Returns
    -------
    (results, needed_names)
        ``results``      — dict mapping model name → ``'ok'`` or error string.
        ``needed_names`` — set of model names that were probed.
    """
    if mode == 'disable':
        logger.info("Probe: disabled — skipping model availability check")
        return {}, set()

    # Determine which models to probe
    needed_names = _models_needed(cfg, mode, phases_completed or set())
    if only_models is not None:
        needed_names = needed_names & only_models

    models_to_probe = [m for m in cfg.models if m.name in needed_names]

    if not models_to_probe:
        logger.info("Probe: no models to probe (all phases already completed?)")
        return {}, needed_names

    logger.info(
        f"Probe: testing {len(models_to_probe)} model(s) "
        f"(mode='{mode}', on_fail='{on_fail}') ..."
    )

    # Resolved provider credentials from the key file (may be empty dict)
    provider_keys: dict = getattr(cfg, '_provider_keys', {}) or {}

    results: dict[str, str] = {}
    for model in models_to_probe:
        logger.info(f"Probe: testing '{model.name}' ({model.interface}) ...")
        try:
            _probe_one(model, provider_keys)
        except Exception as exc:
            results[model.name] = str(exc)
            logger.error(f"Probe: '{model.name}' - UNAVAILABLE: {exc}")
        else:
            results[model.name] = 'ok'
            logger.info(f"Probe: '{model.name}' - available [OK]")

    # Write probe_results.json if path provided
    if probe_results_path is not None:
        try:
            probe_results_path.parent.mkdir(parents=True, exist_ok=True)
            probe_results_path.write_text(
                json.dumps(
                    {
                        'mode': mode,
                        'on_fail': on_fail,
                        'results': results,
                        'probed_models': sorted(needed_names),
                    },
                    indent=2,
                ),
                encoding='utf-8',
            )
            logger.info(f"Probe results written to {probe_results_path}")
        except Exception as exc:
            logger.warning(f"Could not write probe_results.json: {exc}")

    n_ok   = sum(1 for v in results.values() if v == 'ok')
    n_fail = len(results) - n_ok
    if n_fail:
        if on_fail == 'abort':
            logger.error(
                f"Probe: {n_fail} model(s) unavailable — "
                "aborting to prevent a partial run."
            )
        else:
            logger.warning(
                f"Probe: {n_fail} model(s) unavailable — "
                "continuing with warnings (some phases may fail)."
            )
    else:
        logger.info(f"Probe: all {n_ok} model(s) available [OK]")

    return results, needed_names


# ---------------------------------------------------------------------------
# Legacy helper kept for backward compatibility
# ---------------------------------------------------------------------------

def probe_models(
    cfg: 'CoEvalConfig',
    logger: 'RunLogger',
) -> dict[str, str]:
    """Test every model in *cfg* for availability (legacy API).

    .. deprecated::
        Use :func:`run_probe` with explicit ``mode`` / ``on_fail`` parameters.
        This wrapper always uses ``mode='full'`` and ``on_fail='abort'``.
    """
    results, _ = run_probe(cfg, logger, mode='full', on_fail='abort')
    return results


# ---------------------------------------------------------------------------
# Helper: determine which models are needed
# ---------------------------------------------------------------------------

def _models_needed(
    cfg: 'CoEvalConfig',
    mode: str,
    phases_completed: 'set[str]',
) -> 'set[str]':
    """Return the set of model names that should be probed."""
    if mode == 'full':
        return {m.name for m in cfg.models}

    # resume mode: only probe models needed for remaining phases
    from ..config import PHASE_IDS
    needed: set[str] = set()
    for phase_id in PHASE_IDS:
        if phase_id in phases_completed:
            continue
        for role in _PHASE_TO_ROLES.get(phase_id, []):
            for model in cfg.get_models_by_role(role):
                needed.add(model.name)
    return needed


# ---------------------------------------------------------------------------
# Per-model probes
# ---------------------------------------------------------------------------

def _probe_one(model: 'ModelConfig', provider_keys: dict) -> None:
    """Probe a single model.  Raises on any failure."""
    iface = model.interface
    if iface == 'benchmark':
        # Virtual interface — no API to probe; data pre-ingested by `coeval ingest`
        return
    if iface == 'metric':
        # Virtual interface — deterministic metric judge, no model weights or API
        return
    if iface == 'openai':
        _probe_openai(model, provider_keys)
    elif iface == 'anthropic':
        _probe_anthropic(model, provider_keys)
    elif iface == 'gemini':
        _probe_gemini(model, provider_keys)
    elif iface == 'azure_openai':
        _probe_azure_openai(model, provider_keys)
    elif iface == 'bedrock':
        _probe_bedrock(model, provider_keys)
    elif iface == 'vertex':
        _probe_vertex(model, provider_keys)
    elif iface == 'openrouter':
        _probe_openrouter(model, provider_keys)
    elif iface == 'azure_ai':
        _probe_azure_ai(model, provider_keys)
    elif iface in ('groq', 'deepseek', 'mistral', 'deepinfra', 'cerebras', 'ollama'):
        _probe_openai_compat(model, iface, provider_keys)
    else:
        _probe_huggingface(model, provider_keys)


def _probe_openai(model: 'ModelConfig', provider_keys: dict) -> None:
    """Call OpenAI models.list() — authenticates without consuming tokens."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed (pip install openai)")
    key = (
        model.access_key
        or provider_keys.get('openai', {}).get('api_key')
        or os.environ.get('OPENAI_API_KEY')
    )
    client = OpenAI(api_key=key)
    client.models.list()


def _probe_anthropic(model: 'ModelConfig', provider_keys: dict) -> None:
    """Call Anthropic models.list() — authenticates without consuming tokens."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed (pip install anthropic)")
    key = (
        model.access_key
        or provider_keys.get('anthropic', {}).get('api_key')
        or os.environ.get('ANTHROPIC_API_KEY')
    )
    client = anthropic.Anthropic(api_key=key)
    client.models.list()


def _probe_gemini(model: 'ModelConfig', provider_keys: dict) -> None:
    """Call client.models.list() via google-genai SDK — no tokens consumed."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError(
            "google-genai package not installed (pip install google-genai)"
        )
    pk = provider_keys.get('gemini', {})
    key = (
        model.access_key
        or pk.get('api_key')
        or os.environ.get('GEMINI_API_KEY')
        or os.environ.get('GOOGLE_API_KEY')
    )
    client = genai.Client(api_key=key)
    next(iter(client.models.list()), None)


def _probe_huggingface(model: 'ModelConfig', provider_keys: dict) -> None:
    """Query the HuggingFace Hub metadata API — no GPU/weights loaded."""
    model_id = model.parameters.get('model', '')
    if not model_id:
        raise RuntimeError(
            f"HuggingFace model '{model.name}' has no 'model' parameter set"
        )
    try:
        from huggingface_hub import model_info
    except ImportError:
        return  # huggingface_hub not installed; skip silently
    access_token = (
        model.access_key
        or provider_keys.get('huggingface', {}).get('token')
        or os.environ.get('HF_TOKEN')
        or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    )
    model_info(model_id, token=access_token)


def _probe_azure_openai(model: 'ModelConfig', provider_keys: dict) -> None:
    """Call Azure OpenAI models.list() to verify endpoint + API key."""
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise RuntimeError("openai package not installed (pip install openai)")
    params = model.parameters
    pk = provider_keys.get('azure_openai', {})
    key = (
        model.access_key
        or pk.get('api_key')
        or os.environ.get('AZURE_OPENAI_API_KEY')
    )
    endpoint = (
        params.get('azure_endpoint')
        or pk.get('endpoint')
        or os.environ.get('AZURE_OPENAI_ENDPOINT')
    )
    api_version = (
        params.get('api_version')
        or pk.get('api_version')
        or os.environ.get('AZURE_OPENAI_API_VERSION')
        or '2024-08-01-preview'
    )
    if not endpoint:
        raise RuntimeError(
            f"Azure OpenAI model '{model.name}' has no 'azure_endpoint' parameter "
            "and AZURE_OPENAI_ENDPOINT is not set"
        )
    client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version)
    client.models.list()


def _probe_bedrock(model: 'ModelConfig', provider_keys: dict) -> None:
    """List Bedrock foundation models to verify credentials and region."""
    params = model.parameters
    pk = provider_keys.get('bedrock', {})
    region = (
        params.get('region')
        or pk.get('region')
        or os.environ.get('AWS_DEFAULT_REGION')
        or 'us-east-1'
    )

    # Native Bedrock API key takes priority (no boto3 required)
    api_key = model.access_key or params.get('api_key') or pk.get('api_key')
    if api_key:
        _probe_bedrock_api_key(api_key, region)
        return

    # IAM credentials via boto3
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 not installed (pip install boto3)")

    access_key_id = (
        params.get('access_key_id')
        or pk.get('access_key_id')
        or os.environ.get('AWS_ACCESS_KEY_ID')
    )
    secret_access_key = (
        params.get('secret_access_key')
        or pk.get('secret_access_key')
        or os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    session_token = (
        params.get('session_token')
        or pk.get('session_token')
        or os.environ.get('AWS_SESSION_TOKEN')
    )

    session_kwargs: dict = {'region_name': region}
    if access_key_id:
        session_kwargs['aws_access_key_id'] = access_key_id
    if secret_access_key:
        session_kwargs['aws_secret_access_key'] = secret_access_key
    if session_token:
        session_kwargs['aws_session_token'] = session_token

    client = boto3.client('bedrock', **session_kwargs)
    # list_foundation_models returns immediately and validates credentials
    client.list_foundation_models(byOutputModality='TEXT')


def _probe_bedrock_api_key(api_key: str, region: str) -> None:
    """Probe Bedrock using native API key via direct HTTP (no boto3 needed)."""
    import json
    import urllib.error
    import urllib.request

    url = f"https://bedrock.{region}.amazonaws.com/foundation-models?byOutputModality=TEXT"
    req = urllib.request.Request(url, headers={'x-amzn-bedrock-key': api_key})
    try:
        with urllib.request.urlopen(req) as resp:
            json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            err = json.loads(raw)
        except Exception:
            err = {'message': raw.decode('utf-8', errors='replace')}
        raise RuntimeError(
            f"Bedrock API key probe failed — HTTP {exc.code}: "
            f"{err.get('message', str(err))}"
        ) from exc


def _probe_openrouter(model: 'ModelConfig', provider_keys: dict) -> None:
    """Call the OpenRouter models endpoint to verify API key."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed (pip install openai)")
    pk = provider_keys.get('openrouter', {})
    key = (
        model.access_key
        or pk.get('api_key')
        or os.environ.get('OPENROUTER_API_KEY')
    )
    if not key:
        raise RuntimeError(
            f"OpenRouter model '{model.name}' has no API key — "
            "set OPENROUTER_API_KEY or add 'openrouter' to the provider key file"
        )
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    client.models.list()


def _probe_azure_ai(model: 'ModelConfig', provider_keys: dict) -> None:
    """Call the Azure AI / GitHub Models models endpoint to verify API key."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed (pip install openai)")
    pk = provider_keys.get('azure_ai', {})
    key = (
        model.access_key
        or pk.get('api_key')
        or os.environ.get('AZURE_AI_API_KEY')
        or os.environ.get('GITHUB_TOKEN')
    )
    if not key:
        raise RuntimeError(
            f"Azure AI model '{model.name}' has no API key — "
            "set AZURE_AI_API_KEY, GITHUB_TOKEN, or add 'azure_ai' to the provider key file"
        )
    endpoint = (
        model.parameters.get('azure_endpoint')
        or pk.get('endpoint')
        or "https://models.inference.ai.azure.com"
    )
    client = OpenAI(api_key=key, base_url=endpoint)
    client.models.list()


def _probe_vertex(model: 'ModelConfig', provider_keys: dict) -> None:
    """Initialise the Vertex AI SDK to verify project credentials."""
    try:
        import vertexai
    except ImportError:
        raise RuntimeError(
            "google-cloud-aiplatform not installed "
            "(pip install google-cloud-aiplatform)"
        )
    params = model.parameters
    pk = provider_keys.get('vertex', {})
    project = (
        params.get('project')
        or pk.get('project')
        or os.environ.get('GOOGLE_CLOUD_PROJECT')
        or os.environ.get('GCLOUD_PROJECT')
    )
    location = (
        params.get('location')
        or pk.get('location')
        or os.environ.get('GOOGLE_CLOUD_LOCATION')
        or 'us-central1'
    )
    sa_key = (
        params.get('service_account_key')
        or model.access_key
        or pk.get('service_account_key')
        or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    )
    if not project:
        raise RuntimeError(
            f"Vertex AI model '{model.name}' has no 'project' parameter "
            "and GOOGLE_CLOUD_PROJECT is not set"
        )
    if sa_key:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = sa_key
    # vertexai.init() validates project/credentials; lightweight, no model calls
    vertexai.init(project=project, location=location)


def _probe_openai_compat(model: 'ModelConfig', interface: str, provider_keys: dict) -> None:
    """Probe an OpenAI-compatible provider by calling models.list()."""
    from .openai_compat_iface import _REGISTRY
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed (pip install openai)")
    default_url, env_key, label = _REGISTRY[interface]
    pk = provider_keys.get(interface, {})

    # Resolve base URL (Ollama may override via params, key file, or env var)
    base_url = (
        model.parameters.get('base_url') if model.parameters else None
    ) or (
        pk.get('base_url') if isinstance(pk, dict) else None
    ) or (
        os.environ.get('OLLAMA_HOST') if interface == 'ollama' else None
    ) or default_url

    # Resolve API key — optional for Ollama
    if env_key is not None:
        key = (
            model.access_key
            or (pk.get('api_key') if isinstance(pk, dict) else pk)
            or os.environ.get(env_key)
        )
        if not key:
            raise RuntimeError(
                f"{label} API key not found — set {env_key} or add "
                f"'{interface}' to the provider key file"
            )
    else:
        key = model.access_key or 'ollama'

    client = OpenAI(api_key=key, base_url=base_url)
    client.models.list()
