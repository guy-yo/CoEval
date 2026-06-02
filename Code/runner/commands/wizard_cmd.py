"""coeval wizard — interactive LLM-assisted experiment configuration wizard.

Guides the user through defining an evaluation experiment via a conversational
interface.  The user describes their goal in natural language, answers a few
clarifying questions, and the wizard uses an LLM to generate a complete,
valid YAML configuration ready for ``coeval run``.

Usage::

    coeval wizard                               # write final YAML to stdout
    coeval wizard --out my-experiment.yaml      # write to file
    coeval wizard --model gpt-4o-mini           # use specific model for generation
    coeval wizard --keys ~/.coeval/keys.yaml    # custom key file

Workflow::

    1. Describe your evaluation goal in plain English
    2. Answer clarifying questions (models, item count, output path)
    3. Review the generated YAML — refine in natural language if needed
    4. Wizard writes the final config to disk
    5. Run: coeval run --config <output.yaml>
"""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# YAML schema reference fed to the LLM
# ---------------------------------------------------------------------------

_SCHEMA_DOC = """
COEVAL YAML CONFIGURATION SCHEMA
=================================

Top-level keys: experiment, models, tasks

━━━ experiment ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  id:             str   REQUIRED  Experiment identifier (letters, digits, ._-)
  storage_folder: str   REQUIRED  Path to results folder (e.g. benchmark/runs)
  log_level:      str   optional  DEBUG | INFO | WARNING | ERROR  (default INFO)

━━━ models  (list of model entries) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each entry:
  name:       str   REQUIRED  Unique model identifier (letters, digits, ._-)
  interface:  str   REQUIRED  One of: openai | anthropic | gemini | huggingface
                               | azure_openai | bedrock | vertex | openrouter
  parameters: dict  REQUIRED  Provider-specific call parameters, e.g.:
                               {model: gpt-4o-mini, temperature: 0.7, max_tokens: 512}
                               For huggingface: {model: HuggingFaceTB/SmolLM2-135M-Instruct,
                                                 device: cpu, max_new_tokens: 256}
  roles:      list  REQUIRED  Non-empty list of: student | teacher | judge
                               (one model can hold multiple roles)
  role_parameters: dict  optional  Per-role parameter overrides, e.g.:
                               {teacher: {temperature: 0.8}, judge: {temperature: 0.0}}

━━━ tasks  (list of task entries) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each entry:
  name:               str   REQUIRED  Task name (letters, digits, _-)
  description:        str   REQUIRED  Plain-text description of the task
  output_description: str   REQUIRED  Expected output format/structure
  target_attributes:  dict  REQUIRED  Attributes that vary the task prompts.
                                       Keys = attribute names, values = lists of
                                       possible values.  Use "auto" to have the
                                       teacher model generate these.
                                       Example: {difficulty: [easy, medium, hard],
                                                 domain: [science, history]}
  nuanced_attributes: dict  REQUIRED  Fine-grained quality/nuance dimensions.
                                       Same format as target_attributes.
                                       Use "" (empty string) for none.
                                       Example: {tone: [formal, casual]}
  sampling:           dict  REQUIRED  How many attributes to sample per item:
                                       target: [min, max]  (or "all")
                                       nuance: [min, max]
                                       total: N            (items per teacher)
                                       Example: {target: [1, 2], nuance: [1, 1], total: 20}
  rubric:             dict  REQUIRED  Evaluation criteria.
                                       Keys = factor names, values = descriptions.
                                       Use "auto" to generate via the teacher model.
                                       Example: {clarity: "Is the response clear and readable?",
                                                 accuracy: "Is the information factually correct?"}
  evaluation_mode: str  optional  "single" (default) or "per_factor"

━━━ IMPORTANT RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Every experiment needs at least one model with each role: teacher, student, judge
  (the same model can hold multiple roles)
- Model names must be unique; use simple identifiers (no spaces)
- Task names must be unique; use underscores not spaces
- sampling.total is per-teacher per-task; start with 5–20 for quick experiments,
  50–200 for production benchmarks
- For quick tests with HuggingFace models: use interface: huggingface and
  max_new_tokens instead of max_tokens
- rubric factors are scored as: High / Medium / Low by the judge model
- If target_attributes or rubric is "auto", teacher models MUST exist in the config
"""

_GENERATION_SYSTEM = (
    "You are an expert at configuring CoEval evaluation experiments.\n"
    "Given a description of what the user wants to evaluate, generate a complete, "
    "valid YAML configuration.\n\n"
    + _SCHEMA_DOC
    + "\n\nReturn ONLY the YAML content — no markdown fences, no prose, no explanations.\n"
    "Start your response with 'experiment:' on the first line."
)

_REFINEMENT_SYSTEM = (
    "You are an expert at configuring CoEval evaluation experiments.\n"
    "The user has a YAML configuration and wants to modify it.\n"
    "Apply their requested changes and return the complete updated YAML.\n\n"
    + _SCHEMA_DOC
    + "\n\nReturn ONLY the complete YAML — no markdown fences, no prose, no explanations.\n"
    "Start your response with 'experiment:' on the first line."
)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_generator(keys: dict) -> tuple[str, str, str] | None:
    """Return (provider, model_id, api_key) for the best available generator.

    Priority order: openai → anthropic → gemini → openrouter.
    Returns None if no provider is configured.
    """
    # OpenAI — best YAML generation reliability
    import os
    openai_key = (
        keys.get('openai')
        or os.environ.get('OPENAI_API_KEY', '')
    )
    if openai_key and isinstance(openai_key, str) and openai_key.startswith('sk-'):
        return 'openai', 'gpt-4o-mini', openai_key

    # Anthropic
    anthropic_key = (
        keys.get('anthropic')
        or os.environ.get('ANTHROPIC_API_KEY', '')
    )
    if anthropic_key and isinstance(anthropic_key, str):
        return 'anthropic', 'claude-3-5-haiku-20241022', anthropic_key

    # Gemini
    gemini_key = (
        keys.get('gemini')
        or os.environ.get('GEMINI_API_KEY', '')
        or os.environ.get('GOOGLE_API_KEY', '')
    )
    if gemini_key and isinstance(gemini_key, str):
        return 'gemini', 'gemini-2.0-flash', gemini_key

    # OpenRouter
    or_key = (
        keys.get('openrouter')
        or os.environ.get('OPENROUTER_API_KEY', '')
    )
    if or_key and isinstance(or_key, str):
        return 'openrouter', 'openai/gpt-4o-mini', or_key

    return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(
    provider: str,
    model_id: str,
    api_key: str,
    messages: list[dict],
    *,
    max_tokens: int = 4096,
) -> str:
    """Call the LLM and return the text response."""
    if provider == 'openai':
        import openai  # type: ignore[import-untyped]
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ''

    if provider == 'anthropic':
        import anthropic  # type: ignore[import-untyped]
        client = anthropic.Anthropic(api_key=api_key)
        system = next((m['content'] for m in messages if m['role'] == 'system'), None)
        user_messages = [m for m in messages if m['role'] != 'system']
        resp = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system or '',
            messages=user_messages,
        )
        return resp.content[0].text if resp.content else ''

    if provider == 'gemini':
        import google.generativeai as genai  # type: ignore[import-untyped]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        # Combine system + user into single prompt for Gemini
        system = next((m['content'] for m in messages if m['role'] == 'system'), '')
        user = '\n\n'.join(
            m['content'] for m in messages if m['role'] == 'user'
        )
        prompt = f"{system}\n\n{user}" if system else user
        resp = model.generate_content(prompt)
        return resp.text or ''

    if provider == 'openrouter':
        import openai  # type: ignore[import-untyped]
        client = openai.OpenAI(
            api_key=api_key,
            base_url='https://openrouter.ai/api/v1',
        )
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ''

    raise ValueError(f"Unsupported provider for wizard: {provider!r}")


# ---------------------------------------------------------------------------
# YAML cleaning
# ---------------------------------------------------------------------------

def _clean_yaml(text: str) -> str:
    """Strip markdown code fences and leading/trailing whitespace."""
    text = text.strip()
    # Remove ```yaml ... ``` or ``` ... ```
    text = re.sub(r'^```(?:yaml)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def _hr(char: str = '─', width: int = 60) -> None:
    print(char * width)


def _banner(title: str) -> None:
    print()
    _hr('━')
    print(f"  {title}")
    _hr('━')
    print()


def _ask(prompt: str, default: str = '') -> str:
    """Ask a single-line question; use default on empty input."""
    hint = f' [{default}]' if default else ''
    try:
        answer = input(f'{prompt}{hint}: ').strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return answer if answer else default


def _ask_int(prompt: str, default: int, min_val: int = 1) -> int:
    """Ask for an integer with validation."""
    while True:
        raw = _ask(prompt, str(default))
        try:
            val = int(raw)
            if val < min_val:
                print(f"  Please enter a number ≥ {min_val}.")
                continue
            return val
        except ValueError:
            print(f"  Please enter a valid integer.")


def _read_multiline(prompt: str) -> str:
    """Collect multi-line input.  An empty line ends input."""
    print(prompt)
    print("  (Enter your text; press Enter twice when done)")
    lines = []
    blank_count = 0
    try:
        while True:
            line = input('  > ')
            if line == '':
                blank_count += 1
                if blank_count >= 1 and lines:
                    break
            else:
                blank_count = 0
                lines.append(line)
    except (EOFError, KeyboardInterrupt):
        print()
    return '\n'.join(lines).strip()


def _paged_print(text: str, label: str = 'Generated YAML') -> None:
    """Print a block of text with a simple header and footer."""
    print()
    _hr('─')
    print(f'  {label}')
    _hr('─')
    print(text)
    _hr('─')
    print()


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _build_user_message(
    description: str,
    available_providers: list[str],
    models_hint: str,
    items_per_task: int,
    experiment_id: str,
    storage_folder: str,
) -> str:
    providers_str = ', '.join(available_providers) if available_providers else 'any'
    return (
        f"Create a CoEval experiment configuration for the following goal:\n\n"
        f"{description}\n\n"
        f"Requirements:\n"
        f"- experiment.id: {experiment_id!r}\n"
        f"- experiment.storage_folder: {storage_folder!r}\n"
        f"- Available providers (use only these): {providers_str}\n"
        f"- Sampling total (items per teacher per task): {items_per_task}\n"
        + (f"- Preferred models: {models_hint}\n" if models_hint else '')
        + "- Make sure every required role is covered (teacher, student, judge)\n"
        "- Generate 1-4 tasks appropriate for the goal\n"
        "- Keep rubric factors to 2-4 per task\n"
        "- Keep target_attributes to 2-4 per task with 2-4 values each\n"
        "- Use role_parameters to set temperature=0.0 for judge, temperature=0.7 for teacher\n"
    )


def _generate_yaml(
    description: str,
    available_providers: list[str],
    models_hint: str,
    items_per_task: int,
    experiment_id: str,
    storage_folder: str,
    provider: str,
    model_id: str,
    api_key: str,
) -> str:
    """Call the LLM to generate a YAML config."""
    user_msg = _build_user_message(
        description, available_providers, models_hint,
        items_per_task, experiment_id, storage_folder,
    )
    messages = [
        {'role': 'system', 'content': _GENERATION_SYSTEM},
        {'role': 'user', 'content': user_msg},
    ]
    raw = _call_llm(provider, model_id, api_key, messages)
    return _clean_yaml(raw)


def _refine_yaml(
    current_yaml: str,
    feedback: str,
    provider: str,
    model_id: str,
    api_key: str,
) -> str:
    """Call the LLM to apply refinements to the current YAML."""
    messages = [
        {'role': 'system', 'content': _REFINEMENT_SYSTEM},
        {
            'role': 'user',
            'content': (
                f"Current configuration:\n\n{current_yaml}\n\n"
                f"Please make the following changes:\n\n{feedback}"
            ),
        },
    ]
    raw = _call_llm(provider, model_id, api_key, messages)
    return _clean_yaml(raw)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _try_validate(yaml_str: str) -> list[str]:
    """Parse and validate a YAML string.  Returns list of error strings."""
    try:
        import yaml
    except ImportError:
        return ["PyYAML not installed — cannot validate"]
    try:
        raw = yaml.safe_load(yaml_str)
    except yaml.YAMLError as exc:
        return [f"YAML parse error: {exc}"]

    try:
        from ..config import _parse_config, validate_config
        cfg = _parse_config(raw)
        errors = validate_config(cfg, continue_in_place=False)
        # Filter out runtime-state errors irrelevant to a freshly generated config:
        # the storage folder does not exist yet and no prior run/meta.json is present.
        def _is_runtime_state(e: str) -> bool:
            return ('does not exist' in e or 'already exists' in e
                    or 'meta.json is missing' in e or '--continue' in e)
        errors = [e for e in errors if not _is_runtime_state(e)]
        return errors
    except Exception as exc:
        return [f"Config parse error: {exc}"]


# ---------------------------------------------------------------------------
# Main wizard flow
# ---------------------------------------------------------------------------

def cmd_wizard(args) -> None:
    """Entry point for ``coeval wizard``."""
    # --- Load keys ---
    from ..interfaces.registry import load_keys_file
    keys = load_keys_file(getattr(args, 'keys', None))

    # --- Override generator model if requested ---
    override_model: str | None = getattr(args, 'model', None)

    # --- Detect generator ---
    generator = _detect_generator(keys)
    if generator is None:
        print(
            "ERROR: No LLM provider configured.\n"
            "Add credentials to ~/.coeval/keys.yaml or set OPENAI_API_KEY.\n"
            "Supported providers: openai, anthropic, gemini, openrouter",
            file=sys.stderr,
        )
        sys.exit(1)

    gen_provider, gen_model, gen_key = generator

    # Apply --model override
    if override_model:
        # Determine provider from model ID prefix
        if '/' in override_model:
            gen_provider = 'openrouter'
            gen_model = override_model
        elif override_model.startswith('claude'):
            gen_provider = 'anthropic'
            gen_model = override_model
        elif override_model.startswith('gemini'):
            gen_provider = 'gemini'
            gen_model = override_model
        else:
            # Default to openai
            gen_model = override_model

    # --- Welcome (interactive only; non-interactive --objective stays quiet for piping) ---
    _noninteractive = bool(getattr(args, 'objective', None))
    if not _noninteractive:
        _banner('CoEval Experiment Wizard')
        print(textwrap.fill(
            "Welcome! This wizard will help you create a CoEval evaluation experiment "
            "configuration using AI assistance.  Answer a few questions and the wizard "
            "will generate a ready-to-run YAML config for you.",
            width=70, initial_indent='  ', subsequent_indent='  ',
        ))
        print()
        print(f"  Generator: {gen_provider} / {gen_model}")
        print()

    # --- Gather available providers ---
    available_providers = [p for p in ['openai', 'anthropic', 'gemini', 'huggingface',
                                        'bedrock', 'vertex', 'azure_openai', 'openrouter']
                           if p in keys or _provider_env_available(p)]

    # --- Non-interactive: one-shot objective -> config (no questions asked) ---
    objective = getattr(args, 'objective', None)
    if objective:
        description = objective.strip()
        experiment_id = re.sub(r'[^A-Za-z0-9._-]', '-', description[:30].strip()).strip('-')
        experiment_id = re.sub(r'-+', '-', experiment_id).strip('-') or 'coeval-experiment'
        storage_folder = getattr(args, 'storage_folder', None) or './Runs'
        items_per_task = getattr(args, 'items', None) or 12
        models_hint = getattr(args, 'models', None) or ''
        print(f"  Generating config from objective via {gen_provider}/{gen_model} ...", file=sys.stderr)
        try:
            yaml_str = _generate_yaml(
                description=description, available_providers=available_providers,
                models_hint=models_hint, items_per_task=items_per_task,
                experiment_id=experiment_id, storage_folder=storage_folder,
                provider=gen_provider, model_id=gen_model, api_key=gen_key,
            )
        except Exception as exc:
            print(f"  ERROR: LLM call failed: {exc}", file=sys.stderr)
            sys.exit(1)
        # Auto-refine on validation errors, up to 2 retries, with no human in the loop.
        for _ in range(2):
            errors = _try_validate(yaml_str)
            if not errors:
                break
            print(f"  Auto-fixing {len(errors)} validation issue(s) ...", file=sys.stderr)
            try:
                yaml_str = _refine_yaml(
                    current_yaml=yaml_str,
                    feedback="Fix exactly these validation errors and return only the YAML: "
                             + "; ".join(errors),
                    provider=gen_provider, model_id=gen_model, api_key=gen_key,
                )
            except Exception:
                break
        errors = _try_validate(yaml_str)
        out_path_arg = getattr(args, 'out', None)
        if out_path_arg:
            Path(out_path_arg).write_text(yaml_str, encoding='utf-8')
            status = f"{len(errors)} residual validation issue(s)" if errors else "validated OK"
            print(f"  Wrote config to {out_path_arg} ({status})", file=sys.stderr)
        else:
            print(yaml_str)
        return

    # --- Step 1: Describe the goal ---
    _hr()
    print("  STEP 1 — Describe what you want to evaluate")
    _hr()
    description = _read_multiline("\n  Describe your evaluation goal in plain English:")
    if not description:
        print("No description provided. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Clarifying questions ---
    print()
    _hr()
    print("  STEP 2 — Clarifying questions")
    _hr()
    print()

    exp_id_default = re.sub(r'[^A-Za-z0-9._-]', '-', description[:30].strip()).strip('-')
    exp_id_default = re.sub(r'-+', '-', exp_id_default).strip('-') or 'my-experiment'
    experiment_id = _ask("  Experiment ID", exp_id_default)
    experiment_id = re.sub(r'[^A-Za-z0-9._-]', '-', experiment_id).strip('-') or exp_id_default

    storage_folder = _ask("  Storage folder", 'benchmark/runs')
    items_per_task = _ask_int("  Items per task per teacher model", 10, min_val=1)

    if available_providers:
        print(f"\n  Available providers: {', '.join(available_providers)}")
    models_hint = _ask(
        "  Preferred models (e.g. 'gpt-4o-mini, claude-3-5-haiku') or press Enter to let the wizard decide",
        '',
    )

    # --- Step 3: Generate config ---
    print()
    _hr()
    print("  STEP 3 — Generating configuration…")
    _hr()
    print(f"\n  Calling {gen_provider}/{gen_model}…")

    try:
        yaml_str = _generate_yaml(
            description=description,
            available_providers=available_providers,
            models_hint=models_hint,
            items_per_task=items_per_task,
            experiment_id=experiment_id,
            storage_folder=storage_folder,
            provider=gen_provider,
            model_id=gen_model,
            api_key=gen_key,
        )
    except Exception as exc:
        print(f"\n  ERROR: LLM call failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Validate
    errors = _try_validate(yaml_str)
    if errors:
        print(f"\n  ⚠  Generated config has {len(errors)} validation issue(s):")
        for e in errors[:5]:
            print(f"     • {e}")
        print("  The wizard will attempt to fix these in the refinement step.\n")

    _paged_print(yaml_str, label='Generated YAML')

    # --- Step 4: Refinement loop ---
    while True:
        print("  Options:")
        print("    [Enter]      Accept and save this configuration")
        print("    [type text]  Describe changes to make, then press Enter")
        print("    q            Quit without saving")
        print()
        feedback = _ask("  Your choice", '').strip()

        if feedback.lower() in ('q', 'quit', 'exit'):
            print("  Wizard exited without saving.")
            sys.exit(0)

        if not feedback:
            break  # Accept

        # Apply refinement
        print(f"\n  Applying changes via {gen_provider}/{gen_model}…")
        try:
            yaml_str = _refine_yaml(
                current_yaml=yaml_str,
                feedback=feedback,
                provider=gen_provider,
                model_id=gen_model,
                api_key=gen_key,
            )
        except Exception as exc:
            print(f"\n  ERROR: LLM call failed: {exc}", file=sys.stderr)
            continue

        errors = _try_validate(yaml_str)
        if errors:
            print(f"\n  ⚠  Config has {len(errors)} validation issue(s) after refinement:")
            for e in errors[:5]:
                print(f"     • {e}")

        _paged_print(yaml_str, label='Updated YAML')

    # --- Step 5: Write output ---
    out_path_arg: str | None = getattr(args, 'out', None)
    if out_path_arg:
        out_path = Path(out_path_arg)
    else:
        out_path_str = _ask("  Save to file (or press Enter to print to stdout)", '')
        out_path = Path(out_path_str) if out_path_str else None

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_str + '\n', encoding='utf-8')
        print()
        _hr('━')
        print(f"  ✔  Config saved to: {out_path}")
        _hr('━')
        print()
        print("  Next steps:")
        print(f"    coeval probe --config {out_path}  # verify model access")
        print(f"    coeval plan  --config {out_path}  # estimate cost")
        print(f"    coeval run   --config {out_path}  # run the experiment")
        print()
    else:
        print()
        _hr('─')
        print(yaml_str)
        _hr('─')
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provider_env_available(provider: str) -> bool:
    """Check if a provider has credentials available via environment variables."""
    import os
    checks = {
        'openai':       ('OPENAI_API_KEY',),
        'anthropic':    ('ANTHROPIC_API_KEY',),
        'gemini':       ('GEMINI_API_KEY', 'GOOGLE_API_KEY'),
        'huggingface':  ('HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN'),
        'openrouter':   ('OPENROUTER_API_KEY',),
        'azure_openai': ('AZURE_OPENAI_API_KEY',),
        'bedrock':      ('AWS_ACCESS_KEY_ID',),
        'vertex':       ('GOOGLE_CLOUD_PROJECT',),
    }
    return any(os.environ.get(var) for var in checks.get(provider, ()))
