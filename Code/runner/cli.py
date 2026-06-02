"""CLI entry point for CoEval subcommands.

Subcommands
-----------
run       Execute an evaluation experiment (EER).
probe     Standalone model availability probe (no experiment phases started).
plan      Standalone cost/time estimation (no experiment phases started).
status    Experiment progress dashboard; optionally fetch completed batch results.
repair    Scan experiment JSONL files for invalid records and mark them for
          re-generation; follow with `coeval run --continue` to regenerate.
describe  Generate a self-contained HTML summary of an experiment configuration:
          models, tasks, rubrics, phase plan, and estimated call budget.
wizard    Interactive LLM-assisted wizard: describe your goal in plain English
          and get a ready-to-run YAML configuration.
generate  Run phases 1-2 (attribute + rubric mapping) and write a materialized
          YAML config with static attributes and rubric ready for `coeval run`.
ingest    Inject downloaded benchmark data as a virtual teacher into an existing
          EES run; updates config.yaml so `coeval run --continue` includes the
          benchmark teacher in Phases 4-5.
models    List available text-generation models from each configured provider.
analyze   Analyze an EES experiment folder (EEA).

Provider key file
-----------------
A global provider key file (default: ~/.coeval/keys.yaml, or set COEVAL_KEYS_FILE
env var) stores API keys/credentials shared across experiments.  Any subcommand
that loads a config supports ``--keys PATH`` to point at a non-default key file.
Per-model ``access_key`` values in the experiment YAML always take precedence.
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config, validate_config
from .runner import print_execution_plan, run_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='coeval',
        description='CoEval — Self-evaluating LLM ensemble benchmarking system',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ---- coeval run ----
    run_p = sub.add_parser('run', help='Execute an evaluation experiment (EER)')
    run_p.add_argument(
        '--config', required=True, metavar='PATH',
        help='Path to the YAML configuration file',
    )
    run_p.add_argument(
        '--resume', metavar='EXPERIMENT_ID',
        help='Experiment ID to resume. Overrides experiment.resume_from in config.',
    )
    run_p.add_argument(
        '--continue', dest='continue_in_place', action='store_true',
        help=(
            'Continue a previously interrupted experiment in-place. '
            'Already-completed phases are skipped; partial phases resume '
            'from the last saved item (datapoint / response / evaluation).'
        ),
    )
    run_p.add_argument(
        '--only-models', metavar='MODEL_IDS', default=None,
        help=(
            'Comma-separated model IDs to activate; all others are skipped. '
            'Applied as teacher filter in Phase 3, student filter in Phase 4, '
            'and judge filter in Phase 5. '
            'Use with --continue to run OpenAI models in a separate parallel '
            'process while HF models run in the main process. '
            'When set, phase-completion markers are NOT written to meta.json '
            'so the main process is unaffected.'
        ),
    )
    run_p.add_argument(
        '--dry-run', action='store_true',
        help='Validate config and print execution plan without making LLM calls',
    )
    run_p.add_argument(
        '--probe',
        dest='probe_mode',
        choices=['disable', 'full', 'resume'],
        default=None,
        metavar='MODE',
        help=(
            'Model availability probe mode (overrides experiment.probe_mode in config). '
            '"full" (default) — probe all models; '
            '"resume" — probe only models needed for remaining phases; '
            '"disable" — skip the probe entirely.'
        ),
    )
    run_p.add_argument(
        '--probe-on-fail',
        dest='probe_on_fail',
        choices=['abort', 'warn'],
        default=None,
        metavar='MODE',
        help=(
            'What to do when a probed model is unavailable. '
            '"abort" (default) — stop the run immediately; '
            '"warn" — log a warning and continue (phases may fail later).'
        ),
    )
    run_p.add_argument(
        '--skip-probe', dest='skip_probe', action='store_true',
        help=(
            'Deprecated alias for --probe disable. '
            'Skip the model availability pre-flight check entirely.'
        ),
    )
    run_p.add_argument(
        '--estimate-only', dest='estimate_only', action='store_true',
        help=(
            'Run the cost and time estimator, print a breakdown, write '
            'cost_estimate.json, and exit without starting any pipeline phases. '
            'Sample calls are made to calibrate latency (unless --estimate-samples 0).'
        ),
    )
    run_p.add_argument(
        '--estimate-samples', dest='estimate_samples', type=int, default=None,
        metavar='N',
        help=(
            'Number of sample LLM calls per model for cost/time calibration '
            '(overrides experiment.estimate_samples in config; default 2). '
            'Set to 0 to use heuristics only without making real API calls.'
        ),
    )
    run_p.add_argument(
        '--log-level', metavar='LEVEL',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Override the log level from config',
    )
    run_p.add_argument(
        '--keys', metavar='PATH', default=None,
        help=(
            'Path to a provider key file (YAML). Overrides the default '
            '~/.coeval/keys.yaml / COEVAL_KEYS_FILE location. '
            'Keys in this file act as fallbacks; model-level access_key always wins.'
        ),
    )

    # ---- coeval probe ----
    probe_p = sub.add_parser(
        'probe',
        help='Test model availability without starting any experiment phases',
    )
    probe_p.add_argument(
        '--config', required=True, metavar='PATH',
        help='Path to the YAML configuration file',
    )
    probe_p.add_argument(
        '--probe',
        dest='probe_mode',
        choices=['disable', 'full', 'resume'],
        default=None,
        metavar='MODE',
        help=(
            'Model probe scope (overrides experiment.probe_mode in config). '
            '"full" (default) — test all models; '
            '"resume" — test only models needed for remaining phases; '
            '"disable" — skip probe (exits immediately).'
        ),
    )
    probe_p.add_argument(
        '--probe-on-fail',
        dest='probe_on_fail',
        choices=['abort', 'warn'],
        default=None,
        metavar='MODE',
        help=(
            'Behaviour when a model is unavailable. '
            '"abort" (default) — exit with code 2; '
            '"warn" — print a warning but exit 0.'
        ),
    )
    probe_p.add_argument(
        '--log-level', metavar='LEVEL',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Console log level (default: INFO)',
    )
    probe_p.add_argument(
        '--keys', metavar='PATH', default=None,
        help='Path to a provider key file (YAML); overrides default ~/.coeval/keys.yaml',
    )

    # ---- coeval plan ----
    plan_p = sub.add_parser(
        'plan',
        help='Estimate cost and runtime without starting any experiment phases',
    )
    plan_p.add_argument(
        '--config', required=True, metavar='PATH',
        help='Path to the YAML configuration file',
    )
    plan_p.add_argument(
        '--continue', dest='continue_in_place', action='store_true',
        help=(
            'Estimate only remaining work for an already-started experiment. '
            'Reads existing phase artifacts from storage to subtract completed '
            'items from the full budget.'
        ),
    )
    plan_p.add_argument(
        '--estimate-samples', dest='estimate_samples', type=int, default=None,
        metavar='N',
        help=(
            'Number of sample LLM calls per model for latency calibration '
            '(overrides experiment.estimate_samples; default 2). '
            'Set to 0 to use heuristics only without real API calls.'
        ),
    )
    plan_p.add_argument(
        '--log-level', metavar='LEVEL',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Console log level (default: INFO)',
    )
    plan_p.add_argument(
        '--keys', metavar='PATH', default=None,
        help='Path to a provider key file (YAML); overrides default ~/.coeval/keys.yaml',
    )

    # ---- coeval status ----
    status_p = sub.add_parser(
        'status',
        help='Show experiment progress and pending batch job status',
    )
    status_p.add_argument(
        '--run', required=True, metavar='PATH',
        help='Path to the experiment folder (EES run folder)',
    )
    status_p.add_argument(
        '--fetch-batches', dest='fetch_batches', action='store_true',
        help=(
            'Poll provider APIs for each tracked batch job and, for completed '
            'jobs, download and apply the results to the experiment storage. '
            'Phase 4 and Phase 5 results are applied automatically; '
            'Phase 3 results require a subsequent `--continue` run.'
        ),
    )

    # ---- coeval repair ----
    repair_p = sub.add_parser(
        'repair',
        help=(
            'Scan experiment JSONL files for invalid records '
            'and mark them for re-generation'
        ),
    )
    repair_p.add_argument(
        '--run', required=True, metavar='PATH',
        help='Path to the experiment folder (EES run folder)',
    )
    repair_p.add_argument(
        '--dry-run', action='store_true',
        help=(
            'Scan and report invalid records without modifying any files. '
            'Useful for auditing an experiment before committing to repair.'
        ),
    )
    repair_p.add_argument(
        '--stats', action='store_true',
        help=(
            'Print a compact per-phase summary table showing valid, invalid, '
            'and gap record counts. Exits without modifying any files.'
        ),
    )
    repair_p.add_argument(
        '--examples', type=int, default=5, metavar='N',
        help=(
            'Number of example records to show per issue group in the detailed '
            'report (default: 5; use 0 to suppress examples, -1 to show all).'
        ),
    )
    repair_p.add_argument(
        '--phase', type=int, choices=[3, 4, 5], metavar='PHASE',
        help='Restrict the scan and report to a single phase (3, 4, or 5).',
    )
    repair_p.add_argument(
        '--breakdown', action='store_true',
        help=(
            'Show a per-file breakdown table with valid/invalid/gap counts '
            'for every JSONL file. Can be combined with --stats.'
        ),
    )
    repair_p.add_argument(
        '--show-valid', type=int, default=0, metavar='N', dest='show_valid',
        help=(
            'Show N example valid records per phase for spot-checking data '
            'quality and format (default: 0 = disabled).'
        ),
    )

    # ---- coeval generate ----
    gen_p = sub.add_parser(
        'generate',
        help=(
            'Run phases 1-2 (attribute + rubric mapping) and write a '
            'materialized YAML config with static values ready for `coeval run`'
        ),
    )
    gen_p.add_argument(
        '--config', required=True, metavar='PATH',
        help='Path to the draft YAML configuration file',
    )
    gen_p.add_argument(
        '--out', required=True, metavar='PATH',
        help='Output path for the materialized YAML design file',
    )
    gen_p.add_argument(
        '--probe',
        dest='probe_mode',
        choices=['disable', 'full', 'resume'],
        default=None,
        metavar='MODE',
        help=(
            'Model availability probe mode before generation. '
            '"full" (default) — probe all teacher models; '
            '"disable" — skip probe.'
        ),
    )
    gen_p.add_argument(
        '--probe-on-fail',
        dest='probe_on_fail',
        choices=['abort', 'warn'],
        default=None,
        metavar='MODE',
        help=(
            'What to do when a probed model is unavailable. '
            '"abort" (default) — stop immediately; '
            '"warn" — log a warning and continue.'
        ),
    )
    gen_p.add_argument(
        '--log-level', metavar='LEVEL',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Console log level (default: INFO)',
    )
    gen_p.add_argument(
        '--keys', metavar='PATH', default=None,
        help='Path to a provider key file (YAML); overrides default ~/.coeval/keys.yaml',
    )

    # ---- coeval describe ----
    describe_p = sub.add_parser(
        'describe',
        help=(
            'Generate a self-contained HTML summary of an experiment configuration: '
            'models, tasks, rubrics, phase plan, and estimated call budget.'
        ),
    )
    describe_p.add_argument(
        '--config', required=True, metavar='PATH',
        help='Path to the YAML configuration file',
    )
    describe_p.add_argument(
        '--out', metavar='PATH', default=None,
        help=(
            'Output HTML file path '
            '(default: <config_stem>_description.html next to the config file)'
        ),
    )
    describe_p.add_argument(
        '--no-open', action='store_true',
        help='Do not open the HTML file in the default browser after writing',
    )
    describe_p.add_argument(
        '--keys', metavar='PATH', default=None,
        help='Path to a provider key file (YAML); overrides default ~/.coeval/keys.yaml',
    )
    describe_p.add_argument(
        '--probe', action='store_true', default=False,
        help=(
            'Run 1 sample API call per model to measure real latency and '
            'throughput. Results are shown in a "Provider Budget Probe" section '
            'of the HTML output. Off by default (makes describe a live network call).'
        ),
    )

    # ---- coeval wizard ----
    wizard_p = sub.add_parser(
        'wizard',
        help=(
            'Interactive LLM-assisted experiment wizard: describe your goal in '
            'plain English and get a ready-to-run YAML configuration'
        ),
    )
    wizard_p.add_argument(
        '--out', metavar='PATH', default=None,
        help='Output path for the generated YAML config file (default: prompt interactively)',
    )
    wizard_p.add_argument(
        '--model', metavar='MODEL_ID', default=None,
        help=(
            'Model to use for config generation (default: best available from key file). '
            'Examples: gpt-4o-mini, claude-3-5-haiku-20241022, gemini-2.0-flash'
        ),
    )
    wizard_p.add_argument(
        '--keys', metavar='PATH', default=None,
        help='Path to a provider key file (YAML); overrides default ~/.coeval/keys.yaml',
    )
    wizard_p.add_argument(
        '--objective', metavar='TEXT', default=None,
        help=(
            'Non-interactive: a one-line high-level objective; the LLM generates and '
            'auto-validates a full config in one shot (no questions). Writes to --out '
            'or stdout. Example: --objective "rank LLMs on legal contract-clause classification"'
        ),
    )
    wizard_p.add_argument(
        '--items', type=int, default=None,
        help='Non-interactive mode: items per task per teacher (default 12)',
    )
    wizard_p.add_argument(
        '--models', metavar='LIST', default=None,
        help="Non-interactive mode: preferred models, e.g. 'gpt-4o-mini, claude-3-5-haiku' (default: LLM decides)",
    )
    wizard_p.add_argument(
        '--storage-folder', dest='storage_folder', metavar='PATH', default=None,
        help='Non-interactive mode: experiment storage folder (default ./Runs)',
    )

    # ---- coeval ingest ----
    ingest_p = sub.add_parser(
        'ingest',
        help=(
            'Inject downloaded benchmark data as a virtual teacher into an existing '
            'EES run. Updates config.yaml so `coeval run --continue` runs Phases 4–5 '
            'on the benchmark teacher.'
        ),
    )
    ingest_p.add_argument(
        '--run', required=True, metavar='PATH',
        help='Path to the existing EES experiment folder',
    )
    ingest_p.add_argument(
        '--benchmarks', nargs='+', required=True, metavar='NAME',
        help=(
            'One or more benchmark names to ingest '
            '(e.g. mmlu hellaswag humaneval). '
            'Available: mmlu, hellaswag, truthfulqa, humaneval, medqa, gsm8k'
        ),
    )
    ingest_p.add_argument(
        '--data-dir', dest='data_dir', default='stdbenchmarks/data', metavar='PATH',
        help=(
            'Directory containing downloaded benchmark JSONL files '
            '(default: stdbenchmarks/data). '
            'Run `python stdbenchmarks/download_benchmarks.py` first.'
        ),
    )
    ingest_p.add_argument(
        '--split', default=None, metavar='SPLIT',
        help='Dataset split to ingest (default: per-benchmark default, usually "test")',
    )
    ingest_p.add_argument(
        '--limit', type=int, default=None, metavar='N',
        help='Maximum number of items to ingest per benchmark (default: all)',
    )
    ingest_p.add_argument(
        '--task-name', dest='task_name', default=None, metavar='NAME',
        help=(
            'Override the CoEval task name (default: benchmark name). '
            'Useful when mapping a benchmark onto an existing task.'
        ),
    )
    ingest_p.add_argument(
        '--verbose', action='store_true',
        help='Print progress every 100 items',
    )

    # ---- coeval models ----
    models_p = sub.add_parser(
        'models',
        help='List available text-generation models from each configured provider',
    )
    models_p.add_argument(
        '--providers', metavar='LIST', default=None,
        help=(
            'Comma-separated list of providers to query '
            '(e.g. "openai,anthropic"). Default: all providers with credentials.'
        ),
    )
    models_p.add_argument(
        '--verbose', action='store_true',
        help='Show additional model details (context window, capabilities, etc.)',
    )
    models_p.add_argument(
        '--keys', metavar='PATH', default=None,
        help='Path to a provider key file (YAML); overrides default ~/.coeval/keys.yaml',
    )

    # ---- coeval analyze ---- (REQ-A-8.1)
    analyze_p = sub.add_parser('analyze', help='Analyze an EES experiment (EEA)')
    analyze_sub = analyze_p.add_subparsers(dest='subcommand', required=True)

    _SUBCOMMANDS = [
        ('complete-report',   'Excel workbook with all slice/aggregate data'),
        ('score-distribution','HTML: score distribution by aspect, model, attribute'),
        ('teacher-report',    'HTML: teacher differentiation scores'),
        ('judge-report',      'HTML: judge agreement and reliability scores'),
        ('student-report',    'HTML: student model performance report'),
        ('interaction-matrix','HTML: teacher-student interaction heatmap'),
        ('judge-consistency', 'HTML: within-judge consistency analysis'),
        ('coverage-summary',  'HTML: EES artifact coverage and error breakdown'),
        ('robust-summary',    'HTML: robust student ranking with filtered datapoints'),
        ('export-benchmark',  'JSONL/Parquet: export robust benchmark datapoints'),
        ('summary-report',    'HTML: interactive single-page summary dashboard'),
        ('ensemble-ablation', 'HTML: EXP-002 ensemble-size reliability ablation'),
        ('verbosity-bias',    'HTML: EXP-004 response-length vs score correlation'),
        ('rubric-overlap',    'HTML: EXP-005 cross-task rubric criterion overlap'),
        ('all',               'Generate all HTML reports + Excel into subdirectories'),
    ]

    for sc_name, sc_help in _SUBCOMMANDS:
        sc_p = analyze_sub.add_parser(sc_name, help=sc_help)
        sc_p.add_argument('--run', required=True, metavar='PATH',
                          help='Path to the EES experiment folder')
        sc_p.add_argument('--out', required=True, metavar='PATH',
                          help='Output path (file for Excel/JSONL; folder for HTML/all)')
        sc_p.add_argument('--partial-ok', action='store_true',
                          help='Allow analysis on in-progress experiments without warning')
        sc_p.add_argument('--log-level', metavar='LEVEL',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          default='INFO', help='Log level')
        # Robust filtering options (apply to robust-summary, export-benchmark, all)
        if sc_name in ('robust-summary', 'export-benchmark', 'all'):
            sc_p.add_argument('--judge-selection', default='top_half',
                              choices=['top_half', 'all'],
                              help='Judge selection for robust filtering (default: top_half)')
            sc_p.add_argument('--agreement-metric', default='spa',
                              choices=['spa', 'wpa', 'kappa'],
                              help='Agreement metric for judge ranking (default: spa)')
            sc_p.add_argument('--agreement-threshold', type=float, default=1.0,
                              metavar='FLOAT',
                              help='Min judge-consistency fraction theta (default: 1.0)')
            sc_p.add_argument('--teacher-score-formula', default='v1',
                              choices=['v1', 's2', 'r3'],
                              help='Teacher score formula for T* selection (default: v1)')
        if sc_name == 'export-benchmark':
            sc_p.add_argument('--benchmark-format', default='jsonl',
                              choices=['jsonl', 'parquet'],
                              help='Output format (default: jsonl)')
        # Experiment subcommands (EXP-002 / EXP-004 / EXP-005)
        if sc_name in ('ensemble-ablation', 'verbosity-bias'):
            sc_p.add_argument('--judges', nargs='*', default=None, metavar='MODEL',
                              help='Judge models in addition order (default: all judges)')
            sc_p.add_argument('--tasks', nargs='*', default=None, metavar='TASK',
                              help='Restrict to these tasks (default: all)')
        if sc_name == 'verbosity-bias':
            sc_p.add_argument('--method', default='pearson',
                              choices=['pearson', 'spearman'],
                              help='Correlation method (default: pearson)')
        if sc_name == 'rubric-overlap':
            sc_p.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                              metavar='NAME',
                              help='sentence-transformers model (default: all-MiniLM-L6-v2)')

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == 'run':
        _cmd_run(args)
    elif args.command == 'probe':
        from .commands.probe_cmd import cmd_probe
        cmd_probe(args)
    elif args.command == 'plan':
        from .commands.plan_cmd import cmd_plan
        cmd_plan(args)
    elif args.command == 'status':
        from .commands.status_cmd import cmd_status
        cmd_status(args)
    elif args.command == 'repair':
        from .commands.repair_cmd import cmd_repair
        cmd_repair(args)
    elif args.command == 'describe':
        from .commands.describe_cmd import cmd_describe
        cmd_describe(args)
    elif args.command == 'wizard':
        from .commands.wizard_cmd import cmd_wizard
        cmd_wizard(args)
    elif args.command == 'generate':
        from .commands.generate_cmd import cmd_generate
        cmd_generate(args)
    elif args.command == 'ingest':
        from .commands.ingest_cmd import cmd_ingest
        cmd_ingest(args)
    elif args.command == 'models':
        from .commands.models_cmd import cmd_models
        cmd_models(args)
    elif args.command == 'analyze':
        _cmd_analyze(args)


def _cmd_run(args: argparse.Namespace) -> None:
    # Load config (pass provider key file if supplied)
    try:
        cfg = load_config(args.config, keys_file=getattr(args, 'keys', None))
    except Exception as exc:
        print(f"ERROR: Failed to load config '{args.config}': {exc}", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides
    if args.resume:
        cfg.experiment.resume_from = args.resume
    if args.log_level:
        cfg.experiment.log_level = args.log_level

    # Validate — report all errors, exit 1 if any (REQ-8.1.5)
    errors = validate_config(cfg, continue_in_place=args.continue_in_place)
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for err in errors:
            print(f"  • {err}", file=sys.stderr)
        sys.exit(1)

    # Print execution plan (always, not just dry-run)
    print_execution_plan(cfg)

    if args.dry_run:
        print("Dry-run: config valid. Exiting without making LLM calls.")
        sys.exit(0)

    only_models: set[str] | None = None
    if args.only_models:
        only_models = {m.strip() for m in args.only_models.split(',') if m.strip()}

    exit_code = run_experiment(
        cfg,
        dry_run=False,
        continue_in_place=args.continue_in_place,
        only_models=only_models,
        skip_probe=args.skip_probe,
        probe_mode=getattr(args, 'probe_mode', None),
        probe_on_fail=getattr(args, 'probe_on_fail', None),
        estimate_only=getattr(args, 'estimate_only', False),
        estimate_samples=getattr(args, 'estimate_samples', None),
    )
    sys.exit(exit_code)


def _cmd_analyze(args: argparse.Namespace) -> None:
    from analyzer.main import run_analyze

    # Build robust kwargs (only if the subcommand supports them)
    robust_supported = args.subcommand in ('robust-summary', 'export-benchmark', 'all')

    exit_code = run_analyze(
        run_path=args.run,
        out_path=args.out,
        subcommand=args.subcommand,
        judge_selection=getattr(args, 'judge_selection', 'top_half') if robust_supported else 'top_half',
        agreement_metric=getattr(args, 'agreement_metric', 'spa') if robust_supported else 'spa',
        agreement_threshold=getattr(args, 'agreement_threshold', 1.0) if robust_supported else 1.0,
        teacher_score_formula=getattr(args, 'teacher_score_formula', 'v1') if robust_supported else 'v1',
        benchmark_format=getattr(args, 'benchmark_format', 'jsonl'),
        partial_ok=getattr(args, 'partial_ok', False),
        log_level=getattr(args, 'log_level', 'INFO'),
        judges=getattr(args, 'judges', None),
        tasks=getattr(args, 'tasks', None),
        method=getattr(args, 'method', 'pearson'),
        embedding_model=getattr(args, 'embedding_model', 'all-MiniLM-L6-v2'),
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
