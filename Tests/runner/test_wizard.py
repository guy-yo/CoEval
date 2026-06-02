"""Tests for the `coeval wizard` config generator.

The LLM generation path (`_generate_yaml` / `_refine_yaml`) requires a live
provider and is exercised by the end-to-end smoke, not here.  These tests lock
in the pure-logic pieces: `_try_validate` must accept a freshly generated config
(no run folder / meta.json yet) while still rejecting structurally broken ones.
"""
from runner.commands.wizard_cmd import _try_validate


_GOOD_CONFIG = """
experiment:
  id: ticket-urgency
  storage_folder: ./Runs/ticket-urgency
models:
  - name: gpt-4o-mini
    interface: openai
    parameters: {model: gpt-4o-mini, temperature: 0.7, max_tokens: 512}
    roles: [teacher, student]
  - name: claude-3-5-haiku
    interface: anthropic
    parameters: {model: claude-3-5-haiku, temperature: 0.0, max_tokens: 512}
    roles: [judge]
tasks:
  - name: classify_ticket_urgency
    description: Classify customer support tickets into urgency levels.
    output_description: A single urgency level (low, medium, or high).
    target_attributes:
      urgency_level: [low, medium, high]
    sampling: {target: [1, 1], nuance: [0, 0], total: 6}
    rubric:
      accuracy: "Is the classification correct?"
"""


def test_fresh_config_validates_clean():
    """A freshly generated config has no run folder / meta.json yet; those
    runtime-state checks must NOT surface as validation errors."""
    errors = _try_validate(_GOOD_CONFIG)
    assert errors == [], f"expected no errors, got: {errors}"


def test_runtime_state_errors_are_filtered():
    """Even though the storage folder does not exist and there is no meta.json,
    validation stays clean (the wizard writes the file before any run)."""
    errors = _try_validate(_GOOD_CONFIG)
    joined = " ".join(errors)
    assert "does not exist" not in joined
    assert "meta.json is missing" not in joined
    assert "--continue" not in joined


def test_yaml_parse_error_reported():
    errors = _try_validate("experiment: [unclosed")
    assert errors and "YAML parse error" in errors[0]


def test_structural_error_still_caught():
    """A config with no models is structurally invalid and must be rejected,
    proving the runtime-state filter did not swallow real errors."""
    bad = """
experiment:
  id: x
  storage_folder: ./Runs/x
models: []
tasks:
  - name: t
    description: d
    output_description: o
    target_attributes: {a: [1, 2]}
    sampling: {target: [1, 1], nuance: [0, 0], total: 2}
    rubric: {acc: "ok?"}
"""
    errors = _try_validate(bad)
    assert errors, "expected structural errors for an empty model list"
