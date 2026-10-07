"""Tests for .github/workflows/ci.yml.

This workflow has NEVER been executed on GitHub -- this sandbox has no
network access and no `git remote` configured (confirmed: empty `git
remote -v`, and a direct request to github.com is denied by this
sandbox's network policy). These tests check the workflow's STRUCTURE
using PyYAML (available, no `dvc`/`ragas`/GitHub needed), which is
everything that can be honestly verified here. They do not and cannot
confirm the workflow actually passes/fails when GitHub itself runs it.
"""
from pathlib import Path

import yaml

CI_YAML_PATH = Path(".github/workflows/ci.yml")


def _load_ci_yaml() -> dict:
    with open(CI_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_ci_yaml_exists_and_is_valid_yaml():
    assert CI_YAML_PATH.exists()
    config = _load_ci_yaml()
    assert "jobs" in config


def test_ci_triggers_on_push_and_pull_request():
    config = _load_ci_yaml()
    triggers = config["on"]  # PyYAML resolves the quoted "on" key correctly
    assert "push" in triggers
    assert "pull_request" in triggers


def test_ci_has_a_test_job_with_setup_python():
    config = _load_ci_yaml()
    test_job = config["jobs"]["test"]
    uses_list = [s.get("uses", "") for s in test_job["steps"]]
    assert any("setup-python" in u for u in uses_list)
    assert any("checkout" in u for u in uses_list)


def test_ci_install_command_uses_project_pyproject_extras_not_a_separate_list():
    config = _load_ci_yaml()
    test_job = config["jobs"]["test"]
    install_steps = [s for s in test_job["steps"] if "Install project" in s.get("name", "")]
    assert install_steps
    install_cmd = install_steps[0]["run"]
    # must install via `pip install -e ".[...]"` (this project's own
    # pyproject.toml config), never a hand-written duplicate requirements list
    assert "pip install -e" in install_cmd
    assert ".[" in install_cmd


def test_ci_runs_the_real_test_suite():
    config = _load_ci_yaml()
    test_job = config["jobs"]["test"]
    run_commands = [s.get("run", "") for s in test_job["steps"]]
    assert any("pytest" in c for c in run_commands)
    assert any("run_tests.py" in c for c in run_commands)


def test_ci_runs_the_corpus_quality_gate():
    config = _load_ci_yaml()
    test_job = config["jobs"]["test"]
    run_commands = [s.get("run", "") for s in test_job["steps"]]
    assert any("check_corpus_quality.py" in c for c in run_commands)


def test_ci_has_a_ragas_quality_gate_job_that_depends_on_test():
    config = _load_ci_yaml()
    job = config["jobs"]["ragas-quality-gate"]
    assert job["needs"] == "test"
    run_commands = [s.get("run", "") for s in job["steps"]]
    assert any("ragas_gate.py" in c for c in run_commands)


def test_ci_ragas_gate_step_does_not_hardcode_a_fake_passing_score():
    config = _load_ci_yaml()
    job = config["jobs"]["ragas-quality-gate"]
    full_text = str(job)
    # guards against ever hand-writing a fabricated numeric result into
    # the workflow itself instead of computing one for real
    for forbidden in ("faithfulness: 0.", "score: 0.", "score=1.0"):
        assert forbidden not in full_text.lower()


def test_ci_has_a_dvc_pipeline_job_that_checks_for_the_source_pdf():
    config = _load_ci_yaml()
    job = config["jobs"]["dvc-pipeline"]
    run_commands = [s.get("run", "") for s in job["steps"]]
    assert any("dvc repro" in c for c in run_commands)
    assert any("civil_code.pdf" in c for c in run_commands)


def test_ci_does_not_reference_any_local_sandbox_path():
    config = _load_ci_yaml()
    full_text = str(config)
    # guards against ever hardcoding this sandbox's own filesystem layout
    assert "/home/claude" not in full_text
    assert "/mnt/user-data" not in full_text


def test_ci_install_steps_never_bypass_project_extras():
    # Day 7 audit: every pip install in this workflow must go through
    # this project's own pyproject.toml extras (`pip install -e ".[...]"`)
    # rather than a separate, hand-written package list that can drift
    # out of sync with pyproject.toml's pinned versions.
    config = _load_ci_yaml()
    for job in config["jobs"].values():
        for step in job["steps"]:
            run = step.get("run", "")
            for bad_pattern in ("pip install ruff", "pip install dvc", "pip install ragas"):
                assert bad_pattern not in run, f"found bypass: {bad_pattern!r} in {run!r}"


def test_pyproject_defines_extras_referenced_by_ci():
    # every extra name the workflow references must actually exist in
    # pyproject.toml -- guards against a typo'd or removed extra
    # silently breaking the CI install step.
    import re
    import tomllib

    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    extras = set(pyproject["project"]["optional-dependencies"].keys())

    config = _load_ci_yaml()
    referenced = set()
    for job in config["jobs"].values():
        for step in job["steps"]:
            run = step.get("run", "")
            for match in re.findall(r'\.\[([a-z,]+)\]', run):
                referenced.update(match.split(","))

    assert referenced  # sanity: the regex actually found something
    assert referenced.issubset(extras), f"CI references undefined extras: {referenced - extras}"
