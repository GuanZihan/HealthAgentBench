"""Regression tests for the family-level experiment launchers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


# Execute a batch launcher in plan mode without starting Harbor or downloading data.
def run_plan(
    script_name: str,
    *,
    scenarios: str | None = None,
    replicate: str = "test-rep",
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PLAN_ONLY"] = "1"
    environment["REPLICATE"] = replicate
    if scenarios is None:
        environment.pop("SCENARIOS", None)
    else:
        environment["SCENARIOS"] = scenarios
    return subprocess.run(
        ["bash", script_name],
        cwd=BENCHMARK_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


# Verify that every user-facing family launcher resolves the full expected matrix.
@pytest.mark.parametrize(
    ("script_name", "family", "task_count", "scenario_count", "trial_count"),
    [
        ("run_ct_experiments.sh", "ct-abnormality", 10, 9, 180),
        ("run_xray_experiments.sh", "xray-report", 10, 9, 180),
        ("run_pathology_experiments.sh", "pathology", 10, 9, 180),
        (
            "run_clinical_trial_experiments.sh",
            "clinical-trial-matching",
            9,
            9,
            162,
        ),
        ("run_ehr_data_quality_experiments.sh", "ehr-data-quality", 8, 9, 144),
        (
            "run_ehr_event_modelling_experiments.sh",
            "ehr-event-modelling",
            6,
            9,
            108,
        ),
    ],
)
def test_shared_family_plan(
    script_name: str,
    family: str,
    task_count: int,
    scenario_count: int,
    trial_count: int,
) -> None:
    result = run_plan(script_name)

    assert result.returncode == 0, result.stderr
    assert (
        f"family={family} tasks={task_count} scenarios={scenario_count} "
        f"conditions=2 trials={trial_count} replicate=test-rep"
    ) in result.stdout
    assert result.stdout.count("[family-batch] task=") == trial_count


# Verify EHR-to-MEDS combines task-specific, permission, and evidence faults.
def test_ehr_to_meds_plan() -> None:
    result = run_plan("run_ehr_to_meds_experiments.sh")

    assert result.returncode == 0, result.stderr
    assert ("scenarios=8 conditions=2 trials=16 replicate=test-rep") in result.stdout
    assert result.stdout.count("[ehr-meds-batch] scenario=") == 16


# Verify scenario subsetting preserves adjacent matched pairs and replicate names.
def test_shared_family_subset_plan() -> None:
    result = run_plan(
        "run_xray_experiments.sh",
        scenarios="file-not-found missing-dependency",
        replicate="subset-rep",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "tasks=10 scenarios=2 conditions=2 trials=40 replicate=subset-rep"
    ) in result.stdout
    assert result.stdout.count("[family-batch] task=") == 40
    assert "subset-rep-control" in result.stdout
    assert "subset-rep-error" in result.stdout


# Reject typoed scenarios before any experiment process can start.
def test_invalid_scenario_is_rejected() -> None:
    result = run_plan(
        "run_pathology_experiments.sh",
        scenarios="not-a-scenario",
    )

    assert result.returncode == 2
    assert "Unsupported shared-family scenario" in result.stderr


# Verify the dedicated GPT matrix never requires telemetry or a local vLLM server.
def test_gpt_matrix_plan() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "PLAN_ONLY": "1",
            "MODEL": "gpt-5.4-mini",
            "REPLICATE_PREFIX": "gpt54-mini-test",
            "REPLICATES": "1",
            "FAMILIES": "ct",
            "SCENARIOS": "file-not-found",
        }
    )

    result = subprocess.run(
        ["bash", "run_gpt_matrix.sh"],
        cwd=BENCHMARK_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "model=gpt-5.4-mini replicate=gpt54-mini-test-rep1 telemetry=0"
        in result.stdout
    )
    assert "trials=20 replicate=gpt54-mini-test-rep1 telemetry=0" in result.stdout
