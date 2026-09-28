from __future__ import annotations

import pytest

from faults.family_singularity import FAMILY_SPECS, family_for_task


@pytest.mark.parametrize(
    ("task_name", "family", "image"),
    (
        (
            "clinical_trial_matching_task_19",
            "clinical_trial_matching",
            "healthagentbench-clinical-trial-matching.sif",
        ),
        (
            "ct_abnormality_valid_137_a_1",
            "ct_abnormality",
            "healthagentbench-ct-abnormality.sif",
        ),
        (
            "ehr_data_quality_task_combined",
            "ehr_data_quality",
            "healthagentbench-ehr-data-quality.sif",
        ),
        (
            "ehr_event_modelling_new_acutemi",
            "ehr_event_modelling",
            "healthagentbench-ehr-event-modelling.sif",
        ),
        (
            "tumor_area_selection_pathology_slide_0001",
            "tumor_area_selection",
            "healthagentbench-pathology.sif",
        ),
        (
            "xray_report_correction_case_01",
            "xray_report_correction",
            "healthagentbench-xray-report.sif",
        ),
    ),
)
def test_family_for_task(task_name: str, family: str, image: str) -> None:
    spec = family_for_task(task_name)
    assert spec.name == family
    assert spec.image_name == image


def test_family_specs_keep_bootstrap_material_out_of_main_mounts() -> None:
    for spec in FAMILY_SPECS:
        main_targets = {mount.target for mount in spec.runtime_mounts}
        assert "/workspace" in main_targets
        assert all(mount.target not in main_targets for mount in spec.bootstrap_files)
        assert all(mount.target not in main_targets for mount in spec.cache_mounts)
        assert "/tests" not in main_targets


def test_family_for_task_rejects_unsupported_task() -> None:
    with pytest.raises(ValueError, match="matched none"):
        family_for_task("ehr_to_meds_etl")


def test_sensitive_data_mounts_are_read_only() -> None:
    data_quality = family_for_task("ehr_data_quality_task_combined")
    xray = family_for_task("xray_report_correction_case_01")

    assert any(
        mount.target == "/workspace/data" and mount.read_only
        for mount in data_quality.runtime_mounts
    )
    assert any(
        mount.target == "/data/patient" and mount.read_only
        for mount in xray.runtime_mounts
    )
