from __future__ import annotations

import base64
import csv
import functools
import gzip
import http.server
import json
import logging
import os
import shlex
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from harbor.environments.base import ExecResult
from harbor.environments.singularity import SingularityEnvironment
from harbor.models.trial.paths import TrialPaths

from faults.codex_agent import TextOnlyCodex
from faults.text_only_proxy import sanitize_request_body
from faults.evidence.modalities import (
    apply_ct_evidence,
    apply_ehr_data_quality_evidence,
    apply_ehr_event_evidence,
    apply_xray_evidence,
    prepare_pathology_evidence,
)
from faults.evidence.prepare_clinical_recipes import (
    RecipePreparationError,
    _api_key_validation_error,
    build_edits,
    build_partial_edits,
    load_topics,
    prepare_recipe,
    select_evidence,
)
from faults.evidence.recipes import (
    EVIDENCE_SCENARIOS,
    EvidenceEdit,
    EvidenceRecipe,
    EvidenceRecipeError,
    apply_evidence_recipe,
    sha256_bytes,
)
from faults.family_singularity import family_for_task
from faults.inject import (
    CORRUPTION_MARKER,
    PERMISSION_SCENARIOS,
    Dependency,
    InjectionError,
    apply_fault,
    apply_permission_fault,
    file_state,
    permission_limited_command,
    select_target,
    write_dependency_runtime,
)
from faults.privilege import PrivilegeBroker, write_privilege_runtime
from faults.run_fault import build_harbor_command
from faults.selective_proxy import SelectiveProxyServer, create_proxy_server
from faults.singularity_fault import (
    CHECKSUM_FILE,
    CORRUPTION_MARKER as SIF_CORRUPTION_MARKER,
    SCENARIO,
    SUPPORTED_SCENARIOS,
    TARGET,
    EhrToMedsFaultEnvironment,
    FaultSetupError,
    build_fault_command,
    parse_fault_result,
    validate_fault_result,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


# Refuse a pasted shell-command block before it can become an HTTP Bearer header.
def test_clinical_recipe_rejects_malformed_api_key() -> None:
    error = _api_key_validation_error("export OPENAI_API_KEY\recho")

    assert error == "OPENAI_API_KEY contains whitespace or control characters"
    assert _api_key_validation_error("sk-proj-abcdefghijklmnopqrstuvwxyz") is None


# Match evidence-recipe source bytes to the bootstrap-rendered topic text.
def test_evidence_topic_loading_normalizes_bootstrap_whitespace(
    tmp_path: Path,
) -> None:
    topics_path = tmp_path / "topics.xml"
    topics_path.write_text(
        '<topics><topic number="19">A patient with\n  stage IV NSCLC.</topic></topics>',
        encoding="utf-8",
    )

    assert load_topics(topics_path) == {19: "A patient with stage IV NSCLC."}


# Anchor a repeated short clinical value through its unique surrounding phrase.
def test_clinical_recipe_anchors_repeated_core_in_unique_natural_span() -> None:
    note = "Prior stage IV disease; now diagnosed with stage IV NSCLC."
    natural_text = "with stage IV NSCLC"

    edits = build_edits(
        note,
        {
            "selections": [
                {
                    "text": "stage IV",
                    "natural_text": natural_text,
                    "low_quality_text": "an unverified disease stage",
                    "category": "disease_stage",
                    "importance": "eligibility_critical",
                }
            ]
        },
        maximum=2,
    )

    natural_start = note.index(natural_text)
    assert edits[0]["start"] == natural_start + natural_text.index("stage IV")
    assert edits[0]["natural_start"] == natural_start


# Canonicalize one case-only model mismatch back to exact source bytes and hashes.
def test_clinical_recipe_resolves_unique_case_only_span() -> None:
    note = "He was found to have Richter's transformation."

    edits = build_edits(
        note,
        {
            "selections": [
                {
                    "text": "richter's transformation",
                    "natural_text": "he was found to have Richter's transformation",
                    "low_quality_text": "an unverified transformation diagnosis",
                    "category": "diagnosis",
                    "importance": "eligibility_critical",
                }
            ]
        },
        maximum=1,
    )

    selected = "Richter's transformation"
    natural_text = "He was found to have Richter's transformation"
    assert edits[0]["start"] == note.index(selected)
    assert edits[0]["source_text_sha256"] == sha256_bytes(selected.encode())
    assert edits[0]["natural_text_sha256"] == sha256_bytes(natural_text.encode())


# Reject a broad enclosing paragraph even when every source substring is exact.
def test_clinical_recipe_rejects_broad_natural_deletion() -> None:
    prefix = "The patient has multiple unrelated medical facts. " * 4
    note = f"{prefix}The selected diagnosis is C diff."

    with pytest.raises(RecipePreparationError, match="too broad"):
        build_edits(
            note,
            {
                "selections": [
                    {
                        "text": "C diff",
                        "natural_text": note,
                        "low_quality_text": "an unverified infection diagnosis",
                        "category": "diagnosis",
                        "importance": "eligibility_critical",
                    }
                ]
            },
            maximum=1,
        )


# Keep one valid selection while rejecting a reordered non-verbatim core span.
def test_clinical_recipe_salvages_only_valid_selection() -> None:
    note = "Prior C diff infection. Treated with po vanco."
    response = {
        "selections": [
            {
                "text": "C diff",
                "natural_text": "Prior C diff infection.",
                "low_quality_text": "an unverified infection diagnosis",
                "category": "diagnosis",
                "importance": "eligibility_critical",
            },
            {
                "text": "Vanco po",
                "natural_text": "Treated with po vanco.",
                "low_quality_text": "an unverified antimicrobial treatment",
                "category": "treatment",
                "importance": "eligibility_critical",
            },
        ]
    }

    accepted, rejected = build_partial_edits(note, response, maximum=2)

    assert len(accepted) == 1
    assert note[accepted[0]["start"] : accepted[0]["end"]] == "C diff"
    assert rejected[0]["selection"] == 2
    assert "case-only correction" in rejected[0]["validation_error"]


# Retain a rejected Qwen selection and accept its deterministic corrected retry.
def test_clinical_recipe_retries_invalid_model_selection(tmp_path: Path) -> None:
    note = "Prior stage IV disease; now diagnosed with stage IV NSCLC."
    responses = [
        json.dumps(
            {
                "selections": [
                    {
                        "text": "stage IV",
                        "natural_text": "stage IV",
                        "low_quality_text": "an unverified disease stage",
                        "category": "disease_stage",
                        "importance": "eligibility_critical",
                    }
                ]
            }
        ),
        json.dumps(
            {
                "selections": [
                    {
                        "text": "stage IV",
                        "natural_text": "with stage IV NSCLC",
                        "low_quality_text": "an unverified disease stage",
                        "category": "disease_stage",
                        "importance": "eligibility_critical",
                    }
                ]
            }
        ),
    ]

    # Reproduce only the OpenAI client surface consumed by the offline generator.
    class FakeCompletions:
        def create(self, **_: object) -> SimpleNamespace:
            message = SimpleNamespace(content=responses.pop(0))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    recipes = tmp_path / "recipes"
    generated = tmp_path / "generated"

    recipe_path = prepare_recipe(
        client=client,
        model="qwen-coder",
        model_revision="revision",
        api_mode="chat-completions",
        reasoning_effort="medium",
        prompt="select evidence",
        task="clinical_trial_matching_task_test",
        topic_id=1,
        note=note,
        maximum=1,
        severity="evidence-insufficient",
        recipe_directory=recipes,
        generated_directory=generated,
        max_retries=2,
    )

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    audit = json.loads(
        (generated / "clinical_trial_matching_task_test.json").read_text(
            encoding="utf-8"
        )
    )
    assert recipe["generator"]["attempt_count"] == 2
    assert audit["attempts"][0]["validation_error"] is not None
    assert audit["attempts"][1]["validation_error"] is None
    assert responses == []


# Mark a partially valid final response explicitly in recipe and audit provenance.
def test_clinical_recipe_records_partial_salvage(tmp_path: Path) -> None:
    note = "Prior C diff infection. Treated with po vanco."
    response = json.dumps(
        {
            "selections": [
                {
                    "text": "C diff",
                    "natural_text": "Prior C diff infection.",
                    "low_quality_text": "an unverified infection diagnosis",
                    "category": "diagnosis",
                    "importance": "eligibility_critical",
                },
                {
                    "text": "Vanco po",
                    "natural_text": "Treated with po vanco.",
                    "low_quality_text": "an unverified antimicrobial treatment",
                    "category": "treatment",
                    "importance": "eligibility_critical",
                },
            ]
        }
    )

    # Reproduce one terminal OpenAI-compatible response without another model call.
    class FakeCompletions:
        def create(self, **_: object) -> SimpleNamespace:
            message = SimpleNamespace(content=response)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    recipes = tmp_path / "recipes"
    generated = tmp_path / "generated"

    recipe_path = prepare_recipe(
        client=client,
        model="qwen-coder",
        model_revision="revision",
        api_mode="chat-completions",
        reasoning_effort="medium",
        prompt="select evidence",
        task="clinical_trial_matching_task_test",
        topic_id=1,
        note=note,
        maximum=2,
        severity="evidence-insufficient",
        recipe_directory=recipes,
        generated_directory=generated,
        max_retries=0,
    )

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    audit = json.loads(
        (generated / "clinical_trial_matching_task_test.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(recipe["edits"]) == 1
    assert recipe["generator"]["partial_salvage"] is True
    assert recipe["generator"]["rejected_selection_count"] == 1
    assert audit["attempts"][0]["partial_salvage"]["rejected_selection_count"] == 1


# Refuse to erase a human-curated draft before making any preparation-model call.
def test_clinical_recipe_refuses_locked_curated_draft(tmp_path: Path) -> None:
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    recipe_path = recipes / "clinical_trial_matching_task_test.json"
    recipe_path.write_text(
        json.dumps(
            {
                "status": "draft",
                "curation": {
                    "locked": True,
                    "method": "manual_fact_level_revision",
                },
            }
        ),
        encoding="utf-8",
    )
    client = SimpleNamespace()

    with pytest.raises(RecipePreparationError, match="locked curated recipe"):
        prepare_recipe(
            client=client,
            model="gpt-5.5-2026-04-23",
            model_revision="gpt-5.5-2026-04-23",
            api_mode="responses",
            reasoning_effort="medium",
            prompt="select evidence",
            task="clinical_trial_matching_task_test",
            topic_id=1,
            note="A patient note.",
            maximum=2,
            severity="evidence-insufficient",
            recipe_directory=recipes,
            generated_directory=tmp_path / "generated",
            max_retries=0,
        )


# Pin GPT-5.5 preparation to Responses structured output with server-side storage off.
def test_clinical_recipe_gpt55_uses_responses_structured_output() -> None:
    captured: dict[str, object] = {}
    content = json.dumps(
        {
            "selections": [
                {
                    "text": "stage IV",
                    "natural_text": "with stage IV disease",
                    "low_quality_text": "with an unverified disease stage",
                    "category": "disease_stage",
                    "importance": "eligibility_critical",
                }
            ]
        }
    )

    # Reproduce only the Responses surface and usage metadata consumed by the generator.
    class FakeResponses:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            usage = SimpleNamespace(model_dump=lambda **_: {"input_tokens": 12})
            return SimpleNamespace(
                id="resp_test",
                model="gpt-5.5-2026-04-23",
                output_text=content,
                usage=usage,
            )

    client = SimpleNamespace(responses=FakeResponses())
    result, metadata = select_evidence(
        client,
        model="gpt-5.5-2026-04-23",
        prompt="select evidence",
        task="clinical_trial_matching_task_test",
        note="A patient with stage IV disease.",
        maximum=2,
        api_mode="responses",
        reasoning_effort="medium",
    )

    assert result == content
    assert captured["store"] is False
    assert captured["reasoning"] == {"effort": "medium"}
    assert captured["text"]["format"]["type"] == "json_schema"
    assert metadata["response_id"] == "resp_test"


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("remove", None),
        ("corrupt", CORRUPTION_MARKER),
        ("truncate", b""),
    ],
)
def test_apply_error_faults(
    tmp_path: Path, action: str, expected: bytes | None
) -> None:
    target = tmp_path / "input.dat"
    target.write_bytes(b"valid input\n")

    apply_fault(action=action, condition="error", target=target)

    if expected is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == expected


@pytest.mark.parametrize("action", ["remove", "corrupt", "truncate"])
def test_control_condition_is_read_only(tmp_path: Path, action: str) -> None:
    target = tmp_path / "input.dat"
    original = b"valid input\n"
    target.write_bytes(original)

    apply_fault(action=action, condition="control", target=target)

    assert target.read_bytes() == original


# Verify permission injection changes only the error copy and records exact mode bits.
def test_permission_fault_is_matched_and_recoverable(tmp_path: Path) -> None:
    control = tmp_path / "control.dat"
    error = tmp_path / "error.dat"
    for target in (control, error):
        target.write_bytes(b"clinical input\n")
        target.chmod(0o640)

    apply_permission_fault(condition="control", target=control)
    apply_permission_fault(condition="error", target=error)

    assert stat.S_IMODE(control.stat().st_mode) == 0o640
    assert stat.S_IMODE(error.stat().st_mode) == 0
    assert file_state(error)["mode"] == "0000"
    assert file_state(error)["owner_read"] is False

    # Model the intended owner-level recovery path used inside the trial copy.
    error.chmod(error.stat().st_mode | stat.S_IRUSR)
    assert error.read_bytes() == b"clinical input\n"


# Drop only the fakeroot capabilities that would invalidate permission errors.
def test_permission_limited_command_is_shell_safe() -> None:
    original = "printf '%s\\n' \"$HOME\""
    command = permission_limited_command(original)
    arguments = shlex.split(command)

    assert command.startswith("/usr/bin/setpriv --no-new-privs ")
    assert "--bounding-set=-dac_override,-dac_read_search,-fowner,-chown" in command
    assert arguments[-3:] == ["/bin/bash", "-c", original]

    recoverable = permission_limited_command(original, protect_metadata=False)
    assert "--bounding-set=-dac_override,-dac_read_search " in recoverable
    assert "fowner" not in recoverable


# Build a minimal approved recipe without retaining raw medical text in its metadata.
def _evidence_recipe(source: bytes) -> EvidenceRecipe:
    text = source.decode("utf-8")
    selections = (
        (
            "e01",
            "EGFR mutation",
            "with EGFR mutation",
            "biomarker",
            "an unverified molecular result",
        ),
        (
            "e02",
            "osimertinib",
            "received osimertinib",
            "treatment",
            "an unverified prior treatment",
        ),
    )
    edits = []
    for edit_id, selected, natural_selected, category, low_quality in selections:
        start = text.index(selected)
        natural_start = text.index(natural_selected)
        edits.append(
            EvidenceEdit(
                edit_id=edit_id,
                start=start,
                end=start + len(selected),
                category=category,
                importance="eligibility_critical",
                source_text_sha256=sha256_bytes(selected.encode("utf-8")),
                natural_start=natural_start,
                natural_end=natural_start + len(natural_selected),
                natural_text_sha256=sha256_bytes(natural_selected.encode("utf-8")),
                low_quality_replacement=low_quality,
            )
        )
    return EvidenceRecipe(
        task="clinical_trial_matching_task_test",
        source_sha256=sha256_bytes(source),
        status="approved",
        severity="evidence-insufficient",
        generator={"model": "qwen-coder", "temperature": 0, "seed": 0},
        edits=tuple(edits),
    )


# Keep controls byte-identical while applying three deterministic error renderings.
@pytest.mark.parametrize(
    ("scenario", "expected_fragment"),
    [
        (
            "evidence-explicit-missing",
            b"[BIOMARKER REDACTED] received [TREATMENT REDACTED]",
        ),
        ("evidence-natural-missing", b"The patient."),
        (
            "evidence-low-quality",
            b"an unverified molecular result received an unverified prior treatment",
        ),
    ],
)
def test_evidence_fault_renderings(
    scenario: str,
    expected_fragment: bytes,
) -> None:
    source = b"The patient with EGFR mutation received osimertinib."
    recipe = _evidence_recipe(source)

    control, control_details = apply_evidence_recipe(
        source,
        recipe,
        scenario=scenario,
        condition="control",
    )
    error, error_details = apply_evidence_recipe(
        source,
        recipe,
        scenario=scenario,
        condition="error",
    )

    assert control == source
    assert expected_fragment in error
    assert control_details["edit_count"] == 2
    assert error_details["edit_ids"] == ["e01", "e02"]


# Refuse stale recipes instead of silently editing a changed clinical note.
def test_evidence_fault_rejects_source_drift() -> None:
    source = b"The patient with EGFR mutation received osimertinib."
    recipe = _evidence_recipe(source)

    with pytest.raises(EvidenceRecipeError, match="Source digest"):
        apply_evidence_recipe(
            source + b" Updated.",
            recipe,
            scenario="evidence-explicit-missing",
            condition="error",
        )


# Preserve CT controls byte-for-byte and encode missing voxels as NaN in errors.
def test_ct_evidence_adapter_is_matched_and_readable(tmp_path: Path) -> None:
    import nibabel as nib
    import numpy as np

    target = tmp_path / "scan.nii.gz"
    source = np.arange(16 * 16 * 20, dtype=np.int16).reshape((16, 16, 20))
    nib.save(nib.Nifti1Image(source, np.eye(4)), target)
    original = target.read_bytes()

    control = apply_ct_evidence(
        target,
        condition="control",
        scenario="evidence-explicit-missing",
    )
    assert target.read_bytes() == original
    assert control.details["selected_fraction"] == pytest.approx(0.2)

    error = apply_ct_evidence(
        target,
        condition="error",
        scenario="evidence-explicit-missing",
    )
    degraded = nib.load(target).get_fdata()
    assert np.isnan(degraded[:, :, 8:12]).all()
    assert error.details["transform"] == "replace-selected-voxels-with-nan"


# Restrict X-ray degradation to target-study images and preserve prior studies.
def test_xray_evidence_adapter_targets_latest_study(tmp_path: Path) -> None:
    from PIL import Image

    prior = tmp_path / "study_01" / "view_01.jpg"
    target = tmp_path / "study_02" / "view_01.jpg"
    prior.parent.mkdir()
    target.parent.mkdir()
    Image.new("L", (100, 120), color=40).save(prior)
    Image.new("L", (100, 120), color=180).save(target)
    prior_bytes = prior.read_bytes()
    target_bytes = target.read_bytes()

    application = apply_xray_evidence(
        tmp_path,
        condition="error",
        scenario="evidence-natural-missing",
    )

    assert prior.read_bytes() == prior_bytes
    assert target.read_bytes() != target_bytes
    assert application.details["target_view_count"] == 1


# Apply record-level data-quality deletion only to deterministic hidden clusters.
def test_ehr_data_quality_evidence_adapter_deletes_selected_records(
    tmp_path: Path,
) -> None:
    csv_root = tmp_path / "csv"
    csv_root.mkdir()
    table = csv_root / "labevents.csv.gz"
    with gzip.open(table, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["_row_id", "valuenum"])
        writer.writeheader()
        for index in range(3):
            writer.writerow({"_row_id": f"row-{index}", "valuenum": index + 1})
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["table", "row_id", "field", "cluster_id"],
        )
        writer.writeheader()
        for index in range(3):
            writer.writerow(
                {
                    "table": "labevents",
                    "row_id": f"row-{index}",
                    "field": "valuenum",
                    "cluster_id": f"cluster-{index}",
                }
            )

    application = apply_ehr_data_quality_evidence(
        csv_root,
        labels,
        task_name="ehr_data_quality_task_test",
        condition="error",
        scenario="evidence-natural-missing",
    )
    with gzip.open(table, "rt", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert rows == []
    assert application.details["selected_record_count"] == 3


# Stream event degradation and retain an explicit status for selected test rows.
def test_ehr_event_evidence_adapter_marks_selected_test_history(tmp_path: Path) -> None:
    examples = tmp_path / "test_examples.csv"
    examples.write_text("patient_id,prediction_time\n7,2020-01-02\n", encoding="utf-8")
    events = tmp_path / "events.csv"
    fieldnames = ["patient_id", "start", "end", "code", "value", "unit", "visit_id"]
    with events.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(200):
            writer.writerow(
                {
                    "patient_id": "7",
                    "start": f"2019-01-{index % 28 + 1:02d}",
                    "end": "",
                    "code": f"LOINC/{index}",
                    "value": str(index),
                    "unit": "mg/dL",
                    "visit_id": str(index),
                }
            )

    original = events.read_bytes()
    control = apply_ehr_event_evidence(
        events,
        examples,
        task_name="ehr_event_modelling_test",
        condition="control",
        scenario="evidence-explicit-missing",
    )
    assert events.read_bytes() == original

    application = apply_ehr_event_evidence(
        events,
        examples,
        task_name="ehr_event_modelling_test",
        condition="error",
        scenario="evidence-explicit-missing",
    )
    with events.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    selected = [row for row in rows if row["_evidence_status"]]
    assert selected
    assert all(row["code"] == "" and row["value"] == "" for row in selected)
    assert control.details["selected_record_count"] == len(selected)
    assert application.details["selected_record_count"] == len(selected)


# Install the same pathology delivery shim in both matched conditions.
def test_pathology_evidence_adapter_installs_agent_runtime(tmp_path: Path) -> None:
    shim = REPOSITORY_ROOT / "faults/evidence/pathology_sitecustomize.py"
    application = prepare_pathology_evidence(
        tmp_path,
        task_name="tumor_area_selection_pathology_slide_test",
        condition="control",
        scenario="evidence-low-quality",
        shim_source=shim,
    )

    assert (tmp_path / "workspace/.healthagent_evidence/sitecustomize.py").is_file()
    assert (
        tmp_path / "workspace/.healthagent_evidence/bin/openslide-write-png"
    ).is_file()
    assert application.agent_environment["PYTHONPATH"] == (
        "/workspace/.healthagent_evidence"
    )
    assert application.agent_environment["PATH"].startswith(
        "/workspace/.healthagent_evidence/bin:"
    )
    assert application.details["selection_probability"] == pytest.approx(0.2)


# Exercise the exact-path sudo client, recovery capability, and external audit log.
def test_protected_file_sudo_broker_is_exact_path_allowlisted(
    tmp_path: Path,
) -> None:
    target = tmp_path / "protected.txt"
    target.write_bytes(b"release-4821\n")
    target.chmod(0o600)
    source = target.open("rb")
    target.chmod(0)
    audit_path = tmp_path / "audit.jsonl"
    container_target = "/workspace/protected.txt"
    broker = PrivilegeBroker(
        source=source,
        target=target,
        container_target=container_target,
        audit_path=audit_path,
        condition="error",
        readable_mode=0o600,
    )
    broker.start()
    try:
        runtime = write_privilege_runtime(
            tmp_path / "fault",
            host=broker.host,
            port=broker.port,
            token=broker.token,
        )
        sudo = runtime / "bin/sudo"
        listing = subprocess.run(
            [sudo, "-n", "-l"],
            text=True,
            capture_output=True,
            check=False,
        )
        recovered = subprocess.run(
            [sudo, "/bin/cat", container_target],
            capture_output=True,
            check=False,
        )
        denied = subprocess.run(
            [sudo, "cat", "/etc/shadow"],
            text=True,
            capture_output=True,
            check=False,
        )
        restored = subprocess.run(
            [sudo, "chmod", "u+r", container_target],
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        broker.stop()

    assert listing.returncode == 0
    assert container_target in listing.stdout
    assert recovered.stdout == b"release-4821\n"
    assert denied.returncode == 1
    assert "outside the experiment allowlist" in denied.stderr
    assert restored.returncode == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    events = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [(event["operation"], event["allowed"]) for event in events] == [
        ("list", True),
        ("cat", True),
        ("cat", False),
        ("chmod", True),
    ]


# Support the dedicated SIF, whose protected file lives only in writable-tmpfs.
def test_protected_file_sudo_can_restore_exact_bytes_client_side(
    tmp_path: Path,
) -> None:
    pristine = tmp_path / "pristine.dat"
    pristine.write_bytes(b"exact protected value\n")
    source = pristine.open("rb")
    target = tmp_path / "container-target.dat"
    target.write_bytes(b"unreadable placeholder")
    target.chmod(0)
    audit_path = tmp_path / "audit.jsonl"
    broker = PrivilegeBroker(
        source=source,
        target=None,
        container_target=str(target),
        audit_path=audit_path,
        condition="error",
        readable_mode=0o644,
    )
    broker.start()
    try:
        runtime = write_privilege_runtime(
            tmp_path / "fault",
            host=broker.host,
            port=broker.port,
            token=broker.token,
        )
        restored = subprocess.run(
            [runtime / "bin/sudo", "chmod", "u+r", str(target)],
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        broker.stop()

    assert restored.returncode == 0, restored.stderr
    assert target.read_bytes() == b"exact protected value\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_gzip_state_separates_metadata_from_content(tmp_path: Path) -> None:
    states = []
    for index, mtime in enumerate((1, 2)):
        path = tmp_path / f"input-{index}.csv.gz"
        with path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=mtime) as stream:
                stream.write(b"subject_id,value\n1,42\n")
        states.append(file_state(path))

    assert states[0]["sha256"] != states[1]["sha256"]
    assert states[0]["gzip_valid"] is True
    assert states[0]["gzip_content_sha256"] == states[1]["gzip_content_sha256"]


def test_latest_glob_selects_lexically_latest_file(tmp_path: Path) -> None:
    first = tmp_path / "study_01" / "report.txt"
    latest = tmp_path / "study_02" / "report.txt"
    first.parent.mkdir()
    latest.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    latest.write_text("latest", encoding="utf-8")

    selected = select_target(
        selector="latest-glob",
        target_expression=str(tmp_path / "study_*" / "report.txt"),
        allowed_root=tmp_path,
    )

    assert selected == latest


def test_selector_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("do not touch", encoding="utf-8")
    link = allowed / "link.txt"
    link.symlink_to(secret)

    with pytest.raises(InjectionError, match="escapes allowed root"):
        select_target(
            selector="exact",
            target_expression=str(link),
            allowed_root=allowed,
        )

    assert secret.read_text(encoding="utf-8") == "do not touch"


def test_family_fault_specs_cover_all_sif_families() -> None:
    tasks = sorted(
        path.parent
        for path in (REPOSITORY_ROOT / "tasks").glob("*/task.toml")
        if path.parent.name != "ehr_to_meds_etl"
    )
    assert len(tasks) == 53
    for task in tasks:
        spec = family_for_task(task.name)
        assert spec.primary_input.path.startswith("/")
        assert spec.dependency.name

    with pytest.raises(ValueError, match="matched none"):
        family_for_task("ehr_to_meds_etl")


@pytest.mark.parametrize(
    ("scenario", "agent_env_key"),
    (
        ("file-not-found", None),
        ("missing-dependency", "PYTHONPATH"),
        ("protected-file-sudo", "PATH"),
    ),
)
def test_sif_fault_runner_has_no_compose_arguments(
    tmp_path: Path, scenario: str, agent_env_key: str | None
) -> None:
    task_dir = REPOSITORY_ROOT / "tasks" / "ehr_data_quality_task_combined"
    command, agent_env = build_harbor_command(
        harbor="harbor",
        task_dir=task_dir,
        scenario=scenario,
        condition="error",
        singularity_cache_dir=tmp_path,
        harbor_arguments=["--agent", "codex", "--model", "gpt-5.6-luna"],
    )

    assert "--extra-docker-compose" not in command
    assert "faults.family_singularity:FamilySingularityEnvironment" in command
    assert "fault_condition=error" in command
    assert f"fault_scenario={scenario}" in command
    if agent_env_key is None:
        assert agent_env is None
    else:
        assert agent_env is not None
        assert agent_env.split("=", 1)[0] == agent_env_key


# Remove image attachment from Qwen requests without changing other Codex models.
def test_qwen_coder_uses_text_only_codex_adapter(tmp_path: Path) -> None:
    task_dir = REPOSITORY_ROOT / "tasks" / "ehr_data_quality_task_combined"
    command, _ = build_harbor_command(
        harbor="harbor",
        task_dir=task_dir,
        scenario="file-not-found",
        condition="control",
        singularity_cache_dir=tmp_path,
        harbor_arguments=["--agent", "codex", "--model", "qwen-coder"],
    )

    assert "--agent" not in command
    assert "--agent-import-path" in command
    assert "faults.codex_agent:TextOnlyCodex" in command


# Preserve the stock Harbor Codex adapter for models that accept image inputs.
def test_multimodal_codex_model_keeps_standard_adapter(tmp_path: Path) -> None:
    task_dir = REPOSITORY_ROOT / "tasks" / "ehr_data_quality_task_combined"
    command, _ = build_harbor_command(
        harbor="harbor",
        task_dir=task_dir,
        scenario="file-not-found",
        condition="control",
        singularity_cache_dir=tmp_path,
        harbor_arguments=["--agent", "codex", "--model", "gpt-5.6-luna"],
    )

    assert "--agent" in command
    assert "--agent-import-path" not in command


# Avoid the ignored Codex session flag because the adapter filters requests instead.
def test_text_only_codex_does_not_emit_ignored_view_image_flag(tmp_path: Path) -> None:
    agent = TextOnlyCodex(logs_dir=tmp_path, model_name="qwen-coder")

    assert "tools.view_image" not in agent.build_cli_flags()


# Remove image definitions, completed calls, and residual image blocks together.
def test_text_only_proxy_removes_image_capabilities() -> None:
    request = {
        "model": "qwen-coder",
        "tools": [
            {"type": "function", "name": "exec_command"},
            {"type": "function", "name": "view_image"},
        ],
        "input": [
            {
                "type": "function_call",
                "name": "view_image",
                "call_id": "image-call",
                "arguments": '{"path":"/workspace/slice.png"}',
            },
            {
                "type": "function_call_output",
                "call_id": "image-call",
                "output": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAAA",
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,BBBB",
                    }
                ],
            },
        ],
    }

    sanitized = json.loads(sanitize_request_body(json.dumps(request).encode()))

    assert [tool["name"] for tool in sanitized["tools"]] == ["exec_command"]
    assert all(item.get("call_id") != "image-call" for item in sanitized["input"])
    assert sanitized["input"][0]["content"] == [
        {
            "type": "input_text",
            "text": "[Image omitted by the text-only model adapter.]",
        }
    ]
    assert "data:image" not in json.dumps(sanitized)


def _run_import(runtime: Path, module_root: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(runtime), str(module_root)))
    return subprocess.run(
        [sys.executable, "-c", "import fault_probe_dependency"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_python_dependency_fault_is_agent_scoped(tmp_path: Path) -> None:
    module_root = tmp_path / "modules"
    module = module_root / "fault_probe_dependency"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    dependency = Dependency(kind="python-module", name="fault_probe_dependency")

    control_root = tmp_path / "control"
    control_root.mkdir()
    control_runtime, control_env = write_dependency_runtime(
        control_root, dependency, "control"
    )
    assert control_env == "PYTHONPATH=/opt/healthagent-fault"
    assert _run_import(control_runtime, module_root).returncode == 0

    error_root = tmp_path / "error"
    error_root.mkdir()
    error_runtime, error_env = write_dependency_runtime(error_root, dependency, "error")
    assert error_env == "PYTHONPATH=/opt/healthagent-fault"
    result = _run_import(error_runtime, module_root)
    assert result.returncode != 0
    assert "No module named 'fault_probe_dependency'" in result.stderr

    # Without the Agent-only PYTHONPATH, the dependency remains available to
    # Harbor's verifier process.
    clean_environment = os.environ.copy()
    clean_environment["PYTHONPATH"] = str(module_root)
    verifier_result = subprocess.run(
        [sys.executable, "-c", "import fault_probe_dependency"],
        env=clean_environment,
        capture_output=True,
        check=False,
    )
    assert verifier_result.returncode == 0


def test_executable_dependency_control_and_error(tmp_path: Path) -> None:
    real = tmp_path / "probe_tool"
    real.write_text("#!/bin/bash\nprintf 'available\\n'\n", encoding="utf-8")
    real.chmod(0o755)
    dependency = Dependency(
        kind="executable",
        name="probe_tool",
        real_path=PurePosixPath(str(real)),
    )

    control_root = tmp_path / "control"
    control_root.mkdir()
    control_runtime, _ = write_dependency_runtime(control_root, dependency, "control")
    control = subprocess.run(
        [str(control_runtime / "bin" / "probe_tool")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert control.returncode == 0
    assert control.stdout == "available\n"

    error_root = tmp_path / "error"
    error_root.mkdir()
    error_runtime, _ = write_dependency_runtime(error_root, dependency, "error")
    error = subprocess.run(
        [str(error_runtime / "bin" / "probe_tool")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert error.returncode == 127
    assert "probe_tool: command not found" in error.stderr


def _singularity_state(condition: str, scenario: str = SCENARIO) -> dict[str, object]:
    digest = "a" * 64
    before = {
        "exists": True,
        "size": 123,
        "sha256": digest,
        "declared_sha256": digest,
        "gzip_valid": True,
        "mode": "0644",
        "uid": 0,
    }
    if scenario == "file-not-found":
        after = (
            {
                "exists": True,
                "is_file": True,
                "size": 123,
                "sha256": digest,
                "gzip_valid": True,
            }
            if condition == "control"
            else {"exists": False, "is_file": False}
        )
    elif scenario == "checksum-mismatch":
        after = (
            {
                "exists": True,
                "is_file": True,
                "size": 123,
                "sha256": digest,
                "gzip_valid": True,
            }
            if condition == "control"
            else {
                "exists": True,
                "is_file": True,
                "size": len(SIF_CORRUPTION_MARKER),
                "sha256": "b" * 64,
                "gzip_valid": False,
            }
        )
    elif scenario == "recovery-http-403":
        after = {"exists": False, "is_file": False}
    elif scenario in PERMISSION_SCENARIOS:
        after = (
            {
                "exists": True,
                "is_file": True,
                "size": 123,
                "sha256": digest,
                "gzip_valid": True,
                "mode": "0644",
                "uid": 0,
            }
            if condition == "control"
            else {
                "exists": True,
                "is_file": True,
                "size": 123,
                "mode": "0000",
                "uid": 0,
            }
        )
    elif scenario in EVIDENCE_SCENARIOS:
        after = (
            {
                "exists": True,
                "is_file": True,
                "size": 123,
                "sha256": digest,
                "declared_sha256": digest,
                "gzip_valid": True,
            }
            if condition == "control"
            else {
                "exists": True,
                "is_file": True,
                "size": 120,
                "sha256": "b" * 64,
                "declared_sha256": "b" * 64,
                "gzip_valid": True,
            }
        )
    else:
        raise AssertionError(f"Unhandled test scenario: {scenario}")
    state = {
        "condition": condition,
        "scenario": scenario,
        "target": TARGET,
        "checksum_file": CHECKSUM_FILE,
        "before": before,
        "after": after,
    }
    if scenario in EVIDENCE_SCENARIOS:
        state["evidence"] = {
            "adapter": "ehr-to-meds-admission-record-v1",
            "selected_record_count": 1,
        }
    return state


@pytest.mark.parametrize("condition", ["control", "error"])
@pytest.mark.parametrize("scenario", SUPPORTED_SCENARIOS)
def test_singularity_fault_state_validation(condition: str, scenario: str) -> None:
    state = _singularity_state(condition, scenario)
    parsed = parse_fault_result("setup log\n" + json.dumps(state))

    validate_fault_result(  # type: ignore[arg-type]
        parsed, condition, scenario
    )
    command = build_fault_command(condition, scenario)  # type: ignore[arg-type]
    assert command.startswith("python -c ")
    assert condition in command
    assert scenario in command


def test_singularity_fault_state_rejects_wrong_after_state() -> None:
    state = _singularity_state("error")
    state["after"] = {"exists": True, "is_file": True}

    with pytest.raises(FaultSetupError, match="requested state"):
        validate_fault_result(state, "error")


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["control", "error"])
@pytest.mark.parametrize("scenario", ["file-not-found", "checksum-mismatch"])
async def test_singularity_environment_injects_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    scenario: str,
) -> None:
    calls: list[str] = []

    async def fake_start(self: SingularityEnvironment, force_build: bool) -> None:
        calls.append(f"start:{force_build}")

    async def fake_stop(self: SingularityEnvironment, delete: bool) -> None:
        calls.append(f"stop:{delete}")

    async def fake_exec(
        self: EhrToMedsFaultEnvironment,
        command: str,
        **kwargs: object,
    ) -> ExecResult:
        calls.append("inject")
        assert TARGET in command
        assert kwargs == {"cwd": "/workspace", "timeout_sec": 120}
        return ExecResult(
            stdout=json.dumps(_singularity_state(condition, scenario)),
            return_code=0,
        )

    monkeypatch.setattr(SingularityEnvironment, "start", fake_start)
    monkeypatch.setattr(SingularityEnvironment, "stop", fake_stop)
    monkeypatch.setattr(EhrToMedsFaultEnvironment, "exec", fake_exec)

    environment = object.__new__(EhrToMedsFaultEnvironment)
    environment._fault_condition = condition  # type: ignore[assignment]
    environment._fault_scenario = scenario  # type: ignore[assignment]
    environment._proxy_started = False
    environment._runtime_evidence_collected = False
    environment.trial_paths = TrialPaths(tmp_path)
    environment.logger = logging.getLogger("singularity-fault-test")

    await environment.start(force_build=False)

    assert calls == ["start:False", "inject"]
    manifest = json.loads((tmp_path / "fault-setup.json").read_text())
    assert manifest["condition"] == condition
    assert manifest["scenario"] == scenario
    assert manifest["injected_before_agent"] is True
    assert manifest["sif_modified"] is False


@pytest.mark.asyncio
async def test_singularity_environment_stops_when_injection_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fake_start(self: SingularityEnvironment, force_build: bool) -> None:
        calls.append("start")

    async def fake_stop(self: SingularityEnvironment, delete: bool) -> None:
        calls.append(f"stop:{delete}")

    async def fake_exec(
        self: EhrToMedsFaultEnvironment,
        command: str,
        **kwargs: object,
    ) -> ExecResult:
        calls.append("inject")
        return ExecResult(stderr="target missing", return_code=1)

    monkeypatch.setattr(SingularityEnvironment, "start", fake_start)
    monkeypatch.setattr(SingularityEnvironment, "stop", fake_stop)
    monkeypatch.setattr(EhrToMedsFaultEnvironment, "exec", fake_exec)

    environment = object.__new__(EhrToMedsFaultEnvironment)
    environment._fault_condition = "error"
    environment._fault_scenario = SCENARIO
    environment._proxy_started = False
    environment._runtime_evidence_collected = False
    environment.trial_paths = TrialPaths(tmp_path)
    environment.logger = logging.getLogger("singularity-fault-test")

    with pytest.raises(FaultSetupError, match="target missing"):
        await environment.start(force_build=False)

    assert calls == ["start", "inject", "stop:True"]
    assert not (tmp_path / "fault-setup.json").exists()


def _start_proxy(server: SelectiveProxyServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _proxy_authorization(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def _receive_until(stream: socket.socket, marker: bytes) -> bytes:
    received = bytearray()
    while marker not in received:
        chunk = stream.recv(4096)
        if not chunk:
            break
        received.extend(chunk)
    return bytes(received)


def test_selective_proxy_407_includes_basic_authentication_challenge(
    tmp_path: Path,
) -> None:
    server = create_proxy_server(
        condition="error",
        blocked_host="physionet.org",
        username="agent",
        password="test-secret",
        events_path=tmp_path / "events.jsonl",
    )
    thread = _start_proxy(server)
    try:
        with socket.create_connection(server.server_address, timeout=5) as stream:
            stream.sendall(
                b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com:443\r\n\r\n"
            )
            response = _receive_until(stream, b"\r\n\r\n")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.startswith(b"HTTP/1.1 407 Proxy Authentication Required")
    assert b'Proxy-Authenticate: Basic realm="healthagent-fault"\r\n' in response


def test_selective_proxy_returns_403_and_records_physionet_exposure(
    tmp_path: Path,
) -> None:
    username = "agent"
    password = "test-secret"
    events_path = tmp_path / "events.jsonl"
    server = create_proxy_server(
        condition="error",
        blocked_host="physionet.org",
        username=username,
        password=password,
        events_path=events_path,
    )
    thread = _start_proxy(server)
    try:
        with socket.create_connection(server.server_address, timeout=5) as stream:
            authorization = _proxy_authorization(username, password)
            stream.sendall(
                (
                    "CONNECT physionet.org:443 HTTP/1.1\r\n"
                    "Host: physionet.org:443\r\n"
                    f"Proxy-Authorization: {authorization}\r\n\r\n"
                ).encode()
            )
            response = _receive_until(stream, b"\r\n\r\n")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.startswith(b"HTTP/1.1 403 Forbidden")
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["host"] == "physionet.org"
    assert events[0]["action"] == "blocked"
    assert events[0]["status"] == 403


def test_selective_proxy_control_forwards_same_connect_request(
    tmp_path: Path,
) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    upstream_port = listener.getsockname()[1]
    upstream_received: list[bytes] = []

    def serve_upstream() -> None:
        connection, _ = listener.accept()
        with connection:
            upstream_received.append(connection.recv(4))
            connection.sendall(b"PONG")

    upstream_thread = threading.Thread(target=serve_upstream, daemon=True)
    upstream_thread.start()
    username = "agent"
    password = "test-secret"
    events_path = tmp_path / "events.jsonl"
    server = create_proxy_server(
        condition="control",
        blocked_host="127.0.0.1",
        username=username,
        password=password,
        events_path=events_path,
    )
    proxy_thread = _start_proxy(server)
    try:
        with socket.create_connection(server.server_address, timeout=5) as stream:
            authorization = _proxy_authorization(username, password)
            stream.sendall(
                (
                    f"CONNECT 127.0.0.1:{upstream_port} HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{upstream_port}\r\n"
                    f"Proxy-Authorization: {authorization}\r\n\r\n"
                ).encode()
            )
            response = _receive_until(stream, b"\r\n\r\n")
            assert response.startswith(b"HTTP/1.1 200 Connection Established")
            stream.sendall(b"PING")
            assert stream.recv(4) == b"PONG"
    finally:
        server.shutdown()
        server.server_close()
        listener.close()
        proxy_thread.join(timeout=5)
        upstream_thread.join(timeout=5)

    assert upstream_received == [b"PING"]
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["action"] == "forwarded"
    assert events[0]["status"] == 200


def test_git_client_can_authenticate_and_pass_through_proxy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("proxy test\n", encoding="utf-8")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(source), "add", "README.md"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=HealthAgent test",
            "-c",
            "user.email=healthagent@example.invalid",
            "commit",
            "-m",
            "Initial commit",
        ],
        check=True,
        capture_output=True,
    )
    served = tmp_path / "served"
    served.mkdir()
    bare = served / "repo.git"
    subprocess.run(
        ["git", "clone", "--bare", str(source), str(bare)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", f"--git-dir={bare}", "update-server-info"],
        check=True,
        capture_output=True,
    )

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(served),
    )
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    events_path = tmp_path / "events.jsonl"
    proxy = create_proxy_server(
        condition="control",
        blocked_host="127.0.0.1",
        username="agent",
        password="test-secret",
        events_path=events_path,
    )
    proxy_thread = _start_proxy(proxy)
    proxy_host, proxy_port = proxy.server_address
    upstream_host, upstream_port = upstream.server_address
    proxy_url = f"http://agent:test-secret@{proxy_host}:{proxy_port}"
    environment = os.environ.copy()
    environment.update({"NO_PROXY": "", "no_proxy": ""})
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"http.proxy={proxy_url}",
                "ls-remote",
                f"http://{upstream_host}:{upstream_port}/repo.git",
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()
        proxy_thread.join(timeout=5)
        upstream_thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    assert "HEAD" in result.stdout
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert any(event["action"] == "forwarded" for event in events)
