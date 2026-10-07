# Cross-family evidence degradation

The same three experiment labels are available for every HealthAgentBench task
family. Their concrete transform is modality-specific so a valid CT, slide,
image, table, or note remains usable rather than becoming another corrupt-file
condition.

The three runtime renderings are:

- `evidence-explicit-missing`: replace evidence with visible placeholders such
  as `[BIOMARKER REDACTED]` and `[TREATMENT REDACTED]`.
- `evidence-natural-missing`: remove the same evidence spans and clean only the
  whitespace left by deletion.
- `evidence-low-quality`: replace the same spans with frozen, context-fitting
  uncertainty statements that do not invent a new clinical value.

## Modality contract

| Family | Selection unit | Explicit missing | Natural missing | Low quality |
| --- | --- | --- | --- | --- |
| Clinical trial matching | Human-approved GPT-5.5 text spans | Bracketed category redaction | Delete enclosing phrase | Context-fitting frozen uncertainty text |
| CT abnormality | Central 20% axial slab | NaN voxels | Silent background suppression | 4x4x2 pixelation plus 100-HU quantization |
| X-ray correction | Central thoracic ROI in every target-study view | Labelled redaction region | Unlabelled median-intensity suppression | Blur plus 8x pixelation |
| Pathology | Task-seeded 20% of 4096x4096 analysis tiles | Magenta crosshatch | White background suppression | 8x pixelation |
| EHR data quality | Three deterministic verifier-relevant clusters | Blank field plus missing status | Delete selected records | Retain value plus unverified-source status |
| EHR event modelling | Deterministic 10% of test-patient history | Blank code/value/unit plus status | Delete selected records | Round value, drop unit, mark source unverified |
| EHR-to-MEDS | One deterministic admission record | Redact four demographic fields | Delete the record | Replace fields with `UNKNOWN` and mark source unverified |

Every control validates the same selection policy but leaves source bytes
unchanged. Every error records the adapter version, selection policy, counts,
and non-sensitive hashes in `fault-setup.json`.

Pathology uses a delivery-layer transform because rewriting each 2-3 GB tiled
WSI for every trial is prohibitively expensive. It covers Python OpenSlide and
the standard `openslide-write-png` command and records actual OpenSlide
exposure. Direct raw-TIFF access is not transformed and must be reported as a
bypass during trajectory analysis.

## Clinical text recipe preparation

The clinical layer keeps the preparation model out of the measured rollout.
GPT-5.5 selects a small number of verbatim evidence spans and contextual
low-quality replacements once, the script freezes them with source hashes, and
every control/error pair later uses the same approved recipe.

Clinical recipes use a direct-fact contract. All direct textual mentions of a
selected fact are transformed together; edit IDs with the same stem, such as
`e01a` and `e01b`, identify repeated mentions. Indirect evidence is deliberately
left intact. For example, removing the words `type 1 diabetes` does not also
remove insulin use or hyperglycaemia. This keeps the perturbation localized and
lets the experiment measure whether the agent can recover from corroborating
evidence rather than turning the note into a different patient case.

### 1. Generate drafts with GPT-5.5

Set an OpenAI API key without writing it into the repository. From
`HealthAgentBench/`, regenerate all nine clinical-task drafts with the pinned
GPT-5.5 snapshot:

```bash
unset OPENAI_API_KEY
read -r -s -p 'OpenAI API key: ' OPENAI_API_KEY; export OPENAI_API_KEY; printf '\n'

EVIDENCE_MODEL=gpt-5.5-2026-04-23 \
EVIDENCE_API_MODE=responses \
  .venv/bin/python -m faults.evidence.prepare_clinical_recipes \
  --overwrite-drafts
```

Use `--task clinical_trial_matching_task_19` to prepare only one task. Recipes
are written under `faults/evidence/recipes/` with `status: draft`. The raw model
response and patient note are kept under the git-ignored
`faults/evidence/generated/` for local review. Invalid span selections receive
up to two deterministic correction attempts by default; every attempt and its
validation result remains in the local audit record. Existing drafts are kept
on later runs; use `--overwrite-drafts` only when intentionally regenerating
them. If a final response contains both valid and invalid selections, the
generator keeps the valid non-overlapping subset and records every rejection.
The OpenAI request uses Structured Outputs, medium reasoning, and `store=false`.

Manually revised drafts carry `curation.locked=true`. The generator refuses to
replace those files even when `--overwrite-drafts` is supplied. To regenerate a
curated task intentionally, first make a reviewed backup and explicitly remove
the lock from that one recipe; do not unlock the full recipe directory.

Check all new schema-v2 drafts without approving them:

```bash
.venv/bin/python -m faults.evidence.validate_clinical_recipes --allow-drafts
```

### 2. Review and approve each task

Preview the exact selected spans and all three degraded notes without writing:

```bash
.venv/bin/python -m faults.evidence.approve_clinical_recipe \
  --task clinical_trial_matching_task_19
```

After reviewing the output, repeat with an explicit reviewer identifier:

```bash
.venv/bin/python -m faults.evidence.approve_clinical_recipe \
  --task clinical_trial_matching_task_19 \
  --reviewer bxv6gs \
  --confirm-reviewed
```

The approval command rechecks the source hash, every span hash, the task/topic
mapping, all three transformations, and the ignored generation record. Runtime
execution refuses drafts or recipes whose source has changed.

For each preview, review four semantic properties that hashes cannot establish:

1. every direct repeat or synonymous restatement of the selected fact is edited;
2. explicit placeholders preserve the surrounding sentence grammar;
3. natural deletion leaves fluent prose without deleting unrelated evidence;
4. low-quality replacement preserves grammar but does not reveal the selected
   value.

Do not approve a recipe merely because `--allow-drafts` passes. That command
proves structural integrity and deterministic rendering, not clinical or
linguistic validity.

After all nine reviews are approved, run the strict experiment preflight:

```bash
.venv/bin/python -m faults.evidence.validate_clinical_recipes
```

## Run matched experiments

Non-clinical families need no preparation recipe. Clinical tasks require an
approved recipe before either condition can start. Run one family with:

```bash
SCENARIOS="evidence-explicit-missing evidence-natural-missing evidence-low-quality" \
JOBS_DIR="$PWD/jobs/clinical-evidence-rep1" \
MODEL=qwen-coder REPLICATE=rep1 CAPTURE_TELEMETRY=1 \
VLLM_UPSTREAM_URL=http://127.0.0.1:8010 \
  bash run_clinical_trial_experiments.sh
```

The control input is byte-identical to the staged source. The error input is a
deterministic transform, and `fault-setup.json` records the recipe hash,
severity, edit count, and edited categories without copying source text.

Run only evidence conditions for every family with the same `SCENARIOS` value:

```bash
export SCENARIOS="evidence-explicit-missing evidence-natural-missing evidence-low-quality"
export MODEL=qwen-coder CAPTURE_TELEMETRY=1
export VLLM_UPSTREAM_URL=http://127.0.0.1:8010

bash run_ct_experiments.sh
bash run_xray_experiments.sh
bash run_pathology_experiments.sh
bash run_clinical_trial_experiments.sh
bash run_ehr_data_quality_experiments.sh
bash run_ehr_event_modelling_experiments.sh
bash run_ehr_to_meds_experiments.sh
```
