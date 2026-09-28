# HealthAgentBench on Rivanna (Apptainer)

The benchmark uses seven family-level SIF images. The image contains only the
shared operating-system and Python dependencies. Task-specific workspace files
and clinical inputs are staged for each trial by
`faults.family_singularity:FamilySingularityEnvironment`.

| Family | SIF |
| --- | --- |
| Clinical trial matching | `healthagentbench-clinical-trial-matching.sif` |
| CT abnormality | `healthagentbench-ct-abnormality.sif` |
| EHR data quality | `healthagentbench-ehr-data-quality.sif` |
| EHR event modelling | `healthagentbench-ehr-event-modelling.sif` |
| EHR to MEDS | `healthagentbench-ehr-to-meds.sif` |
| Pathology tumor selection | `healthagentbench-pathology.sif` |
| X-ray report correction | `healthagentbench-xray-report.sif` |

## Build

On Rivanna, build the six shared family images with:

```bash
cd HealthAgentBench
bash containers/build_family_sifs.sh all
```

The script attempts to load `apptainer/1.5.0` when Apptainer is not already on
`PATH`. Existing SIF files are preserved. Rebuild one family explicitly with:

```bash
bash containers/build_family_sifs.sh --force ehr-event-modelling
```

## Configure gated data

Copy `.env.example` to `.env` and fill only the credentials needed by the task
family:

- `HF_TOKEN` for CT abnormality.
- `REDIVIS_API_TOKEN` for EHR event modelling.
- `PN_USER` and `PN_PASS` for X-ray report correction.
- `CHEXPROMPT_*` verifier credentials for X-ray report correction.

The runtime reads dataset credentials from `.env` and sends them only to the
short-lived bootstrap process. Do not source `.env` into the parent shell; use
the launcher below so dataset credentials are not inherited by the Agent
container.

## Run one task

```bash
bash containers/run_family_task.sh clinical_trial_matching_task_19
```

The launcher defaults to `gpt-5.6-luna` with medium reasoning. Override these
without editing the script:

```bash
MODEL=gpt-5.6-luna REASONING_EFFORT=medium \
  JOB_NAME=ct-control \
  bash containers/run_family_task.sh ct_abnormality_valid_137_a_1
```

Additional arguments are forwarded to `harbor run` after the launcher's fixed
arguments.

## Run a matched fault pair

The same family SIF is used for both conditions; only the private per-trial
runtime state changes. For example:

```bash
bash faults/scripts/run_family_control.sh \
  ehr_data_quality_task_combined file-not-found
bash faults/scripts/run_family_error.sh \
  ehr_data_quality_task_combined file-not-found
```

The generic family runner supports `file-not-found`, `corrupt-file`,
`truncated-file`, and `missing-dependency`. See `faults/README.md` for evidence
files, model overrides, and the dedicated EHR-to-MEDS scenarios.

## Runtime isolation

Each trial follows this sequence:

1. Create a private temporary runtime directory.
2. Copy only the task's public workspace skeleton into it.
3. Run the task's original `bootstrap.sh` in a short-lived Apptainer process.
4. Mount only the staged agent-visible workspace/data into Harbor's long-lived
   Agent container.
5. Keep bootstrap scripts, cross-task caches, `/tests`, gold material, and
   dataset download credentials out of Agent-visible mounts.
6. Delete the per-trial runtime directory after verification.

The trial directory records `family-runtime.json`, including the selected
family, SIF SHA-256, visible mount targets, and names (never values) of supplied
credential variables.
