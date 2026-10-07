# Agent Meltdown trace labeling

This pipeline follows the three LLM stages described in Section 4.3 and
Appendix C of *Agent Meltdowns: The Road to Hell Is Paved with Helpful
Agents* (arXiv:2605.19149v1).

The scripts do not maintain hand-copied prompt variants. They download the
pinned arXiv v1 HTML once, extract the three Appendix C prompts and Appendix B
taxonomy verbatim, record SHA-256 hashes, and reuse the cached materials.
Set `PAPER_HTML=/path/to/2605.19149v1.html` on an offline compute node.

## Stages

1. `label_meltdowns_01_discover.py` cleans each Harbor trace and runs the
   behavior-discovery prompt with `gpt-5.4-mini`, high reasoning.
2. `label_meltdowns_02_aggregate.py` supplies the discovery directory as one
   deterministic inline snapshot and runs the aggregation prompt with
   `gpt-5.5`, medium reasoning. Its output is a **candidate** taxonomy.
3. `label_meltdowns_03_annotate.py` runs the behavior-annotation prompt with
   `gpt-5.4-mini`, high reasoning and emits one JSON file per trace.

The paper performed expert review between stages 2 and 3. No script can
faithfully replace that review. Therefore, final annotation uses the paper's
frozen Appendix B taxonomy by default. To use an expert-reviewed local
taxonomy, pass it directly to stage 3. Set `STOP_AFTER_AGGREGATION=1` to pause
at the review boundary. Setting `ANNOTATION_TAXONOMY=candidate` additionally
requires `CANDIDATE_TAXONOMY_REVIEWED=1`; this acknowledgement does not itself
make an unreviewed taxonomy equivalent to the paper protocol.

## Run

From `HealthAgentBench`:

```bash
LABELING_OPENAI_API_KEY="$OPENAI_API_KEY" \
  bash faults/telemetry/run_agent_meltdown_labeling.sh \
    jobs/gpt6-luna-rep1 \
    results/telemetry/meltdown_labels/gpt6-luna-rep1
```

The labeling endpoint is deliberately separated from `OPENAI_BASE_URL` used by
local Qwen/vLLM experiments. Set `LABELING_OPENAI_BASE_URL` only when the
endpoint actually serves the paper's GPT model IDs.

Run a one-trace smoke test before paying for the full corpus:

```bash
LIMIT=1 RUN_NAME=smoke \
LABELING_OPENAI_API_KEY="$OPENAI_API_KEY" \
  bash faults/telemetry/run_agent_meltdown_labeling.sh \
    jobs/gpt6-luna-rep1 \
    results/telemetry/meltdown_labels/smoke
```

Prepare cleaned inputs and verify trace discovery without API calls:

```bash
DRY_RUN=1 \
  bash faults/telemetry/run_agent_meltdown_labeling.sh \
    jobs/gpt6-luna-rep1 \
    results/telemetry/meltdown_labels/preflight
```

Use `OVERWRITE=1` to rerun completed requests. Existing per-trace results are
otherwise reused, making interrupted runs resumable.

## Outputs

The output directory contains:

- `cleaned/`: deterministic trace inputs used by both labeling passes;
- `discovery/`: Stage 1 candidate-behavior JSON;
- `taxonomy/candidate_taxonomy.md`: unreviewed Stage 2 aggregation;
- `taxonomy/paper_frozen_taxonomy.md`: Appendix B taxonomy used by default;
- `annotations/`: raw Stage 3 labels following Appendix C.3;
- `annotation_summary.csv`: flattened labels and a derived medium/high meltdown
  indicator;
- `metadata/`: prompt, trace, and taxonomy hashes, resolved model IDs, response
  IDs, token usage, and raw model outputs;
- `paper/`: pinned HTML, extracted prompt templates, taxonomy, and checksums.

The default output is below `results/`, which is Git-ignored because traces can
contain clinical data and model outputs can quote exact evidence.
