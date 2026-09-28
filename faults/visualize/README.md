# Agent Rollout Explorer

This Streamlit app turns completed Harbor/Codex trajectories into a compact,
chronological view of agent behavior. Consecutive steps can be consolidated by
`openai/gpt-5.6-luna` into semantic behavior blocks. This first version is
deliberately neutral: it summarizes actions and outcomes without classifying
meltdown errors.

## Start the web app

From `HealthAgentBench/`:

```bash
export OPENAI_API_KEY=...
bash faults/visualize/run.sh
```

On Rivanna, bind Streamlit to the compute node and forward the port through SSH:

```bash
bash faults/visualize/run.sh \
  --server.address 0.0.0.0 \
  --server.port 8501
```

The jobs directory defaults to `HealthAgentBench/jobs`. Set
`HEALTHAGENTBENCH_JOBS_DIR` or change it in the sidebar to browse another
artifact root. `OPENAI_BASE_URL` is honored when an OpenAI-compatible endpoint
is required.

The page is useful without an API key: it falls back to deterministic coarse
grouping. An API call is made only when **Generate AI summary** is clicked.

## Precompute after a rollout

To analyze a trial before opening the page:

```bash
uv run python -m faults.visualize \
  jobs/<job-name>/<trial-name>
```

The default analysis model is `openai/gpt-5.6-luna`. A successful analysis is
stored at:

```text
<trial>/visualization/analysis-openai-gpt-5.6-luna.json
```

The cache records a SHA-256 fingerprint of the trajectory, config, result,
metrics, and fault-state artifacts. It is ignored automatically if those
inputs change. Use `--force` to regenerate an otherwise valid cache.

## Page structure

- **Behavior blocks** shows the model-generated overview, rollout-shape chart,
  and expandable evidence for each block.
- **Step explorer** retains direct access to raw messages, tool calls, and
  observations with source filters and full-text search.
- **Run data** shows verifier metrics, fault state, runtime provenance, result,
  and redacted configuration.

No HTML is embedded in the Python application. The interface uses native
Streamlit layout, chart, container, metric, expander, and JSON components.
