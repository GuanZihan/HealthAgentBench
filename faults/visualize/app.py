"""Streamlit application for browsing and summarizing completed rollouts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


HEALTHAGENTBENCH_ROOT = Path(__file__).resolve().parents[2]
if str(HEALTHAGENTBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(HEALTHAGENTBENCH_ROOT))

from faults.visualize.grouping import local_analysis  # noqa: E402
from faults.visualize.loader import (  # noqa: E402
    TrajectoryStep,
    TrialBundle,
    discover_trials,
    load_trial,
    redact_structure,
)
from faults.visualize.models import AnalysisDocument, BehaviorBlock  # noqa: E402
from faults.visualize.summarizer import (  # noqa: E402
    DEFAULT_MODEL,
    analysis_as_download,
    cache_path,
    generate_and_cache,
    load_cached_analysis,
)


st.set_page_config(
    page_title="Agent Rollout Explorer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=15, show_spinner=False)
def cached_trial_paths(jobs_root: str) -> list[str]:
    return [str(path) for path in discover_trials(Path(jobs_root))]


@st.cache_data(show_spinner=False)
def cached_load_trial(path: str, fingerprint_hint: int) -> TrialBundle:
    del fingerprint_hint
    return load_trial(Path(path))


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


def trial_label(path: Path) -> str:
    return f"{path.parent.name}  /  {path.name}"


def source_label(document: AnalysisDocument) -> str:
    if document.generator == "openai":
        return f"AI summary · {document.model}"
    return "Local fallback grouping"


def step_map(bundle: TrialBundle) -> dict[int, TrajectoryStep]:
    return {step.step_id: step for step in bundle.steps}


def render_step_content(step: TrajectoryStep) -> None:
    if step.timestamp:
        st.caption(step.timestamp)
    if step.message:
        st.markdown(step.message)
    for call in step.tool_calls:
        name = call.get("function_name") or call.get("name") or "tool"
        st.caption(f"Tool call · {name}")
        st.code(
            json.dumps(call.get("arguments", {}), ensure_ascii=False, indent=2),
            language="json",
        )
    if step.observation_text:
        st.caption("Observation")
        st.code(step.observation_text, language="text", wrap_lines=True)


def render_step(step: TrajectoryStep, expanded: bool = False) -> None:
    tool_suffix = f" · {', '.join(step.tool_names)}" if step.tool_names else ""
    status = " · failed" if step.failed else ""
    label = f"Step {step.step_id} · {step.source}{tool_suffix}{status}"
    with st.expander(label, expanded=expanded):
        render_step_content(step)


def render_block(
    block: BehaviorBlock,
    index: int,
    steps_by_id: dict[int, TrajectoryStep],
) -> None:
    with st.container(border=True):
        heading, range_column, phase_column = st.columns([5, 1, 1])
        with heading:
            st.subheader(f"{index}. {block.title}")
        with range_column:
            st.metric("Steps", f"{block.start_step}–{block.end_step}")
        with phase_column:
            st.metric("Phase", block.phase.value)
        st.caption(block.intent)
        st.write(block.summary)
        st.markdown(f"**Outcome:** {block.outcome}")
        if block.tools:
            st.caption(f"Tools · {', '.join(block.tools)}")
        evidence = [
            steps_by_id[step_id]
            for step_id in block.evidence_step_ids
            if step_id in steps_by_id
        ]
        if evidence:
            with st.expander(f"Evidence · {len(evidence)} selected steps"):
                for evidence_index, step in enumerate(evidence):
                    st.markdown(f"**Step {step.step_id} · {step.source}**")
                    render_step_content(step)
                    if evidence_index < len(evidence) - 1:
                        st.divider()


def render_block_chart(document: AnalysisDocument) -> None:
    rows = []
    for index, block in enumerate(document.analysis.blocks, start=1):
        rows.append(
            {
                "block": f"B{index:02d}",
                "steps": block.end_step - block.start_step + 1,
                "evidence": len(block.evidence_step_ids),
                "tools": len(block.tools),
            }
        )
    if rows:
        chart = pd.DataFrame(rows).set_index("block")
        st.bar_chart(chart, height=260)


def render_metrics(bundle: TrialBundle) -> None:
    columns = st.columns(6)
    values = (
        ("Condition", bundle.condition),
        ("Scenario", bundle.scenario),
        ("Reward", "—" if bundle.reward is None else f"{bundle.reward:.3f}"),
        ("Duration", format_duration(bundle.duration_seconds)),
        ("Agent cost", "—" if bundle.cost_usd is None else f"${bundle.cost_usd:.4f}"),
        ("Agent model", bundle.agent_model),
    )
    for column, (label, value) in zip(columns, values, strict=True):
        column.metric(label, value)


def sidebar() -> tuple[TrialBundle, str]:
    st.sidebar.title("Rollout source")
    default_jobs = os.environ.get(
        "HEALTHAGENTBENCH_JOBS_DIR",
        str(HEALTHAGENTBENCH_ROOT / "jobs"),
    )
    jobs_root = st.sidebar.text_input("Jobs directory", value=default_jobs)
    paths = [Path(path) for path in cached_trial_paths(jobs_root)]
    query = st.sidebar.text_input("Filter", placeholder="job, task, condition…").lower()
    if query:
        paths = [path for path in paths if query in trial_label(path).lower()]
    if not paths:
        st.sidebar.error("No trials with agent/trajectory.json were found.")
        st.stop()

    selected = st.sidebar.selectbox(
        "Trial",
        paths,
        format_func=trial_label,
    )
    mtime = (selected / "agent" / "trajectory.json").stat().st_mtime_ns
    bundle = cached_load_trial(str(selected), mtime)

    st.sidebar.divider()
    st.sidebar.title("AI analysis")
    model = st.sidebar.text_input("Model", value=DEFAULT_MODEL)
    reasoning = st.sidebar.selectbox(
        "Reasoning effort",
        ["low", "medium", "high"],
        index=1,
    )
    force = st.sidebar.checkbox("Regenerate existing cache", value=False)
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    if has_key:
        st.sidebar.success("OPENAI_API_KEY is available")
    else:
        st.sidebar.warning("Set OPENAI_API_KEY to enable GPT analysis.")
    generate = st.sidebar.button(
        "Generate AI summary",
        type="primary",
        use_container_width=True,
        disabled=not has_key,
    )
    if generate:
        with st.spinner(f"Analyzing with {model}…"):
            try:
                _, output_path = generate_and_cache(
                    bundle,
                    model=model,
                    reasoning_effort=reasoning,
                    force=force,
                )
            except Exception as error:
                st.sidebar.error(f"Analysis failed: {error}")
            else:
                st.sidebar.success(f"Saved {output_path.name}")
                st.rerun()
    return bundle, model


def main() -> None:
    bundle, model = sidebar()
    cached = load_cached_analysis(bundle, model)
    document = cached or local_analysis(bundle)

    st.title("Agent Rollout Explorer")
    st.caption(f"{bundle.job_name} / {bundle.trial_name}")
    st.header(document.analysis.headline)
    render_metrics(bundle)

    summary, strategy, outcome = st.columns([2, 2, 2])
    with summary:
        st.subheader("Overview")
        st.write(document.analysis.overview)
    with strategy:
        st.subheader("Strategy")
        st.write(document.analysis.strategy)
    with outcome:
        st.subheader("Final outcome")
        st.write(document.analysis.final_outcome)

    cache_note = cache_path(bundle, model)
    st.caption(
        f"{source_label(document)} · {len(document.analysis.blocks)} blocks · "
        f"cache: {cache_note.relative_to(bundle.path)}"
    )
    st.download_button(
        "Download analysis JSON",
        analysis_as_download(document),
        file_name=f"{bundle.trial_name}-analysis.json",
        mime="application/json",
    )

    blocks_tab, steps_tab, data_tab = st.tabs(
        ["Behavior blocks", "Step explorer", "Run data"]
    )
    with blocks_tab:
        st.subheader("Rollout shape")
        st.caption("Block size, selected evidence, and tool diversity")
        render_block_chart(document)
        by_id = step_map(bundle)
        for index, block in enumerate(document.analysis.blocks, start=1):
            render_block(block, index, by_id)

    with steps_tab:
        sources = sorted({step.source for step in bundle.steps})
        selected_sources = st.multiselect(
            "Sources",
            sources,
            default=[source for source in sources if source == "agent"],
        )
        search = st.text_input(
            "Search steps", placeholder="command, filename, observation…"
        )
        filtered = []
        for step in bundle.steps:
            if selected_sources and step.source not in selected_sources:
                continue
            haystack = f"{step.message}\n{step.observation_text}".lower()
            if search and search.lower() not in haystack:
                continue
            filtered.append(step)
        st.caption(f"Showing {len(filtered)} of {len(bundle.steps)} steps")
        for step in filtered:
            render_step(step)

    with data_tab:
        with st.expander("Task instruction"):
            st.markdown(
                bundle.instruction or "No task instruction found in the trajectory."
            )
        for label, key in (
            ("Verifier metrics", "metrics"),
            ("Fault setup", "fault_setup"),
            ("Fault final state", "fault_final_state"),
            ("Family runtime", "family_runtime"),
            ("Trial result", "result"),
            ("Trial config", "config"),
        ):
            with st.expander(label):
                st.json(redact_structure(bundle.artifacts.get(key, {})))


if __name__ == "__main__":
    main()
