"""Deterministic fallback grouping when no model analysis is available."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from faults.visualize.loader import TrajectoryStep, TrialBundle
from faults.visualize.models import (
    AnalysisDocument,
    BehaviorBlock,
    BehaviorPhase,
    RolloutNarrative,
)


PHASE_TERMS: tuple[tuple[BehaviorPhase, tuple[str, ...]], ...] = (
    (BehaviorPhase.finalization, ("completed", "final", "saved", "deliver")),
    (BehaviorPhase.validation, ("validate", "verify", "test", "check", "missing ids")),
    (
        BehaviorPhase.implementation,
        ("write", "create", "apply_patch", "submission", "mkdir"),
    ),
    (
        BehaviorPhase.inspection,
        ("inspect", "list", "schema", "profile", "ls ", "find "),
    ),
    (BehaviorPhase.analysis, ("analy", "compare", "groupby", "candidate", "conflict")),
)


def infer_phase(steps: list[TrajectoryStep]) -> BehaviorPhase:
    fragments: list[str] = []
    for step in steps:
        fragments.extend((step.message, step.observation_text, *step.tool_names))
    text = "\n".join(fragments).lower()
    for phase, terms in PHASE_TERMS:
        if any(term in text for term in terms):
            return phase
    return BehaviorPhase.orientation if steps[0].step_id <= 5 else BehaviorPhase.other


def _title(phase: BehaviorPhase, index: int) -> str:
    titles = {
        BehaviorPhase.orientation: "Establish approach",
        BehaviorPhase.inspection: "Inspect task and data",
        BehaviorPhase.analysis: "Analyze evidence",
        BehaviorPhase.implementation: "Build the deliverable",
        BehaviorPhase.validation: "Validate the result",
        BehaviorPhase.finalization: "Report completion",
        BehaviorPhase.other: "Continue task work",
    }
    return f"{titles[phase]} · {index}"


def _group_agent_steps(steps: tuple[TrajectoryStep, ...]) -> list[list[TrajectoryStep]]:
    groups: list[list[TrajectoryStep]] = []
    current: list[TrajectoryStep] = []
    for step in steps:
        narrative_boundary = not step.tool_calls and len(current) >= 2
        if narrative_boundary or len(current) >= 5:
            groups.append(current)
            current = []
        current.append(step)
    if current:
        groups.append(current)
    return groups


def local_analysis(bundle: TrialBundle) -> AnalysisDocument:
    """Build a neutral, coarse analysis without making an API call."""

    blocks: list[BehaviorBlock] = []
    groups = _group_agent_steps(bundle.agent_steps)
    for index, steps in enumerate(groups, start=1):
        phase = infer_phase(steps)
        tools = list(dict.fromkeys(tool for step in steps for tool in step.tool_names))
        narratives = [
            step.message.strip()
            for step in steps
            if step.message.strip() and not step.message.startswith("Executed ")
        ]
        summary = " ".join(narratives)
        if not summary:
            summary = (
                f"Ran {len(tools) or 1} tool type(s) across {len(steps)} agent steps."
            )
        failures = sum(step.failed for step in steps)
        outcome = (
            f"The block contains {failures} tool execution failure(s)."
            if failures
            else "The recorded actions completed without a detected tool failure."
        )
        blocks.append(
            BehaviorBlock(
                block_id=f"B{index:02d}",
                title=_title(phase, index),
                phase=phase,
                start_step=steps[0].step_id,
                end_step=steps[-1].step_id,
                intent=narratives[0]
                if narratives
                else "Continue the current task workflow.",
                summary=summary,
                outcome=outcome,
                tools=tools,
                evidence_step_ids=[step.step_id for step in steps[:3]],
            )
        )

    all_tools = Counter(tool for step in bundle.agent_steps for tool in step.tool_names)
    final_message = (
        bundle.agent_steps[-1].message
        if bundle.agent_steps
        else "No agent steps found."
    )
    overview = (
        f"The rollout contains {len(bundle.agent_steps)} agent steps and "
        f"{sum(all_tools.values())} recorded tool calls."
    )
    return AnalysisDocument(
        generated_at=datetime.now(timezone.utc),
        generator="local",
        model=None,
        source_fingerprint=bundle.fingerprint,
        analysis=RolloutNarrative(
            headline=f"{bundle.task_name}: rollout overview",
            overview=overview,
            strategy="This is deterministic fallback grouping. Generate the AI analysis for a semantic summary.",
            final_outcome=final_message
            or "The trajectory ended without a final agent message.",
            blocks=blocks,
        ),
    )
