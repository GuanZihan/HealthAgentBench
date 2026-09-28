from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from faults.visualize.grouping import local_analysis
from faults.visualize.loader import discover_trials, load_trial, model_input
from faults.visualize.models import BehaviorBlock, BehaviorPhase, RolloutNarrative
from faults.visualize.summarizer import (
    analyze_with_openai,
    load_cached_analysis,
    save_analysis,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_trial(tmp_path: Path) -> Path:
    trial = tmp_path / "jobs" / "example-control" / "task__abc"
    write_json(
        trial / "agent" / "trajectory.json",
        {
            "steps": [
                {"step_id": 1, "source": "system", "message": "API_KEY=do-not-send"},
                {"step_id": 2, "source": "user", "message": "Inspect the dataset."},
                {
                    "step_id": 3,
                    "source": "agent",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": "I will inspect the files.",
                },
                {
                    "step_id": 4,
                    "source": "agent",
                    "message": "Executed list call",
                    "tool_calls": [
                        {
                            "function_name": "exec",
                            "arguments": {
                                "input": "OPENAI_API_KEY=do-not-send ls /workspace/data"
                            },
                        }
                    ],
                    "observation": {"results": [{"content": "data.csv"}]},
                },
                {
                    "step_id": 5,
                    "source": "agent",
                    "message": "Completed the review.",
                },
            ]
        },
    )
    write_json(
        trial / "config.json",
        {
            "agent": {"model_name": "gpt-5.6-luna"},
            "environment": {
                "kwargs": {
                    "fault_condition": "control",
                    "fault_scenario": "file-not-found",
                }
            },
        },
    )
    write_json(
        trial / "result.json",
        {
            "task_name": "example_task",
            "trial_name": "task__abc",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:01:00Z",
            "agent_result": {
                "cost_usd": 0.01,
                "n_input_tokens": 100,
                "n_output_tokens": 20,
            },
            "verifier_result": {"rewards": {"reward": 0.5}},
        },
    )
    write_json(trial / "verifier" / "metrics.json", {"reward": 0.5, "f1": 0.7})
    return trial


def sample_narrative() -> RolloutNarrative:
    return RolloutNarrative(
        headline="Agent inspected and completed the task",
        overview="The agent inspected one input file.",
        strategy="It listed the data before reporting completion.",
        final_outcome="The run ended with a completion message.",
        blocks=[
            BehaviorBlock(
                block_id="B01",
                title="Inspect input",
                phase=BehaviorPhase.inspection,
                start_step=3,
                end_step=5,
                intent="Inspect the files.",
                summary="Listed the data and concluded the review.",
                outcome="The agent reported completion.",
                tools=["exec"],
                evidence_step_ids=[3, 4, 5],
            )
        ],
    )


def test_discovery_loading_and_local_grouping(tmp_path: Path) -> None:
    trial = make_trial(tmp_path)
    assert discover_trials(tmp_path / "jobs") == [trial]

    bundle = load_trial(trial)
    assert bundle.task_name == "example_task"
    assert bundle.duration_seconds == 60
    assert bundle.reward == 0.5
    assert bundle.instruction == "Inspect the dataset."

    document = local_analysis(bundle)
    assert document.generator == "local"
    assert document.analysis.blocks
    assert document.analysis.blocks[0].start_step == 3


def test_model_input_is_bounded_and_omits_system_prompt(tmp_path: Path) -> None:
    bundle = load_trial(make_trial(tmp_path))
    payload = model_input(bundle, max_chars=5_000)
    assert len(payload) <= 5_000
    assert "do-not-send" not in payload
    assert "Inspect the dataset" in payload
    assert "ls /workspace/data" in payload


def test_structured_analysis_and_cache(tmp_path: Path) -> None:
    bundle = load_trial(make_trial(tmp_path))

    class FakeResponses:
        def parse(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["model"] == "gpt-5.6-luna"
            assert kwargs["text_format"] is RolloutNarrative
            return SimpleNamespace(
                id="resp_test",
                output_parsed=sample_narrative(),
                usage=SimpleNamespace(
                    input_tokens=50, output_tokens=25, total_tokens=75
                ),
            )

    client = SimpleNamespace(responses=FakeResponses())
    document = analyze_with_openai(bundle, client=client)
    assert document.generator == "openai"
    assert document.response_id == "resp_test"
    assert document.usage["total_tokens"] == 75

    target = bundle.path / "visualization" / "analysis-openai-gpt-5.6-luna.json"
    save_analysis(document, target)
    cached = load_cached_analysis(bundle)
    assert cached is not None
    assert cached.analysis.blocks[0].title == "Inspect input"
