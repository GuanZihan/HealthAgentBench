"""Discover and normalize Harbor/Codex rollout artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret)",
    re.IGNORECASE,
)
INLINE_SECRET = re.compile(
    r"(?i)\b([a-z0-9_-]*(?:api[_-]?key|access[_-]?token|authorization|password|secret))\b"
    r"([\s\"'=:\\]+)([^\s,;\"'}]+)"
)


@dataclass(frozen=True)
class TrajectoryStep:
    """Normalized representation of an ATIF trajectory step."""

    step_id: int
    source: str
    timestamp: str | None
    message: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    observation: Any = None

    @property
    def tool_names(self) -> list[str]:
        names: list[str] = []
        for call in self.tool_calls:
            name = str(call.get("function_name") or call.get("name") or "tool")
            if name not in names:
                names.append(name)
        return names

    @property
    def observation_text(self) -> str:
        return flatten_text(self.observation)

    @property
    def failed(self) -> bool:
        text = f"{self.message}\n{self.observation_text}".lower()
        return any(
            marker in text for marker in ("script failed", "traceback", "error:")
        )


@dataclass(frozen=True)
class TrialBundle:
    """All visualization-relevant artifacts for one trial."""

    path: Path
    job_name: str
    trial_name: str
    task_name: str
    condition: str
    scenario: str
    agent_model: str
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None
    reward: float | None
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    steps: tuple[TrajectoryStep, ...]
    fingerprint: str
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_steps(self) -> tuple[TrajectoryStep, ...]:
        return tuple(step for step in self.steps if step.source == "agent")

    @property
    def instruction(self) -> str:
        for step in self.steps:
            if step.source != "user":
                continue
            if "<environment_context>" in step.message:
                continue
            return step.message
        return ""


def read_json(path: Path) -> Any:
    """Read a JSON artifact, returning an empty dictionary when absent."""

    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_text(value: Any) -> str:
    """Extract readable text from nested ATIF observation payloads."""

    parts: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            parts.append(item)
            return
        if isinstance(item, dict):
            if isinstance(item.get("text"), str):
                parts.append(item["text"])
                return
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        parts.append(str(item))

    visit(value)
    return "\n".join(part for part in parts if part)


def redact_text(text: str) -> str:
    """Best-effort redaction before trajectory text reaches a model or UI."""

    return INLINE_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text
    )


def redact_structure(value: Any) -> Any:
    """Recursively redact values under credential-like keys."""

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if SENSITIVE_KEY.search(str(key)) and not str(key).endswith("_tokens"):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = redact_structure(item)
        return redacted
    if isinstance(value, list):
        return [redact_structure(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def discover_trials(jobs_root: Path) -> list[Path]:
    """Return trial directories containing a Codex ATIF trajectory."""

    jobs_root = jobs_root.expanduser().resolve()
    if not jobs_root.is_dir():
        return []
    trials = {path.parent.parent for path in jobs_root.rglob("agent/trajectory.json")}
    return sorted(
        trials,
        key=lambda path: (path.stat().st_mtime, path.parent.name, path.name),
        reverse=True,
    )


def resolve_trial(path: Path) -> Path:
    """Resolve either a trial directory or a single-trial job directory."""

    path = path.expanduser().resolve()
    if (path / "agent" / "trajectory.json").is_file():
        return path
    trials = discover_trials(path)
    if not trials:
        raise FileNotFoundError(f"No agent/trajectory.json found under {path}")
    if len(trials) > 1:
        raise ValueError(f"Expected one trial under {path}, found {len(trials)}")
    return trials[0]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _nested(mapping: Any, *keys: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_trial(path: Path) -> TrialBundle:
    """Load a trial and normalize optional metrics/fault artifacts."""

    trial = resolve_trial(path)
    trajectory_path = trial / "agent" / "trajectory.json"
    trajectory = read_json(trajectory_path)
    config = read_json(trial / "config.json")
    result = read_json(trial / "result.json")
    metrics = read_json(trial / "verifier" / "metrics.json")
    fault_setup = read_json(trial / "fault-setup.json")
    fault_final = read_json(trial / "fault-final-state.json")
    family_runtime = read_json(trial / "family-runtime.json")

    steps: list[TrajectoryStep] = []
    for index, raw in enumerate(trajectory.get("steps", []), start=1):
        if not isinstance(raw, dict):
            continue
        step_id = raw.get("step_id", index)
        try:
            normalized_id = int(step_id)
        except (TypeError, ValueError):
            normalized_id = index
        raw_calls = raw.get("tool_calls") or []
        calls = tuple(
            redact_structure(call) for call in raw_calls if isinstance(call, dict)
        )
        steps.append(
            TrajectoryStep(
                step_id=normalized_id,
                source=str(raw.get("source", "unknown")),
                timestamp=raw.get("timestamp"),
                message=redact_text(str(raw.get("message", ""))),
                tool_calls=calls,
                observation=redact_structure(raw.get("observation")),
            )
        )

    started_at = result.get("started_at")
    finished_at = result.get("finished_at")
    start_dt = _parse_timestamp(started_at)
    finish_dt = _parse_timestamp(finished_at)
    duration = (
        (finish_dt - start_dt).total_seconds() if start_dt and finish_dt else None
    )

    task_name = str(
        result.get("task_name")
        or fault_setup.get("task")
        or Path(_nested(config, "task", "path") or "unknown").name
    )
    condition = str(
        fault_setup.get("condition")
        or _nested(config, "environment", "kwargs", "fault_condition")
        or "unknown"
    )
    scenario = str(
        fault_setup.get("scenario")
        or _nested(config, "environment", "kwargs", "fault_scenario")
        or "none"
    )
    agent_model = str(
        _nested(result, "agent_info", "model_info", "name")
        or _nested(config, "agent", "model_name")
        or "unknown"
    )
    reward = _nested(result, "verifier_result", "rewards", "reward")
    if reward is None:
        reward = metrics.get("reward")

    fingerprint_paths = [
        trajectory_path,
        trial / "config.json",
        trial / "result.json",
        trial / "verifier" / "metrics.json",
        trial / "fault-setup.json",
        trial / "fault-final-state.json",
    ]
    return TrialBundle(
        path=trial,
        job_name=trial.parent.name,
        trial_name=str(result.get("trial_name") or trial.name),
        task_name=task_name,
        condition=condition,
        scenario=scenario,
        agent_model=agent_model,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        reward=float(reward) if reward is not None else None,
        cost_usd=_as_float(_nested(result, "agent_result", "cost_usd")),
        input_tokens=_as_int(_nested(result, "agent_result", "n_input_tokens")),
        output_tokens=_as_int(_nested(result, "agent_result", "n_output_tokens")),
        steps=tuple(steps),
        fingerprint=_fingerprint(fingerprint_paths),
        artifacts={
            "config": config,
            "result": result,
            "metrics": metrics,
            "fault_setup": fault_setup,
            "fault_final_state": fault_final,
            "family_runtime": family_runtime,
        },
    )


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def clip_text(text: str, limit: int) -> str:
    """Preserve the beginning and end of long tool output."""

    if len(text) <= limit:
        return text
    head = max(limit * 2 // 3, 1)
    tail = max(limit - head - 40, 1)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}"


def model_input(bundle: TrialBundle, max_chars: int = 180_000) -> str:
    """Create a bounded, redacted trajectory representation for the LLM."""

    included = [
        step
        for step in bundle.steps
        if step.source != "system" and "<environment_context>" not in step.message
    ]
    per_step = min(6_000, max(1_200, (max_chars - 8_000) // max(len(included), 1)))
    rows: list[dict[str, Any]] = []
    for step in included:
        tool_budget = max(per_step // 3, 300)
        observation_budget = max(per_step // 2, 500)
        calls = []
        for call in step.tool_calls:
            arguments = call.get("arguments", {})
            calls.append(
                {
                    "name": call.get("function_name") or call.get("name") or "tool",
                    "arguments": clip_text(
                        json.dumps(
                            redact_structure(arguments), ensure_ascii=False, default=str
                        ),
                        tool_budget,
                    ),
                }
            )
        rows.append(
            {
                "step_id": step.step_id,
                "source": step.source,
                "timestamp": step.timestamp,
                "message": clip_text(step.message, max(per_step // 2, 500)),
                "tool_calls": calls,
                "observation": clip_text(step.observation_text, observation_budget),
                "failed": step.failed,
            }
        )

    metadata = {
        "job": bundle.job_name,
        "trial": bundle.trial_name,
        "task": bundle.task_name,
        "condition": bundle.condition,
        "scenario": bundle.scenario,
        "agent_model": bundle.agent_model,
        "reward": bundle.reward,
        "metrics": redact_structure(bundle.artifacts.get("metrics", {})),
    }
    payload = json.dumps(
        {"metadata": metadata, "trajectory_steps": rows},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return clip_text(payload, max_chars)
