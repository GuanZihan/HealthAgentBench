"""Launch a matched environmental-fault trial with a family-level SIF."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from .family_singularity import family_for_task
from .inject import AGENT_RUNTIME_PATH, DEFAULT_AGENT_PATH, SUPPORTED_SCENARIOS


def _resolve_task(path: Path, repository_root: Path) -> Path:
    for candidate in (path, repository_root / path, repository_root / "tasks" / path):
        resolved = candidate.expanduser().resolve()
        if (resolved / "task.toml").is_file():
            return resolved
    raise FileNotFoundError(f"Could not find a task.toml below {path}")


def _contains_option(arguments: list[str], names: set[str]) -> bool:
    return any(argument.split("=", 1)[0] in names for argument in arguments)


def _agent_env_keys(arguments: list[str]) -> set[str]:
    keys: set[str] = set()
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        value: str | None = None
        if argument in {"--agent-env", "--ae"} and index + 1 < len(arguments):
            value = arguments[index + 1]
            index += 1
        elif argument.startswith("--agent-env=") or argument.startswith("--ae="):
            value = argument.split("=", 1)[1]
        if value and "=" in value:
            keys.add(value.split("=", 1)[0])
        index += 1
    return keys


def _harbor_executable(requested: str | None, repository_root: Path) -> str:
    if requested:
        return requested
    available = shutil.which("harbor")
    if available:
        return available
    local = repository_root / ".venv" / "bin" / "harbor"
    if local.is_file():
        return str(local)
    raise FileNotFoundError(
        "harbor is not on PATH; invoke this module through `uv run python -m` "
        "or pass --harbor-executable"
    )


def _dependency_agent_env(task_name: str) -> str:
    dependency = family_for_task(task_name).dependency
    if dependency.kind == "python-module":
        return f"PYTHONPATH={AGENT_RUNTIME_PATH}"
    return f"PATH={DEFAULT_AGENT_PATH}"


def build_harbor_command(
    *,
    harbor: str,
    task_dir: Path,
    scenario: str,
    condition: str,
    singularity_cache_dir: Path,
    harbor_arguments: list[str],
) -> tuple[list[str], str | None]:
    family_for_task(task_dir.name)
    forbidden = {
        "--path",
        "--extra-docker-compose",
        "--environment-import-path",
        "--environment-type",
        "--environment-kwarg",
        "--mounts",
    }
    if _contains_option(harbor_arguments, forbidden):
        raise ValueError(
            "The SIF fault runner owns --path, environment selection/kwargs, and "
            "--mounts"
        )

    agent_env = None
    if scenario == "missing-dependency":
        agent_env = _dependency_agent_env(task_dir.name)
        key = agent_env.split("=", 1)[0]
        if key in _agent_env_keys(harbor_arguments):
            raise ValueError(
                f"The missing-dependency runtime owns Agent environment key {key}"
            )

    resolv_mount = json.dumps(
        [
            {
                "type": "bind",
                "source": "/etc/resolv.conf",
                "target": "/etc/resolv.conf",
                "read_only": True,
            }
        ],
        separators=(",", ":"),
    )
    command = [
        harbor,
        "run",
        "--path",
        str(task_dir),
        "--environment-import-path",
        "faults.family_singularity:FamilySingularityEnvironment",
        "--environment-kwarg",
        f"fault_condition={condition}",
        "--environment-kwarg",
        f"fault_scenario={scenario}",
        "--environment-kwarg",
        f"singularity_image_cache_dir={singularity_cache_dir}",
        "--mounts",
        resolv_mount,
    ]
    if agent_env is not None:
        command.extend(("--agent-env", agent_env))
    command.extend(harbor_arguments)
    return command, agent_env


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--scenario", choices=SUPPORTED_SCENARIOS, required=True)
    parser.add_argument("--condition", choices=("control", "error"), required=True)
    parser.add_argument("--harbor-executable")
    parser.add_argument(
        "--singularity-cache-dir",
        type=Path,
        default=root / ".cache" / "harbor-sif",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print without launching Harbor."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, harbor_arguments = parser.parse_known_args(argv)
    if harbor_arguments and harbor_arguments[0] == "--":
        harbor_arguments = harbor_arguments[1:]

    repository_root = Path(__file__).resolve().parents[1]
    try:
        task_dir = _resolve_task(args.task, repository_root)
        spec = family_for_task(task_dir.name)
        cache_dir = args.singularity_cache_dir.expanduser()
        if not cache_dir.is_absolute():
            cache_dir = repository_root / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        harbor = _harbor_executable(args.harbor_executable, repository_root)
        command, agent_env = build_harbor_command(
            harbor=harbor,
            task_dir=task_dir,
            scenario=args.scenario,
            condition=args.condition,
            singularity_cache_dir=cache_dir.resolve(),
            harbor_arguments=harbor_arguments,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    summary = {
        "task": task_dir.name,
        "family": spec.name,
        "image": str(repository_root / "containers" / spec.image_name),
        "scenario": args.scenario,
        "condition": args.condition,
        "agent_env_key": agent_env.split("=", 1)[0] if agent_env else None,
        "backend": "apptainer-sif",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Harbor command:")
    print(shlex.join(command))
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=repository_root, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
