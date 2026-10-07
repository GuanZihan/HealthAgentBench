"""Deterministic file and dependency faults for family-level SIF runtimes."""

from __future__ import annotations

import gzip
import glob
import hashlib
import json
import os
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

Action = Literal["remove", "corrupt", "truncate"]
Condition = Literal["control", "error"]
Selector = Literal["exact", "single-glob", "latest-glob"]
Scenario = Literal[
    "file-not-found",
    "corrupt-file",
    "truncated-file",
    "missing-dependency",
    "permission-denied-recoverable",
    "protected-file-sudo",
    "evidence-explicit-missing",
    "evidence-natural-missing",
    "evidence-low-quality",
]
DependencyKind = Literal["python-module", "executable"]

FILE_ACTIONS: dict[str, Action] = {
    "file-not-found": "remove",
    "corrupt-file": "corrupt",
    "truncated-file": "truncate",
}
PERMISSION_SCENARIOS = (
    "permission-denied-recoverable",
    "protected-file-sudo",
)
EVIDENCE_SCENARIOS = (
    "evidence-explicit-missing",
    "evidence-natural-missing",
    "evidence-low-quality",
)
SUPPORTED_SCENARIOS = (
    *FILE_ACTIONS,
    "missing-dependency",
    *PERMISSION_SCENARIOS,
    *EVIDENCE_SCENARIOS,
)
AGENT_RUNTIME_PATH = "/opt/healthagent-fault"
DEFAULT_AGENT_PATH = (
    f"{AGENT_RUNTIME_PATH}/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin"
)
READ_BYPASS_CAPABILITIES = "-dac_override,-dac_read_search"
METADATA_BYPASS_CAPABILITIES = ",-fowner,-chown"
CORRUPTION_MARKER = b"HEALTHAGENTBENCH_CORRUPTED_INPUT\n"


@dataclass(frozen=True)
class PrimaryInput:
    allowed_root: PurePosixPath
    selector: Selector
    path: str


@dataclass(frozen=True)
class Dependency:
    kind: DependencyKind
    name: str
    real_path: PurePosixPath | None = None


class InjectionError(RuntimeError):
    """Raised when a fault cannot be applied or validated safely."""


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_target(path: Path, allowed_root: Path) -> Path:
    root = allowed_root.resolve(strict=True)
    if root == Path(root.anchor):
        raise InjectionError("Refusing to use the filesystem root as allowed-root")

    resolved = path.resolve(strict=True)
    if not _is_within(resolved, root):
        raise InjectionError(f"Target escapes allowed root: {resolved}")
    if not resolved.is_file():
        raise InjectionError(f"Target must be a regular file: {resolved}")
    return resolved


def select_target(
    *, selector: Selector, target_expression: str, allowed_root: Path
) -> Path:
    if selector == "exact":
        return _validate_target(Path(target_expression), allowed_root)

    matches = sorted(Path(match) for match in glob.glob(target_expression))
    if not matches:
        raise InjectionError(f"No files match target expression: {target_expression}")
    if selector == "single-glob" and len(matches) != 1:
        raise InjectionError(
            f"Expected exactly one match for {target_expression!r}, got {len(matches)}"
        )
    selected = matches[-1] if selector == "latest-glob" else matches[0]
    return _validate_target(selected, allowed_root)


def apply_fault(*, action: Action, condition: Condition, target: Path) -> None:
    initial_size = target.stat().st_size
    if initial_size == 0:
        raise InjectionError(f"Prepared target is unexpectedly empty: {target}")

    if condition == "control":
        return

    if action == "remove":
        target.unlink()
        if target.exists():
            raise InjectionError(f"Failed to remove target: {target}")
        return

    if action == "corrupt":
        target.write_bytes(CORRUPTION_MARKER)
        if target.read_bytes() != CORRUPTION_MARKER:
            raise InjectionError(f"Failed to corrupt target deterministically: {target}")
        return

    if action == "truncate":
        target.write_bytes(b"")
        if target.stat().st_size != 0:
            raise InjectionError(f"Failed to truncate target: {target}")
        return

    raise InjectionError(f"Unsupported action: {action}")


# Apply a metadata-only permission fault to the per-trial runtime copy.
def apply_permission_fault(*, condition: Condition, target: Path) -> None:
    if not target.is_file() or target.stat().st_size <= 0:
        raise InjectionError(f"Prepared target is not a non-empty file: {target}")
    if condition == "control":
        return
    target.chmod(0)
    if stat.S_IMODE(target.stat().st_mode) != 0:
        raise InjectionError(f"Failed to remove target permissions: {target}")


# Remove only capabilities that let fakeroot bypass or rewrite file permissions.
def permission_limited_command(
    command: str,
    *,
    protect_metadata: bool = True,
) -> str:
    capabilities = READ_BYPASS_CAPABILITIES
    if protect_metadata:
        capabilities += METADATA_BYPASS_CAPABILITIES
    return " ".join(
        (
            "/usr/bin/setpriv",
            "--no-new-privs",
            f"--bounding-set={capabilities}",
            f"--inh-caps={capabilities}",
            f"--ambient-caps={capabilities}",
            "/bin/bash",
            "-c",
            shlex.quote(command),
        )
    )


def file_state(path: Path) -> dict[str, object]:
    state: dict[str, object] = {
        "exists": path.exists(),
        "is_file": path.is_file(),
    }
    if not path.is_file():
        return state

    metadata = path.stat()
    state.update(
        {
            "size": metadata.st_size,
            "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
            "owner_read": bool(metadata.st_mode & stat.S_IRUSR),
            "process_readable": os.access(path, os.R_OK),
        }
    )

    # Preserve useful permission metadata even when the injected file is unreadable.
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except PermissionError:
        state["sha256_available"] = False
        return state
    state.update({"sha256": digest.hexdigest(), "sha256_available": True})

    if path.suffix == ".gz":
        content_digest = hashlib.sha256()
        try:
            with gzip.open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    content_digest.update(chunk)
        except (EOFError, OSError):
            state["gzip_valid"] = False
        else:
            state.update(
                {
                    "gzip_valid": True,
                    "gzip_content_sha256": content_digest.hexdigest(),
                }
            )
    return state


def _sitecustomize_source(dependency: Dependency, condition: Condition) -> str:
    blocked_root = dependency.name.split(".", 1)[0] if condition == "error" else None
    return f'''# Generated by HealthAgentBench faults; do not edit.
from __future__ import annotations

import importlib.abc
import json
from datetime import datetime, timezone

_BLOCKED_ROOT = {blocked_root!r}
_AUDIT_PATH = "/logs/artifacts/fault-events.jsonl"


def _record(requested: str) -> None:
    event = {{
        "event": "missing_dependency_import",
        "dependency": _BLOCKED_ROOT,
        "requested_module": requested,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }}
    try:
        with open(_AUDIT_PATH, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\\n")
    except OSError:
        pass


class _MissingDependencyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if _BLOCKED_ROOT and (
            fullname == _BLOCKED_ROOT or fullname.startswith(_BLOCKED_ROOT + ".")
        ):
            _record(fullname)
            raise ModuleNotFoundError(
                f"No module named '{{_BLOCKED_ROOT}}'", name=_BLOCKED_ROOT
            )
        return None


if _BLOCKED_ROOT:
    import sys

    sys.meta_path.insert(0, _MissingDependencyFinder())
'''


def _executable_wrapper_source(
    dependency: Dependency, condition: Condition
) -> str:
    if dependency.real_path is None:
        raise InjectionError("Executable dependency has no real_path")
    real_path = shlex.quote(str(dependency.real_path))
    if condition == "control":
        return f'#!/bin/bash\nset -euo pipefail\nexec {real_path} "$@"\n'

    event = json.dumps(
        {
            "event": "missing_dependency_executable",
            "dependency": dependency.name,
        },
        sort_keys=True,
    )
    return f'''#!/bin/bash
set -euo pipefail
printf '%s\\n' {shlex.quote(event)} >> /logs/artifacts/fault-events.jsonl 2>/dev/null || true
printf '%s\\n' {shlex.quote(f"{dependency.name}: command not found")} >&2
exit 127
'''


def write_dependency_runtime(
    directory: Path, dependency: Dependency, condition: Condition
) -> tuple[Path, str]:
    """Create an Agent-only shim and return its mount source and env entry."""

    runtime = directory / "dependency-runtime"
    runtime.mkdir(parents=True, exist_ok=False)

    if dependency.kind == "python-module":
        (runtime / "sitecustomize.py").write_text(
            _sitecustomize_source(dependency, condition), encoding="utf-8"
        )
        return runtime, f"PYTHONPATH={AGENT_RUNTIME_PATH}"

    bin_directory = runtime / "bin"
    bin_directory.mkdir()
    wrapper = bin_directory / dependency.name
    wrapper.write_text(
        _executable_wrapper_source(dependency, condition), encoding="utf-8"
    )
    wrapper.chmod(0o755)
    return runtime, f"PATH={DEFAULT_AGENT_PATH}"


def dependency_validation_command(dependency: Dependency) -> tuple[str, ...]:
    if dependency.kind == "python-module":
        return (
            "python",
            "-c",
            f"import importlib; importlib.import_module({dependency.name!r})",
        )
    if dependency.real_path is None:
        raise InjectionError("Executable dependency has no real_path")
    return ("/usr/bin/test", "-x", str(dependency.real_path))
