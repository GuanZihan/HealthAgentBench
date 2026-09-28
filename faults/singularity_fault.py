from __future__ import annotations

import asyncio
import json
import secrets
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from harbor.environments.singularity import SingularityEnvironment

FaultCondition = Literal["control", "error"]
FaultScenario = Literal[
    "file-not-found",
    "checksum-mismatch",
    "recovery-http-403",
]

SCENARIO: FaultScenario = "file-not-found"
SUPPORTED_SCENARIOS = (
    "file-not-found",
    "checksum-mismatch",
    "recovery-http-403",
)
TARGET = "/workspace/staged_demo/raw_input/hosp/admissions.csv.gz"
CHECKSUM_FILE = "/workspace/staged_demo/raw_input/SHA256SUMS.txt"
CHECKSUM_ENTRY = "hosp/admissions.csv.gz"
BLOCKED_RECOVERY_HOST = "physionet.org"
MANIFEST_NAME = "fault-setup.json"
EVENTS_NAME = "fault-events.jsonl"
FINAL_STATE_NAME = "fault-final-state.json"
RUNTIME_DIR = "/run/healthagent-fault"
PROXY_EVENTS_PATH = f"{RUNTIME_DIR}/proxy-events.jsonl"
PROXY_READY_PATH = f"{RUNTIME_DIR}/proxy-ready.json"
PROXY_PID_PATH = f"{RUNTIME_DIR}/proxy.pid"
PROXY_LOG_PATH = f"{RUNTIME_DIR}/proxy.log"
PROXY_SCRIPT_PATH = f"{RUNTIME_DIR}/selective_proxy.py"
CORRUPTION_MARKER = b"HEALTHAGENTBENCH_DETERMINISTIC_CORRUPTION\n"

_FAULT_PROGRAM = rf"""
import gzip
import hashlib
import json
import pathlib
import sys

condition = sys.argv[1]
scenario = sys.argv[2]
target = pathlib.Path({TARGET!r})
checksum_file = pathlib.Path({CHECKSUM_FILE!r})
checksum_entry = {CHECKSUM_ENTRY!r}
runtime_dir = pathlib.Path({RUNTIME_DIR!r})

if condition not in {{"control", "error"}}:
    raise ValueError(f"Unsupported condition: {{condition!r}}")
if scenario not in {set(SUPPORTED_SCENARIOS)!r}:
    raise ValueError(f"Unsupported scenario: {{scenario!r}}")
if not target.is_file():
    raise FileNotFoundError(f"Required fault target is not a regular file: {{target}}")
if not checksum_file.is_file():
    raise FileNotFoundError(f"Checksum file is missing: {{checksum_file}}")

declared_sha256 = None
for line in checksum_file.read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if len(parts) >= 2 and parts[-1].lstrip("*") == checksum_entry:
        declared_sha256 = parts[0]
        break
if declared_sha256 is None:
    raise ValueError(f"No checksum entry found for {{checksum_entry}}")

before = {{
    "exists": True,
    "size": target.stat().st_size,
    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "declared_sha256": declared_sha256,
}}
if before["size"] <= 0:
    raise ValueError(f"Required fault target is empty: {{target}}")
if before["sha256"] != declared_sha256:
    raise ValueError("Fault target does not match its declared checksum before setup")
with gzip.open(target, "rb") as stream:
    stream.read(1)
before["gzip_valid"] = True

runtime_dir.mkdir(parents=True, exist_ok=True)
if scenario == "file-not-found" and condition == "error":
    target.unlink()
elif scenario == "checksum-mismatch" and condition == "error":
    target.write_bytes({CORRUPTION_MARKER!r})
elif scenario == "recovery-http-403":
    target.unlink()

after = {{
    "exists": target.exists(),
    "is_file": target.is_file(),
}}
if target.is_file():
    after["size"] = target.stat().st_size
    after["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    try:
        with gzip.open(target, "rb") as stream:
            stream.read(1)
        after["gzip_valid"] = True
    except (OSError, EOFError):
        after["gzip_valid"] = False

print(json.dumps({{
    "condition": condition,
    "scenario": scenario,
    "target": str(target),
    "checksum_file": str(checksum_file),
    "before": before,
    "after": after,
}}, sort_keys=True))
"""

_FINAL_STATE_PROGRAM = rf"""
import gzip
import hashlib
import json
import pathlib

target = pathlib.Path({TARGET!r})
state = {{"target": str(target), "exists": target.exists(), "is_file": target.is_file()}}
if target.is_file():
    state["size"] = target.stat().st_size
    state["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    try:
        with gzip.open(target, "rb") as stream:
            stream.read(1)
        state["gzip_valid"] = True
    except (OSError, EOFError):
        state["gzip_valid"] = False
print(json.dumps(state, sort_keys=True))
"""


class FaultSetupError(RuntimeError):
    """Raised when the paired EHR-to-MEDS environment cannot be established."""


def _validate_condition(value: str) -> FaultCondition:
    if value not in {"control", "error"}:
        raise ValueError(
            f"fault_condition must be 'control' or 'error', received {value!r}"
        )
    return cast(FaultCondition, value)


def _validate_scenario(value: str) -> FaultScenario:
    if value not in SUPPORTED_SCENARIOS:
        supported = ", ".join(SUPPORTED_SCENARIOS)
        raise ValueError(
            f"fault_scenario must be one of {supported}; received {value!r}"
        )
    return cast(FaultScenario, value)


def build_fault_command(
    condition: FaultCondition, scenario: FaultScenario = SCENARIO
) -> str:
    """Build the fixed, shell-safe in-container setup command."""
    return " ".join(
        (
            "python",
            "-c",
            shlex.quote(_FAULT_PROGRAM),
            shlex.quote(condition),
            shlex.quote(scenario),
        )
    )


def parse_fault_result(stdout: str | None) -> dict[str, Any]:
    if not stdout or not stdout.strip():
        raise FaultSetupError("Fault setup returned no state report")
    try:
        result = json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, TypeError) as error:
        raise FaultSetupError("Fault setup returned an invalid state report") from error
    if not isinstance(result, dict):
        raise FaultSetupError("Fault setup state report must be a JSON object")
    return result


def _validate_common_state(
    result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = result.get("before")
    after = result.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise FaultSetupError("Fault setup report is missing file state")
    if (
        before.get("exists") is not True
        or before.get("gzip_valid") is not True
        or not isinstance(before.get("size"), int)
        or before["size"] <= 0
        or before.get("sha256") != before.get("declared_sha256")
    ):
        raise FaultSetupError("Fault setup started from an invalid target file")
    return before, after


def validate_fault_result(
    result: dict[str, Any],
    condition: FaultCondition,
    scenario: FaultScenario = SCENARIO,
) -> None:
    if (
        result.get("condition") != condition
        or result.get("scenario") != scenario
        or result.get("target") != TARGET
        or result.get("checksum_file") != CHECKSUM_FILE
    ):
        raise FaultSetupError("Fault setup reported unexpected experiment metadata")

    before, after = _validate_common_state(result)
    if scenario == "file-not-found":
        expected_exists = condition == "control"
        if after.get("exists") is not expected_exists:
            raise FaultSetupError("Fault setup did not establish the requested state")
        if condition == "control" and (
            after.get("sha256") != before.get("sha256")
            or after.get("size") != before.get("size")
            or after.get("gzip_valid") is not True
        ):
            raise FaultSetupError("Control setup did not preserve the target file")
        return

    if scenario == "checksum-mismatch":
        if after.get("exists") is not True or after.get("is_file") is not True:
            raise FaultSetupError("Checksum scenario must leave a regular file")
        if condition == "control":
            if (
                after.get("sha256") != before.get("sha256")
                or after.get("size") != before.get("size")
                or after.get("gzip_valid") is not True
            ):
                raise FaultSetupError("Control setup did not preserve the target file")
        elif (
            after.get("sha256") == before.get("declared_sha256")
            or after.get("gzip_valid") is not False
        ):
            raise FaultSetupError("Checksum mismatch was not established")
        return

    if scenario == "recovery-http-403" and after.get("exists") is not False:
        raise FaultSetupError("Recovery HTTP scenario must begin with a missing file")


class EhrToMedsFaultEnvironment(SingularityEnvironment):
    """Singularity environment with a pre-Agent EHR fault stage."""

    def __init__(
        self,
        *args: Any,
        fault_condition: str,
        fault_scenario: str = SCENARIO,
        **kwargs: Any,
    ) -> None:
        self._fault_condition = _validate_condition(fault_condition)
        self._fault_scenario = _validate_scenario(fault_scenario)
        self._proxy_started = False
        self._runtime_evidence_collected = False
        super().__init__(*args, **kwargs)
        if self.environment_name != "ehr_to_meds_etl":
            raise ValueError(
                "EhrToMedsFaultEnvironment is restricted to ehr_to_meds_etl"
            )

    def _write_fault_manifest(self, state: dict[str, Any]) -> Path:
        manifest = {
            "version": 2,
            "status": "ready",
            "injected_before_agent": True,
            "sif_modified": False,
            "writable_layer": "per-trial writable-tmpfs",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **state,
        }
        path = self.trial_paths.trial_dir / MANIFEST_NAME
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    async def _start_selective_proxy(self) -> dict[str, Any]:
        source = Path(__file__).with_name("selective_proxy.py")
        await self.upload_file(source, PROXY_SCRIPT_PATH)
        username = "healthagent"
        password = secrets.token_urlsafe(24)
        command = f"""
nohup python {shlex.quote(PROXY_SCRIPT_PATH)} \\
  --condition {shlex.quote(self._fault_condition)} \\
  --blocked-host {shlex.quote(BLOCKED_RECOVERY_HOST)} \\
  --username {shlex.quote(username)} \\
  --password {shlex.quote(password)} \\
  --events-path {shlex.quote(PROXY_EVENTS_PATH)} \\
  --ready-path {shlex.quote(PROXY_READY_PATH)} \\
  > {shlex.quote(PROXY_LOG_PATH)} 2>&1 < /dev/null &
proxy_pid=$!
printf '%s\n' "$proxy_pid" > {shlex.quote(PROXY_PID_PATH)}
attempt=0
while [ "$attempt" -lt 100 ]; do
  if [ -s {shlex.quote(PROXY_READY_PATH)} ]; then
    cat {shlex.quote(PROXY_READY_PATH)}
    exit 0
  fi
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    cat {shlex.quote(PROXY_LOG_PATH)} >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  sleep 0.1
done
printf 'Selective proxy did not become ready\n' >&2
exit 1
"""
        execution = await self.exec(command, timeout_sec=30)
        if execution.return_code != 0:
            detail = execution.stderr or execution.stdout or "<no output>"
            raise FaultSetupError(f"Failed to start selective proxy: {detail}")
        ready = parse_fault_result(execution.stdout)
        port = ready.get("port")
        if ready.get("status") != "ready" or not isinstance(port, int):
            raise FaultSetupError("Selective proxy returned invalid readiness state")

        proxy_url = f"http://{username}:{password}@127.0.0.1:{port}"
        self._persistent_env.update(
            {
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )
        self._proxy_started = True
        return {
            "blocked_host": BLOCKED_RECOVERY_HOST,
            "control_behavior": "authenticated pass-through proxy",
            "error_behavior": "HTTP CONNECT 403",
            "proxy_ready": True,
            "proxy_port": port,
        }

    async def _collect_runtime_evidence(self) -> None:
        if self._runtime_evidence_collected:
            return
        self._runtime_evidence_collected = True

        final_execution = await self.exec(
            f"python -c {shlex.quote(_FINAL_STATE_PROGRAM)}", timeout_sec=30
        )
        if final_execution.return_code == 0:
            final_state = parse_fault_result(final_execution.stdout)
            final_state.update(
                {
                    "condition": self._fault_condition,
                    "scenario": self._fault_scenario,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            (self.trial_paths.trial_dir / FINAL_STATE_NAME).write_text(
                json.dumps(final_state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        if not self._proxy_started:
            return
        events = await self.exec(
            f"test ! -f {shlex.quote(PROXY_EVENTS_PATH)} || "
            f"cat {shlex.quote(PROXY_EVENTS_PATH)}",
            timeout_sec=30,
        )
        if events.return_code == 0 and events.stdout and events.stdout.strip():
            (self.trial_paths.trial_dir / EVENTS_NAME).write_text(
                events.stdout.rstrip() + "\n", encoding="utf-8"
            )

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        try:
            execution = await self.exec(
                build_fault_command(self._fault_condition, self._fault_scenario),
                cwd="/workspace",
                timeout_sec=120,
            )
            if execution.return_code != 0:
                detail = execution.stderr or execution.stdout or "<no output>"
                raise FaultSetupError(
                    f"Failed to establish {self._fault_condition} condition: {detail}"
                )
            state = parse_fault_result(execution.stdout)
            validate_fault_result(state, self._fault_condition, self._fault_scenario)
            if self._fault_scenario == "recovery-http-403":
                state["network"] = await self._start_selective_proxy()
            manifest_path = self._write_fault_manifest(state)
            self.logger.info(
                "Established %s condition for %s; manifest: %s",
                self._fault_condition,
                self._fault_scenario,
                manifest_path,
            )
        except Exception:
            await super().stop(delete=True)
            raise

    async def stop(self, delete: bool) -> None:
        try:
            if self._http_client and self._server_port:
                await self._collect_runtime_evidence()
        except (OSError, RuntimeError, asyncio.TimeoutError, ValueError) as error:
            self.logger.warning("Could not collect fault runtime evidence: %s", error)
        finally:
            await super().stop(delete)
