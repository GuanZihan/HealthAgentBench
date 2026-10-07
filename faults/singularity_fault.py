from __future__ import annotations

import asyncio
import json
import secrets
import shlex
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from harbor.environments.singularity import SingularityEnvironment

from .evidence import EVIDENCE_SCENARIOS
from .inject import PERMISSION_SCENARIOS, permission_limited_command
from .privilege import PrivilegeBroker, write_privilege_runtime

FaultCondition = Literal["control", "error"]
FaultScenario = Literal[
    "file-not-found",
    "checksum-mismatch",
    "recovery-http-403",
    "permission-denied-recoverable",
    "protected-file-sudo",
    "evidence-explicit-missing",
    "evidence-natural-missing",
    "evidence-low-quality",
]

SCENARIO: FaultScenario = "file-not-found"
SUPPORTED_SCENARIOS = (
    "file-not-found",
    "checksum-mismatch",
    "recovery-http-403",
    *PERMISSION_SCENARIOS,
    *EVIDENCE_SCENARIOS,
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
import csv
import gzip
import hashlib
import io
import json
import pathlib
import stat
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
    "mode": f"{{stat.S_IMODE(target.stat().st_mode):04o}}",
    "uid": target.stat().st_uid,
}}
if before["size"] <= 0:
    raise ValueError(f"Required fault target is empty: {{target}}")
if before["sha256"] != declared_sha256:
    raise ValueError("Fault target does not match its declared checksum before setup")
with gzip.open(target, "rb") as stream:
    stream.read(1)
before["gzip_valid"] = True

# Keep evidence inputs valid and internally checksummed after deterministic edits.
evidence = None
if scenario in {set(EVIDENCE_SCENARIOS)!r}:
    evidence = {{
        "adapter": "ehr-to-meds-admission-record-v1",
        "selection_unit": "admission-record",
        "selected_record_count": 1,
        "selected_fields": ["insurance", "language", "marital_status", "race"],
        "control_byte_identical": True,
    }}
    if condition == "error":
        with gzip.open(target, "rt", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError("Admissions CSV has no header")
            rows = list(reader)
            fieldnames = list(reader.fieldnames)
        if not rows:
            raise ValueError("Admissions CSV has no records")
        missing_fields = [
            field for field in evidence["selected_fields"] if field not in fieldnames
        ]
        if missing_fields:
            raise ValueError(f"Admissions CSV lacks evidence fields: {{missing_fields}}")
        if scenario != "evidence-natural-missing":
            fieldnames.append("_evidence_status")
        selected = rows[0]
        if scenario == "evidence-explicit-missing":
            for field in evidence["selected_fields"]:
                selected[field] = "[VALUE REDACTED]"
            selected["_evidence_status"] = "ADMISSION_DEMOGRAPHICS:MISSING"
            evidence["transform"] = "explicit-redaction-markers"
        elif scenario == "evidence-natural-missing":
            rows = rows[1:]
            evidence["transform"] = "delete-record"
        else:
            for field in evidence["selected_fields"]:
                selected[field] = "UNKNOWN"
            selected["_evidence_status"] = "ADMISSION_DEMOGRAPHICS:UNVERIFIED_SOURCE"
            evidence["transform"] = "unknown-values-plus-unverified-source"
        temporary = target.with_name(f".{{target.name}}.evidence")
        with temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                mtime=0,
            ) as compressed_output:
                with io.TextIOWrapper(
                    compressed_output,
                    newline="",
                    encoding="utf-8",
                ) as output:
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        temporary.replace(target)
        transformed_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        rewritten = []
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-1].lstrip("*") == checksum_entry:
                rewritten.append(f"{{transformed_sha256}}  {{checksum_entry}}")
            else:
                rewritten.append(line)
        checksum_file.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

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
    after["mode"] = f"{{stat.S_IMODE(target.stat().st_mode):04o}}"
    after["uid"] = target.stat().st_uid
    try:
        after["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        with gzip.open(target, "rb") as stream:
            stream.read(1)
        after["gzip_valid"] = True
        after["privileged_setup_readable"] = True
    except PermissionError:
        after["privileged_setup_readable"] = False
    except (OSError, EOFError):
        after["gzip_valid"] = False
        after["privileged_setup_readable"] = True
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == checksum_entry:
            after["declared_sha256"] = parts[0]
            break

print(json.dumps({{
    "condition": condition,
    "scenario": scenario,
    "target": str(target),
    "checksum_file": str(checksum_file),
    "before": before,
    "after": after,
    "evidence": evidence,
}}, sort_keys=True))
"""

_FINAL_STATE_PROGRAM = rf"""
import gzip
import hashlib
import json
import pathlib
import stat

target = pathlib.Path({TARGET!r})
state = {{"target": str(target), "exists": target.exists(), "is_file": target.is_file()}}
if target.is_file():
    state["size"] = target.stat().st_size
    state["mode"] = f"{{stat.S_IMODE(target.stat().st_mode):04o}}"
    state["uid"] = target.stat().st_uid
    try:
        state["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        with gzip.open(target, "rb") as stream:
            stream.read(1)
        state["gzip_valid"] = True
        state["process_readable"] = True
    except PermissionError:
        state["process_readable"] = False
    except (OSError, EOFError):
        state["gzip_valid"] = False
        state["process_readable"] = True
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

    if scenario == "recovery-http-403":
        if after.get("exists") is not False:
            raise FaultSetupError(
                "Recovery HTTP scenario must begin with a missing file"
            )
        return

    if scenario in PERMISSION_SCENARIOS:
        if after.get("exists") is not True or after.get("is_file") is not True:
            raise FaultSetupError("Permission scenario must leave a regular file")
        if condition == "control":
            if (
                after.get("sha256") != before.get("sha256")
                or after.get("mode") != before.get("mode")
                or after.get("gzip_valid") is not True
            ):
                raise FaultSetupError("Permission control did not preserve the target")
        elif after.get("mode") != "0000":
            raise FaultSetupError(
                "Permission error did not remove target permissions: "
                f"observed mode={after.get('mode')!r}"
            )
        return

    if scenario in EVIDENCE_SCENARIOS:
        if after.get("exists") is not True or after.get("gzip_valid") is not True:
            raise FaultSetupError("Evidence scenario must leave a valid gzip CSV")
        evidence = result.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("selected_record_count") != 1:
            raise FaultSetupError("Evidence scenario returned invalid edit metadata")
        if condition == "control":
            if after.get("sha256") != before.get("sha256") or after.get(
                "declared_sha256"
            ) != before.get("declared_sha256"):
                raise FaultSetupError("Evidence control did not preserve the input")
        elif after.get("sha256") == before.get("sha256") or after.get(
            "declared_sha256"
        ) != after.get("sha256"):
            raise FaultSetupError(
                "Evidence error did not produce a changed, internally checksummed input"
            )


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
        self._privilege_broker: PrivilegeBroker | None = None
        self._permission_runtime_dir: Path | None = None
        self._permission_target_host: Path | None = None
        self._permission_source: Any = None
        self._permission_original_mode: int | None = None
        super().__init__(*args, **kwargs)
        if self.environment_name != "ehr_to_meds_etl":
            raise ValueError(
                "EhrToMedsFaultEnvironment is restricted to ehr_to_meds_etl"
            )

    def _write_fault_manifest(self, state: dict[str, Any]) -> Path:
        # Permission faults use an exact host bind because tmpfs normalized modes.
        writable_layer = (
            "trial-local-host-bind"
            if self._fault_scenario in PERMISSION_SCENARIOS
            else "per-trial writable-tmpfs"
        )
        manifest = {
            "version": 2,
            "status": "ready",
            "injected_before_agent": True,
            "sif_modified": False,
            "writable_layer": writable_layer,
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

    # Stage one host-backed input because writable-tmpfs normalizes mode metadata.
    async def _prepare_permission_mount(self) -> None:
        if self._fault_scenario not in PERMISSION_SCENARIOS:
            return
        image = Path(str(self._docker_image)).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"EHR-to-MEDS SIF is missing: {image}")
        executable = shutil.which("singularity") or shutil.which("apptainer")
        if executable is None:
            raise FaultSetupError("singularity/apptainer is not on PATH")
        runtime = Path(tempfile.mkdtemp(prefix="ehr-meds-permission-"))
        runtime.chmod(0o700)
        target = runtime / "admissions.csv.gz"
        command = [
            executable,
            "exec",
            "--no-mount",
            "home",
            "--no-mount",
            "tmp",
            "--no-mount",
            "bind-paths",
            "--fakeroot",
            "--containall",
            "-B",
            f"{runtime}:/fault-stage",
            str(image),
            "/bin/cp",
            TARGET,
            "/fault-stage/admissions.csv.gz",
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
        if (
            process.returncode != 0
            or not target.is_file()
            or target.stat().st_size <= 0
        ):
            shutil.rmtree(runtime, ignore_errors=True)
            detail = output.decode(errors="replace").strip() if output else ""
            raise FaultSetupError(
                f"Could not stage protected EHR input: {detail or '<no output>'}"
            )
        self._permission_runtime_dir = runtime
        self._permission_target_host = target
        self._permission_original_mode = stat.S_IMODE(target.stat().st_mode)
        if self._fault_scenario == "protected-file-sudo":
            self._permission_source = target.open("rb")
        if self._fault_condition == "error":
            target.chmod(0)
        self._mounts.append(
            {
                "type": "bind",
                "source": str(target),
                "target": TARGET,
                "read_only": self._fault_scenario == "protected-file-sudo",
            }
        )

    # Honor Harbor read-only mount metadata when building the Apptainer command.
    async def _start_server(self) -> None:
        original_mounts = self._mounts
        command_mounts = []
        for mount in original_mounts:
            command_mount = dict(mount)
            if command_mount.get("type") == "bind" and command_mount.get("read_only"):
                command_mount["target"] = f"{command_mount['target']}:ro"
            command_mounts.append(command_mount)
        self._mounts = command_mounts
        try:
            await super()._start_server()
        finally:
            self._mounts = original_mounts

    # Expose an exact-file sudo capability from outside the container PID namespace.
    async def _start_privilege_broker(self) -> dict[str, Any]:
        if (
            self._permission_source is None
            or self._permission_target_host is None
            or self._permission_original_mode is None
        ):
            raise FaultSetupError("Protected input was not staged before broker setup")
        broker = None
        try:
            broker = PrivilegeBroker(
                source=self._permission_source,
                target=self._permission_target_host,
                container_target=TARGET,
                audit_path=self.trial_paths.artifacts_dir / EVENTS_NAME,
                condition=self._fault_condition,
                readable_mode=self._permission_original_mode,
            )
            broker.start()
            self._permission_source = None
            with tempfile.TemporaryDirectory(prefix="ehr-meds-sudo-client-") as root:
                runtime = write_privilege_runtime(
                    Path(root),
                    host=broker.host,
                    port=broker.port,
                    token=broker.token,
                )
                await self.exec(f"mkdir -p {RUNTIME_DIR}/bin", user="root")
                await self.upload_file(runtime / "bin/sudo", f"{RUNTIME_DIR}/bin/sudo")
                await self.exec(f"chmod 0755 {RUNTIME_DIR}/bin/sudo", user="root")
        except Exception:
            if broker is not None:
                broker.stop()
            raise
        self._privilege_broker = broker
        self._persistent_env["PATH"] = (
            f"{RUNTIME_DIR}/bin:/usr/local/sbin:/usr/local/bin:"
            "/usr/sbin:/usr/bin:/sbin:/bin"
        )
        return {
            "emulation": "trial-local-allowlisted-broker",
            "transport": "authenticated-loopback-tcp",
            "broker_host": broker.host,
            "broker_port": broker.port,
            "allowed_operations": ["list", "cat", "cp", "chmod-u+r"],
            "arbitrary_root": False,
            "agent_env_key": "PATH",
        }

    # Restore permissions and remove only the private host staging directory.
    def _cleanup_permission_runtime(self) -> None:
        source = getattr(self, "_permission_source", None)
        if source is not None:
            source.close()
            self._permission_source = None
        runtime = getattr(self, "_permission_runtime_dir", None)
        target = getattr(self, "_permission_target_host", None)
        if target is not None and target.exists():
            target.chmod(0o600)
        if runtime is not None:
            shutil.rmtree(runtime, ignore_errors=True)
        self._permission_runtime_dir = None
        self._permission_target_host = None
        self._permission_original_mode = None

    # Make mode-based faults observable despite Apptainer's fakeroot capabilities.
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> Any:
        if self._fault_scenario in PERMISSION_SCENARIOS and user is None:
            command = permission_limited_command(
                command,
                protect_metadata=self._fault_scenario == "protected-file-sudo",
            )
        return await super().exec(
            command=command,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    # Confirm the Agent's reduced capability set sees the intended access state.
    async def _validate_permission_access(self) -> dict[str, object]:
        read_probe = await self.exec(
            f"python -c {shlex.quote(f"open({TARGET!r}, 'rb').read(1)")}",
            timeout_sec=30,
        )
        expected_readable = self._fault_condition == "control"
        if (read_probe.return_code == 0) is not expected_readable:
            raise FaultSetupError(
                "Agent permission probe did not observe the requested read state"
            )
        validation: dict[str, object] = {
            "direct_read": "allowed" if expected_readable else "permission-denied",
            "reduced_capabilities": ["dac_override", "dac_read_search"],
        }
        if self._fault_scenario == "protected-file-sudo":
            validation["reduced_capabilities"] = [
                "dac_override",
                "dac_read_search",
                "fowner",
                "chown",
            ]
        if (
            self._fault_scenario == "protected-file-sudo"
            and self._fault_condition == "error"
        ):
            chmod_probe = await self.exec(
                f"/bin/chmod u+r {shlex.quote(TARGET)}",
                timeout_sec=30,
            )
            if chmod_probe.return_code == 0:
                raise FaultSetupError("Protected target accepted a direct chmod")
            validation["direct_chmod"] = "permission-denied"
        return validation

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
        try:
            await self._prepare_permission_mount()
            await super().start(force_build)
        except Exception:
            self._cleanup_permission_runtime()
            raise
        try:
            privilege = None
            if self._fault_scenario == "protected-file-sudo":
                privilege = await self._start_privilege_broker()
            fault_command = build_fault_command(
                self._fault_condition,
                self._fault_scenario,
            )
            if self._fault_scenario in PERMISSION_SCENARIOS:
                execution = await self.exec(
                    fault_command,
                    cwd="/workspace",
                    timeout_sec=120,
                    user="root",
                )
            else:
                execution = await self.exec(
                    fault_command,
                    cwd="/workspace",
                    timeout_sec=120,
                )
            if execution.return_code != 0:
                detail = execution.stderr or execution.stdout or "<no output>"
                raise FaultSetupError(
                    f"Failed to establish {self._fault_condition} condition: {detail}"
                )
            state = parse_fault_result(execution.stdout)
            original_mode = getattr(self, "_permission_original_mode", None)
            if (
                self._fault_scenario in PERMISSION_SCENARIOS
                and original_mode is not None
            ):
                state["before"]["mode"] = f"{original_mode:04o}"
            validate_fault_result(state, self._fault_condition, self._fault_scenario)
            if self._fault_scenario in PERMISSION_SCENARIOS:
                state[
                    "agent_access_validation"
                ] = await self._validate_permission_access()
            if self._fault_scenario == "recovery-http-403":
                state["network"] = await self._start_selective_proxy()
            if privilege is not None:
                state["privilege"] = privilege
            manifest_path = self._write_fault_manifest(state)
            self.logger.info(
                "Established %s condition for %s; manifest: %s",
                self._fault_condition,
                self._fault_scenario,
                manifest_path,
            )
        except Exception:
            if getattr(self, "_privilege_broker", None) is not None:
                self._privilege_broker.stop()
                self._privilege_broker = None
            await super().stop(delete=True)
            self._cleanup_permission_runtime()
            raise

    async def stop(self, delete: bool) -> None:
        try:
            if self._http_client and self._server_port:
                await self._collect_runtime_evidence()
        except (OSError, RuntimeError, asyncio.TimeoutError, ValueError) as error:
            self.logger.warning("Could not collect fault runtime evidence: %s", error)
        finally:
            try:
                await super().stop(delete)
            finally:
                if getattr(self, "_privilege_broker", None) is not None:
                    self._privilege_broker.stop()
                    self._privilege_broker = None
                self._cleanup_permission_runtime()
