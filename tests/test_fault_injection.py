from __future__ import annotations

import base64
import functools
import gzip
import http.server
import json
import logging
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path, PurePosixPath

import pytest
from harbor.environments.base import ExecResult
from harbor.environments.singularity import SingularityEnvironment
from harbor.models.trial.paths import TrialPaths

from faults.family_singularity import family_for_task
from faults.inject import (
    CORRUPTION_MARKER,
    Dependency,
    InjectionError,
    apply_fault,
    file_state,
    select_target,
    write_dependency_runtime,
)
from faults.run_fault import build_harbor_command
from faults.selective_proxy import SelectiveProxyServer, create_proxy_server
from faults.singularity_fault import (
    CHECKSUM_FILE,
    CORRUPTION_MARKER as SIF_CORRUPTION_MARKER,
    SCENARIO,
    SUPPORTED_SCENARIOS,
    TARGET,
    EhrToMedsFaultEnvironment,
    FaultSetupError,
    build_fault_command,
    parse_fault_result,
    validate_fault_result,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("remove", None),
        ("corrupt", CORRUPTION_MARKER),
        ("truncate", b""),
    ],
)
def test_apply_error_faults(
    tmp_path: Path, action: str, expected: bytes | None
) -> None:
    target = tmp_path / "input.dat"
    target.write_bytes(b"valid input\n")

    apply_fault(action=action, condition="error", target=target)

    if expected is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == expected


@pytest.mark.parametrize("action", ["remove", "corrupt", "truncate"])
def test_control_condition_is_read_only(tmp_path: Path, action: str) -> None:
    target = tmp_path / "input.dat"
    original = b"valid input\n"
    target.write_bytes(original)

    apply_fault(action=action, condition="control", target=target)

    assert target.read_bytes() == original


def test_gzip_state_separates_metadata_from_content(tmp_path: Path) -> None:
    states = []
    for index, mtime in enumerate((1, 2)):
        path = tmp_path / f"input-{index}.csv.gz"
        with path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=mtime) as stream:
                stream.write(b"subject_id,value\n1,42\n")
        states.append(file_state(path))

    assert states[0]["sha256"] != states[1]["sha256"]
    assert states[0]["gzip_valid"] is True
    assert states[0]["gzip_content_sha256"] == states[1]["gzip_content_sha256"]


def test_latest_glob_selects_lexically_latest_file(tmp_path: Path) -> None:
    first = tmp_path / "study_01" / "report.txt"
    latest = tmp_path / "study_02" / "report.txt"
    first.parent.mkdir()
    latest.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    latest.write_text("latest", encoding="utf-8")

    selected = select_target(
        selector="latest-glob",
        target_expression=str(tmp_path / "study_*" / "report.txt"),
        allowed_root=tmp_path,
    )

    assert selected == latest


def test_selector_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("do not touch", encoding="utf-8")
    link = allowed / "link.txt"
    link.symlink_to(secret)

    with pytest.raises(InjectionError, match="escapes allowed root"):
        select_target(
            selector="exact",
            target_expression=str(link),
            allowed_root=allowed,
        )

    assert secret.read_text(encoding="utf-8") == "do not touch"


def test_family_fault_specs_cover_all_sif_families() -> None:
    tasks = sorted(
        path.parent
        for path in (REPOSITORY_ROOT / "tasks").glob("*/task.toml")
        if path.parent.name != "ehr_to_meds_etl"
    )
    assert len(tasks) == 53
    for task in tasks:
        spec = family_for_task(task.name)
        assert spec.primary_input.path.startswith("/")
        assert spec.dependency.name

    with pytest.raises(ValueError, match="matched none"):
        family_for_task("ehr_to_meds_etl")


@pytest.mark.parametrize(
    ("scenario", "agent_env_key"),
    (("file-not-found", None), ("missing-dependency", "PYTHONPATH")),
)
def test_sif_fault_runner_has_no_compose_arguments(
    tmp_path: Path, scenario: str, agent_env_key: str | None
) -> None:
    task_dir = REPOSITORY_ROOT / "tasks" / "ehr_data_quality_task_combined"
    command, agent_env = build_harbor_command(
        harbor="harbor",
        task_dir=task_dir,
        scenario=scenario,
        condition="error",
        singularity_cache_dir=tmp_path,
        harbor_arguments=["--agent", "codex", "--model", "gpt-5.6-luna"],
    )

    assert "--extra-docker-compose" not in command
    assert "faults.family_singularity:FamilySingularityEnvironment" in command
    assert "fault_condition=error" in command
    assert f"fault_scenario={scenario}" in command
    if agent_env_key is None:
        assert agent_env is None
    else:
        assert agent_env is not None
        assert agent_env.split("=", 1)[0] == agent_env_key


def _run_import(runtime: Path, module_root: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(runtime), str(module_root)))
    return subprocess.run(
        [sys.executable, "-c", "import fault_probe_dependency"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_python_dependency_fault_is_agent_scoped(tmp_path: Path) -> None:
    module_root = tmp_path / "modules"
    module = module_root / "fault_probe_dependency"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    dependency = Dependency(kind="python-module", name="fault_probe_dependency")

    control_root = tmp_path / "control"
    control_root.mkdir()
    control_runtime, control_env = write_dependency_runtime(
        control_root, dependency, "control"
    )
    assert control_env == "PYTHONPATH=/opt/healthagent-fault"
    assert _run_import(control_runtime, module_root).returncode == 0

    error_root = tmp_path / "error"
    error_root.mkdir()
    error_runtime, error_env = write_dependency_runtime(
        error_root, dependency, "error"
    )
    assert error_env == "PYTHONPATH=/opt/healthagent-fault"
    result = _run_import(error_runtime, module_root)
    assert result.returncode != 0
    assert "No module named 'fault_probe_dependency'" in result.stderr

    # Without the Agent-only PYTHONPATH, the dependency remains available to
    # Harbor's verifier process.
    clean_environment = os.environ.copy()
    clean_environment["PYTHONPATH"] = str(module_root)
    verifier_result = subprocess.run(
        [sys.executable, "-c", "import fault_probe_dependency"],
        env=clean_environment,
        capture_output=True,
        check=False,
    )
    assert verifier_result.returncode == 0


def test_executable_dependency_control_and_error(tmp_path: Path) -> None:
    real = tmp_path / "probe_tool"
    real.write_text("#!/bin/bash\nprintf 'available\\n'\n", encoding="utf-8")
    real.chmod(0o755)
    dependency = Dependency(
        kind="executable",
        name="probe_tool",
        real_path=PurePosixPath(str(real)),
    )

    control_root = tmp_path / "control"
    control_root.mkdir()
    control_runtime, _ = write_dependency_runtime(control_root, dependency, "control")
    control = subprocess.run(
        [str(control_runtime / "bin" / "probe_tool")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert control.returncode == 0
    assert control.stdout == "available\n"

    error_root = tmp_path / "error"
    error_root.mkdir()
    error_runtime, _ = write_dependency_runtime(error_root, dependency, "error")
    error = subprocess.run(
        [str(error_runtime / "bin" / "probe_tool")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert error.returncode == 127
    assert "probe_tool: command not found" in error.stderr


def _singularity_state(condition: str, scenario: str = SCENARIO) -> dict[str, object]:
    digest = "a" * 64
    before = {
        "exists": True,
        "size": 123,
        "sha256": digest,
        "declared_sha256": digest,
        "gzip_valid": True,
    }
    if scenario == "file-not-found":
        after = (
            {
                "exists": True,
                "is_file": True,
                "size": 123,
                "sha256": digest,
                "gzip_valid": True,
            }
            if condition == "control"
            else {"exists": False, "is_file": False}
        )
    elif scenario == "checksum-mismatch":
        after = (
            {
                "exists": True,
                "is_file": True,
                "size": 123,
                "sha256": digest,
                "gzip_valid": True,
            }
            if condition == "control"
            else {
                "exists": True,
                "is_file": True,
                "size": len(SIF_CORRUPTION_MARKER),
                "sha256": "b" * 64,
                "gzip_valid": False,
            }
        )
    else:
        after = {"exists": False, "is_file": False}
    return {
        "condition": condition,
        "scenario": scenario,
        "target": TARGET,
        "checksum_file": CHECKSUM_FILE,
        "before": before,
        "after": after,
    }


@pytest.mark.parametrize("condition", ["control", "error"])
@pytest.mark.parametrize("scenario", SUPPORTED_SCENARIOS)
def test_singularity_fault_state_validation(condition: str, scenario: str) -> None:
    state = _singularity_state(condition, scenario)
    parsed = parse_fault_result("setup log\n" + json.dumps(state))

    validate_fault_result(  # type: ignore[arg-type]
        parsed, condition, scenario
    )
    command = build_fault_command(condition, scenario)  # type: ignore[arg-type]
    assert command.startswith("python -c ")
    assert condition in command
    assert scenario in command


def test_singularity_fault_state_rejects_wrong_after_state() -> None:
    state = _singularity_state("error")
    state["after"] = {"exists": True, "is_file": True}

    with pytest.raises(FaultSetupError, match="requested state"):
        validate_fault_result(state, "error")


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["control", "error"])
@pytest.mark.parametrize("scenario", ["file-not-found", "checksum-mismatch"])
async def test_singularity_environment_injects_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    scenario: str,
) -> None:
    calls: list[str] = []

    async def fake_start(self: SingularityEnvironment, force_build: bool) -> None:
        calls.append(f"start:{force_build}")

    async def fake_stop(self: SingularityEnvironment, delete: bool) -> None:
        calls.append(f"stop:{delete}")

    async def fake_exec(
        self: EhrToMedsFaultEnvironment,
        command: str,
        **kwargs: object,
    ) -> ExecResult:
        calls.append("inject")
        assert TARGET in command
        assert kwargs == {"cwd": "/workspace", "timeout_sec": 120}
        return ExecResult(
            stdout=json.dumps(_singularity_state(condition, scenario)),
            return_code=0,
        )

    monkeypatch.setattr(SingularityEnvironment, "start", fake_start)
    monkeypatch.setattr(SingularityEnvironment, "stop", fake_stop)
    monkeypatch.setattr(EhrToMedsFaultEnvironment, "exec", fake_exec)

    environment = object.__new__(EhrToMedsFaultEnvironment)
    environment._fault_condition = condition  # type: ignore[assignment]
    environment._fault_scenario = scenario  # type: ignore[assignment]
    environment._proxy_started = False
    environment._runtime_evidence_collected = False
    environment.trial_paths = TrialPaths(tmp_path)
    environment.logger = logging.getLogger("singularity-fault-test")

    await environment.start(force_build=False)

    assert calls == ["start:False", "inject"]
    manifest = json.loads((tmp_path / "fault-setup.json").read_text())
    assert manifest["condition"] == condition
    assert manifest["scenario"] == scenario
    assert manifest["injected_before_agent"] is True
    assert manifest["sif_modified"] is False


@pytest.mark.asyncio
async def test_singularity_environment_stops_when_injection_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fake_start(self: SingularityEnvironment, force_build: bool) -> None:
        calls.append("start")

    async def fake_stop(self: SingularityEnvironment, delete: bool) -> None:
        calls.append(f"stop:{delete}")

    async def fake_exec(
        self: EhrToMedsFaultEnvironment,
        command: str,
        **kwargs: object,
    ) -> ExecResult:
        calls.append("inject")
        return ExecResult(stderr="target missing", return_code=1)

    monkeypatch.setattr(SingularityEnvironment, "start", fake_start)
    monkeypatch.setattr(SingularityEnvironment, "stop", fake_stop)
    monkeypatch.setattr(EhrToMedsFaultEnvironment, "exec", fake_exec)

    environment = object.__new__(EhrToMedsFaultEnvironment)
    environment._fault_condition = "error"
    environment._fault_scenario = SCENARIO
    environment._proxy_started = False
    environment._runtime_evidence_collected = False
    environment.trial_paths = TrialPaths(tmp_path)
    environment.logger = logging.getLogger("singularity-fault-test")

    with pytest.raises(FaultSetupError, match="target missing"):
        await environment.start(force_build=False)

    assert calls == ["start", "inject", "stop:True"]
    assert not (tmp_path / "fault-setup.json").exists()


def _start_proxy(server: SelectiveProxyServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _proxy_authorization(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def _receive_until(stream: socket.socket, marker: bytes) -> bytes:
    received = bytearray()
    while marker not in received:
        chunk = stream.recv(4096)
        if not chunk:
            break
        received.extend(chunk)
    return bytes(received)


def test_selective_proxy_407_includes_basic_authentication_challenge(
    tmp_path: Path,
) -> None:
    server = create_proxy_server(
        condition="error",
        blocked_host="physionet.org",
        username="agent",
        password="test-secret",
        events_path=tmp_path / "events.jsonl",
    )
    thread = _start_proxy(server)
    try:
        with socket.create_connection(server.server_address, timeout=5) as stream:
            stream.sendall(
                b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com:443\r\n\r\n"
            )
            response = _receive_until(stream, b"\r\n\r\n")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.startswith(b"HTTP/1.1 407 Proxy Authentication Required")
    assert b'Proxy-Authenticate: Basic realm="healthagent-fault"\r\n' in response


def test_selective_proxy_returns_403_and_records_physionet_exposure(
    tmp_path: Path,
) -> None:
    username = "agent"
    password = "test-secret"
    events_path = tmp_path / "events.jsonl"
    server = create_proxy_server(
        condition="error",
        blocked_host="physionet.org",
        username=username,
        password=password,
        events_path=events_path,
    )
    thread = _start_proxy(server)
    try:
        with socket.create_connection(server.server_address, timeout=5) as stream:
            authorization = _proxy_authorization(username, password)
            stream.sendall(
                (
                    "CONNECT physionet.org:443 HTTP/1.1\r\n"
                    "Host: physionet.org:443\r\n"
                    f"Proxy-Authorization: {authorization}\r\n\r\n"
                ).encode()
            )
            response = _receive_until(stream, b"\r\n\r\n")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.startswith(b"HTTP/1.1 403 Forbidden")
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["host"] == "physionet.org"
    assert events[0]["action"] == "blocked"
    assert events[0]["status"] == 403


def test_selective_proxy_control_forwards_same_connect_request(
    tmp_path: Path,
) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    upstream_port = listener.getsockname()[1]
    upstream_received: list[bytes] = []

    def serve_upstream() -> None:
        connection, _ = listener.accept()
        with connection:
            upstream_received.append(connection.recv(4))
            connection.sendall(b"PONG")

    upstream_thread = threading.Thread(target=serve_upstream, daemon=True)
    upstream_thread.start()
    username = "agent"
    password = "test-secret"
    events_path = tmp_path / "events.jsonl"
    server = create_proxy_server(
        condition="control",
        blocked_host="127.0.0.1",
        username=username,
        password=password,
        events_path=events_path,
    )
    proxy_thread = _start_proxy(server)
    try:
        with socket.create_connection(server.server_address, timeout=5) as stream:
            authorization = _proxy_authorization(username, password)
            stream.sendall(
                (
                    f"CONNECT 127.0.0.1:{upstream_port} HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{upstream_port}\r\n"
                    f"Proxy-Authorization: {authorization}\r\n\r\n"
                ).encode()
            )
            response = _receive_until(stream, b"\r\n\r\n")
            assert response.startswith(b"HTTP/1.1 200 Connection Established")
            stream.sendall(b"PING")
            assert stream.recv(4) == b"PONG"
    finally:
        server.shutdown()
        server.server_close()
        listener.close()
        proxy_thread.join(timeout=5)
        upstream_thread.join(timeout=5)

    assert upstream_received == [b"PING"]
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["action"] == "forwarded"
    assert events[0]["status"] == 200


def test_git_client_can_authenticate_and_pass_through_proxy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("proxy test\n", encoding="utf-8")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(source), "add", "README.md"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=HealthAgent test",
            "-c",
            "user.email=healthagent@example.invalid",
            "commit",
            "-m",
            "Initial commit",
        ],
        check=True,
        capture_output=True,
    )
    served = tmp_path / "served"
    served.mkdir()
    bare = served / "repo.git"
    subprocess.run(
        ["git", "clone", "--bare", str(source), str(bare)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", f"--git-dir={bare}", "update-server-info"],
        check=True,
        capture_output=True,
    )

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(served),
    )
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    events_path = tmp_path / "events.jsonl"
    proxy = create_proxy_server(
        condition="control",
        blocked_host="127.0.0.1",
        username="agent",
        password="test-secret",
        events_path=events_path,
    )
    proxy_thread = _start_proxy(proxy)
    proxy_host, proxy_port = proxy.server_address
    upstream_host, upstream_port = upstream.server_address
    proxy_url = f"http://agent:test-secret@{proxy_host}:{proxy_port}"
    environment = os.environ.copy()
    environment.update({"NO_PROXY": "", "no_proxy": ""})
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"http.proxy={proxy_url}",
                "ls-remote",
                f"http://{upstream_host}:{upstream_port}/repo.git",
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()
        proxy_thread.join(timeout=5)
        upstream_thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    assert "HEAD" in result.stdout
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert any(event["action"] == "forwarded" for event in events)
