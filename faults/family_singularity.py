from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from dotenv import dotenv_values
from harbor.environments.singularity import SingularityEnvironment

from .inject import (
    AGENT_RUNTIME_PATH,
    FILE_ACTIONS,
    SUPPORTED_SCENARIOS,
    Condition,
    Dependency,
    PrimaryInput,
    Scenario,
    apply_fault,
    dependency_validation_command,
    file_state,
    select_target,
    write_dependency_runtime,
)


@dataclass(frozen=True)
class ContainerMount:
    source: str
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class FamilySpec:
    name: str
    task_glob: str
    image_name: str
    bootstrap_files: tuple[ContainerMount, ...]
    cache_mounts: tuple[ContainerMount, ...]
    runtime_mounts: tuple[ContainerMount, ...]
    primary_input: PrimaryInput
    dependency: Dependency
    tests_read_only: bool | None = None
    bootstrap_secrets: tuple[tuple[str, tuple[str, ...]], ...] = ()
    main_env_keys: tuple[str, ...] = ()


FAMILY_SPECS = (
    FamilySpec(
        name="clinical_trial_matching",
        task_glob="clinical_trial_matching_task_*",
        image_name="healthagentbench-clinical-trial-matching.sif",
        bootstrap_files=(
            ContainerMount("bootstrap.sh", "/bootstrap.sh", True),
            ContainerMount("fetch_trials.py", "/opt/ctm/fetch_trials.py", True),
            ContainerMount(
                "extract_task_inputs.py", "/opt/ctm/extract_task_inputs.py", True
            ),
        ),
        cache_mounts=(
            ContainerMount(
                "assets/clinical_trial_matching/assets/raw_cache", "/data/_cache"
            ),
        ),
        runtime_mounts=(ContainerMount("workspace", "/workspace"),),
        primary_input=PrimaryInput(
            PurePosixPath("/workspace/data"),
            "exact",
            "/workspace/data/topic.txt",
        ),
        dependency=Dependency("python-module", "remotezip"),
        tests_read_only=False,
    ),
    FamilySpec(
        name="ct_abnormality",
        task_glob="ct_abnormality_valid_*",
        image_name="healthagentbench-ct-abnormality.sif",
        bootstrap_files=(
            ContainerMount("bootstrap.sh", "/bootstrap.sh", True),
            ContainerMount("gold_derivation.py", "/opt/gold_derivation.py", True),
        ),
        cache_mounts=(
            ContainerMount("assets/ct_abnormality/assets/raw_cache", "/data/_cache"),
        ),
        runtime_mounts=(ContainerMount("workspace", "/workspace"),),
        primary_input=PrimaryInput(
            PurePosixPath("/workspace/data"),
            "exact",
            "/workspace/data/scan.nii.gz",
        ),
        dependency=Dependency("python-module", "nibabel"),
        tests_read_only=False,
        bootstrap_secrets=(("HF_TOKEN", ("HF_TOKEN",)),),
    ),
    FamilySpec(
        name="ehr_data_quality",
        task_glob="ehr_data_quality_task_*",
        image_name="healthagentbench-ehr-data-quality.sif",
        bootstrap_files=(
            ContainerMount("bootstrap.sh", "/bootstrap.sh", True),
            ContainerMount("bootstrap_inputs", "/opt/bootstrap_inputs", True),
        ),
        cache_mounts=(
            ContainerMount(
                "assets/ehr_data_quality/assets/raw_cache",
                "/data/_src/raw_cache",
            ),
        ),
        runtime_mounts=(
            ContainerMount("workspace", "/workspace"),
            ContainerMount("workspace/data", "/workspace/data", True),
        ),
        primary_input=PrimaryInput(
            PurePosixPath("/workspace/data"),
            "exact",
            "/workspace/data/csv/labevents.csv.gz",
        ),
        dependency=Dependency("python-module", "pandas"),
        tests_read_only=True,
    ),
    FamilySpec(
        name="ehr_event_modelling",
        task_glob="ehr_event_modelling_new_*",
        image_name="healthagentbench-ehr-event-modelling.sif",
        bootstrap_files=(
            ContainerMount("bootstrap.sh", "/bootstrap.sh", True),
            ContainerMount(
                "stage_data.py", "/opt/ehr_event_modelling/stage_data.py", True
            ),
        ),
        cache_mounts=(
            ContainerMount("assets/ehr_event_modelling/assets", "/data/_cache"),
        ),
        runtime_mounts=(ContainerMount("workspace", "/workspace"),),
        primary_input=PrimaryInput(
            PurePosixPath("/workspace/data"),
            "exact",
            "/workspace/data/events.csv",
        ),
        dependency=Dependency("python-module", "pandas"),
        tests_read_only=False,
        bootstrap_secrets=(("REDIVIS_API_TOKEN", ("REDIVIS_API_TOKEN",)),),
    ),
    FamilySpec(
        name="tumor_area_selection",
        task_glob="tumor_area_selection_pathology_slide_*",
        image_name="healthagentbench-pathology.sif",
        bootstrap_files=(ContainerMount("bootstrap.sh", "/bootstrap.sh", True),),
        cache_mounts=(
            ContainerMount(
                "assets/tumor_area_selection_pathology/assets/raw_cache",
                "/data/_cache",
            ),
        ),
        runtime_mounts=(
            ContainerMount("workspace", "/workspace"),
            ContainerMount("data/slide/current", "/data/slide/current"),
        ),
        primary_input=PrimaryInput(
            PurePosixPath("/data/slide/current"),
            "single-glob",
            "/data/slide/current/slide.*",
        ),
        dependency=Dependency("python-module", "openslide"),
    ),
    FamilySpec(
        name="xray_report_correction",
        task_glob="xray_report_correction_case_*",
        image_name="healthagentbench-xray-report.sif",
        bootstrap_files=(
            ContainerMount("bootstrap.sh", "/bootstrap.sh", True),
            ContainerMount("task_manifest.json", "/opt/task_manifest.json", True),
        ),
        cache_mounts=(
            ContainerMount(
                "assets/xray_report_correction/assets/mimic-cxr-jpg/2.1.0",
                "/data/_src/jpg_root",
            ),
            ContainerMount(
                "assets/xray_report_correction/assets/mimic-cxr/2.1.0/extracted",
                "/data/_src/patient_reports_root",
            ),
        ),
        runtime_mounts=(
            ContainerMount("workspace", "/workspace"),
            ContainerMount("data/patient", "/data/patient", True),
        ),
        primary_input=PrimaryInput(
            PurePosixPath("/data/patient"),
            "latest-glob",
            "/data/patient/study_*/report.txt",
        ),
        dependency=Dependency(
            "executable", "jq", PurePosixPath("/usr/bin/jq")
        ),
        tests_read_only=False,
        bootstrap_secrets=(
            ("PN_USER", ("PN_USER", "PHYSIONET_USER")),
            ("PN_PASS", ("PN_PASS", "PHYSIONET_PASSWORD")),
        ),
        main_env_keys=(
            "CHEXPROMPT_OPENAI_API_KEY",
            "CHEXPROMPT_OPENAI_BASE_URL",
            "CHEXPROMPT_AZURE_OPENAI_API_KEY",
            "CHEXPROMPT_AZURE_OPENAI_ENDPOINT",
            "CHEXPROMPT_AZURE_OPENAI_API_VERSION",
            "CHEXPROMPT_DEPLOYMENT",
        ),
    ),
)


class FamilyRuntimeError(RuntimeError):
    """Raised when a task-family runtime cannot be staged safely."""


def family_for_task(task_name: str) -> FamilySpec:
    matches = [
        spec for spec in FAMILY_SPECS if fnmatch.fnmatchcase(task_name, spec.task_glob)
    ]
    if len(matches) != 1:
        names = ", ".join(spec.name for spec in matches) or "none"
        raise ValueError(
            f"Expected exactly one Apptainer family for {task_name!r}; matched {names}"
        )
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_argument(source: Path, target: str, *, read_only: bool = False) -> str:
    suffix = ":ro" if read_only else ""
    return f"{source}:{target}{suffix}"


def _mount_for_container_path(spec: FamilySpec, container_path: str) -> ContainerMount:
    static_prefix = container_path.split("*", 1)[0]
    matches = [
        mount
        for mount in spec.runtime_mounts
        if static_prefix == mount.target
        or static_prefix.startswith(mount.target.rstrip("/") + "/")
    ]
    if not matches:
        raise FamilyRuntimeError(
            f"Fault path is not inside an Agent-visible runtime mount: {container_path}"
        )
    return max(matches, key=lambda mount: len(mount.target))


def container_path_to_host(
    runtime: Path, spec: FamilySpec, container_path: str
) -> Path:
    mount = _mount_for_container_path(spec, container_path)
    relative = container_path[len(mount.target) :].lstrip("/")
    if ".." in PurePosixPath(relative).parts:
        raise FamilyRuntimeError(f"Unsafe fault path: {container_path}")
    return runtime / mount.source / relative


def host_path_to_container(runtime: Path, spec: FamilySpec, host_path: Path) -> str:
    resolved = host_path.absolute()
    candidates: list[tuple[int, str]] = []
    for mount in spec.runtime_mounts:
        source = (runtime / mount.source).absolute()
        if resolved == source or source in resolved.parents:
            relative = resolved.relative_to(source).as_posix()
            container = mount.target.rstrip("/")
            if relative != ".":
                container = f"{container}/{relative}"
            candidates.append((len(source.parts), container))
    if not candidates:
        raise FamilyRuntimeError(f"Host fault target is not in a runtime mount: {host_path}")
    return max(candidates)[1]


class FamilySingularityEnvironment(SingularityEnvironment):
    """Run a task family through a shared SIF with optional matched faults.

    Bootstrap scripts, caches, tests, answer material, and download credentials
    are absent from the long-lived Agent filesystem. Faults are established
    after bootstrap and before the Agent server starts.
    """

    def __init__(
        self,
        *args: Any,
        family_runtime_root: Path | str | None = None,
        family_keep_runtime: bool = False,
        fault_condition: str | None = None,
        fault_scenario: str | None = None,
        **kwargs: Any,
    ) -> None:
        environment_name = kwargs.get("environment_name")
        environment_dir = kwargs.get("environment_dir")
        if not isinstance(environment_name, str):
            raise ValueError("environment_name is required")
        if not isinstance(environment_dir, Path):
            environment_dir = Path(environment_dir)

        if (fault_condition is None) != (fault_scenario is None):
            raise ValueError(
                "fault_condition and fault_scenario must be supplied together"
            )
        if fault_condition is not None and fault_condition not in {"control", "error"}:
            raise ValueError(f"Unsupported fault condition: {fault_condition}")
        if fault_scenario is not None and fault_scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(f"Unsupported fault scenario: {fault_scenario}")

        self._family = family_for_task(environment_name)
        self._task_dir = environment_dir.resolve().parent
        if self._task_dir.name != environment_name:
            raise ValueError(
                f"Environment name {environment_name!r} does not match task directory "
                f"{self._task_dir.name!r}"
            )
        self._repository_root = self._task_dir.parents[1]
        self._family_runtime_root = (
            Path(family_runtime_root).expanduser().resolve()
            if family_runtime_root is not None
            else None
        )
        self._family_keep_runtime = family_keep_runtime
        self._fault_condition = fault_condition
        self._fault_scenario = fault_scenario
        self._fault_target_host: Path | None = None
        self._fault_target_container: str | None = None
        self._fault_agent_env: str | None = None
        self._family_runtime_dir: Path | None = None
        self._family_mounts_added = False
        super().__init__(*args, **kwargs)

    @property
    def _docker_image(self) -> str:
        return str(self._repository_root / "containers" / self._family.image_name)

    @property
    def family(self) -> FamilySpec:
        return self._family

    def _configuration_values(self) -> dict[str, str]:
        values: dict[str, str] = {}
        env_path = self._repository_root / ".env"
        if env_path.is_file():
            for key, value in dotenv_values(env_path).items():
                if value:
                    values[key] = value
        for key, value in os.environ.items():
            if value:
                values[key] = value
        return values

    def _resolve_bootstrap_environment(self) -> tuple[dict[str, str], list[str]]:
        configured = self._configuration_values()
        environment = os.environ.copy()
        available: list[str] = []
        for target_key, aliases in self._family.bootstrap_secrets:
            value = next((configured[key] for key in aliases if configured.get(key)), None)
            if value is None:
                continue
            environment[f"APPTAINERENV_{target_key}"] = value
            available.append(target_key)
        environment["APPTAINERENV_PYTHONUNBUFFERED"] = "1"
        return environment, available

    def _set_main_environment(self) -> list[str]:
        configured = self._configuration_values()
        available: list[str] = []
        for key in self._family.main_env_keys:
            value = configured.get(key)
            if value:
                self._persistent_env[key] = value
                available.append(key)
        return available

    def _new_runtime_dir(self) -> Path:
        if self._family_runtime_root is not None:
            self._family_runtime_root.mkdir(parents=True, exist_ok=True)
        path = Path(
            tempfile.mkdtemp(
                prefix=f"hab-{self._family.name}-",
                dir=self._family_runtime_root,
            )
        )
        path.chmod(0o700)
        return path

    def _prepare_runtime_layout(self) -> Path:
        runtime = self._new_runtime_dir()
        workspace = runtime / "workspace"
        workspace.mkdir(parents=True)

        workspace_source = self.environment_dir / "workspace"
        if workspace_source.is_dir():
            shutil.copytree(workspace_source, workspace, dirs_exist_ok=True)

        for relative in (
            "workspace/data",
            "workspace/submission",
            "data/slide/current",
            "data/patient",
        ):
            (runtime / relative).mkdir(parents=True, exist_ok=True)
        return runtime

    def _bootstrap_bindings(self, runtime: Path) -> list[str]:
        bindings = [
            _bind_argument(runtime / "workspace", "/workspace"),
            _bind_argument(Path("/etc/resolv.conf"), "/etc/resolv.conf", read_only=True),
        ]

        for mount in self._family.runtime_mounts:
            if mount.target == "/workspace":
                continue
            source = runtime / mount.source
            source.mkdir(parents=True, exist_ok=True)
            bindings.append(_bind_argument(source, mount.target))

        for mount in self._family.bootstrap_files:
            source = self.environment_dir / mount.source
            if not source.exists():
                raise FamilyRuntimeError(f"Missing bootstrap input: {source}")
            bindings.append(_bind_argument(source.resolve(), mount.target, read_only=True))

        for mount in self._family.cache_mounts:
            source = self._repository_root / mount.source
            source.mkdir(parents=True, exist_ok=True)
            bindings.append(_bind_argument(source.resolve(), mount.target))

        if self._family.tests_read_only is not None:
            tests = self._task_dir / "tests"
            if not tests.is_dir():
                raise FamilyRuntimeError(f"Missing task tests directory: {tests}")
            bindings.append(
                _bind_argument(
                    tests.resolve(),
                    "/tests",
                    read_only=self._family.tests_read_only,
                )
            )
        return bindings

    def _apptainer_executable(self) -> str:
        executable = shutil.which("singularity") or shutil.which("apptainer")
        if executable is None:
            raise FamilyRuntimeError("singularity/apptainer is not on PATH")
        return executable

    async def _run_bootstrap(self, runtime: Path) -> tuple[list[str], str]:
        if self._sif_path is None:
            raise FamilyRuntimeError("SIF path is unresolved before bootstrap")

        command = [
            self._apptainer_executable(),
            "exec",
            "--no-mount",
            "home",
            "--no-mount",
            "tmp",
            "--no-mount",
            "bind-paths",
            "--pwd",
            "/workspace",
            "--writable-tmpfs",
            "--fakeroot",
            "--containall",
            "--pid",
        ]
        for binding in self._bootstrap_bindings(runtime):
            command.extend(("-B", binding))
        command.extend((str(self._sif_path), "/bin/bash", "/bootstrap.sh"))

        environment, secret_keys = self._resolve_bootstrap_environment()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        try:
            stdout, _ = await process.communicate()
        except asyncio.CancelledError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            raise

        output = stdout.decode(errors="replace") if stdout else ""
        for line in output.splitlines():
            self.logger.debug("[family-bootstrap] %s", line)
        if process.returncode != 0:
            detail = output.strip() or "<no output>"
            raise FamilyRuntimeError(
                f"Bootstrap failed for {self.environment_name} with "
                f"rc={process.returncode}: {detail}"
            )
        return secret_keys, output

    async def _validate_base_dependency(self) -> None:
        if self._sif_path is None:
            raise FamilyRuntimeError("SIF path is unresolved before validation")
        command = [
            self._apptainer_executable(),
            "exec",
            "--no-mount",
            "home",
            "--no-mount",
            "tmp",
            "--no-mount",
            "bind-paths",
            "--containall",
            str(self._sif_path),
            *dependency_validation_command(self._family.dependency),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            detail = stdout.decode(errors="replace").strip() if stdout else ""
            raise FamilyRuntimeError(
                f"Base dependency {self._family.dependency.name!r} is unavailable "
                f"in {self._family.image_name}: {detail or '<no output>'}"
            )

    def _add_runtime_mounts(self, runtime: Path) -> None:
        if self._family_mounts_added:
            raise FamilyRuntimeError("Family runtime mounts were already added")
        existing_targets = {mount.get("target") for mount in self._mounts}
        for mount in self._family.runtime_mounts:
            if mount.target in existing_targets:
                raise FamilyRuntimeError(
                    f"Runtime mount target conflicts with a Harbor mount: {mount.target}"
                )
            self._mounts.append(
                {
                    "type": "bind",
                    "source": str((runtime / mount.source).resolve()),
                    "target": mount.target,
                    **({"read_only": True} if mount.read_only else {}),
                }
            )
            existing_targets.add(mount.target)
        self._family_mounts_added = True

    def _add_dependency_mount(self, runtime: Path) -> None:
        if any(mount.get("target") == AGENT_RUNTIME_PATH for mount in self._mounts):
            raise FamilyRuntimeError(
                f"Fault runtime mount conflicts with Harbor mount: {AGENT_RUNTIME_PATH}"
            )
        self._mounts.append(
            {
                "type": "bind",
                "source": str(runtime.resolve()),
                "target": AGENT_RUNTIME_PATH,
                "read_only": True,
            }
        )

    def _write_json(self, name: str, value: dict[str, object]) -> None:
        path = self.trial_paths.trial_dir / name
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    async def _prepare_fault(self, runtime: Path) -> None:
        if self._fault_condition is None or self._fault_scenario is None:
            return

        condition: Condition = self._fault_condition  # type: ignore[assignment]
        scenario: Scenario = self._fault_scenario  # type: ignore[assignment]
        setup: dict[str, object] = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "task": self.environment_name,
            "family": self._family.name,
            "condition": condition,
            "scenario": scenario,
            "injected_after_bootstrap": True,
            "injected_before_agent": True,
            "sif_modified": False,
        }

        if scenario in FILE_ACTIONS:
            profile = self._family.primary_input
            allowed_root = container_path_to_host(
                runtime, self._family, str(profile.allowed_root)
            )
            expression = container_path_to_host(runtime, self._family, profile.path)
            target = select_target(
                selector=profile.selector,
                target_expression=str(expression),
                allowed_root=allowed_root,
            )
            self._fault_target_host = target
            self._fault_target_container = host_path_to_container(
                runtime, self._family, target
            )
            before = file_state(target)
            action = FILE_ACTIONS[scenario]
            apply_fault(action=action, condition=condition, target=target)
            after = file_state(target)
            setup["primary_input"] = {
                "allowed_root": str(profile.allowed_root),
                "selector": profile.selector,
                "expression": profile.path,
                "selected_target": self._fault_target_container,
                "action": action,
                "before": before,
                "after": after,
            }
        else:
            await self._validate_base_dependency()
            dependency_runtime, agent_env = write_dependency_runtime(
                runtime / "fault", self._family.dependency, condition
            )
            self._fault_agent_env = agent_env
            self._add_dependency_mount(dependency_runtime)
            setup["dependency"] = {
                "kind": self._family.dependency.kind,
                "name": self._family.dependency.name,
                "real_path": (
                    str(self._family.dependency.real_path)
                    if self._family.dependency.real_path is not None
                    else None
                ),
                "base_image_validation": "present",
                "agent_env_key": agent_env.split("=", 1)[0],
                "agent_scoped": True,
            }
        self._write_json("fault-setup.json", setup)

    def _write_runtime_manifest(
        self,
        *,
        bootstrap_secret_keys: list[str],
        main_env_keys: list[str],
    ) -> None:
        if self._sif_path is None:
            raise FamilyRuntimeError("Cannot write runtime manifest without a SIF")
        image = self._sif_path.resolve()
        manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "bootstrap-complete",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "task": self.environment_name,
            "family": self._family.name,
            "image": str(image),
            "image_size": image.stat().st_size,
            "image_sha256": _sha256(image),
            "bootstrap_secret_keys": sorted(bootstrap_secret_keys),
            "main_env_keys": sorted(main_env_keys),
            "agent_visible_mounts": [
                {"target": mount.target, "read_only": mount.read_only}
                for mount in self._family.runtime_mounts
            ],
            "agent_excluded": [
                "bootstrap scripts",
                "dataset caches",
                "task tests and gold",
                "bootstrap download credentials",
            ],
            "fault": (
                {
                    "condition": self._fault_condition,
                    "scenario": self._fault_scenario,
                }
                if self._fault_condition is not None
                else None
            ),
        }
        self._write_json("family-runtime.json", manifest)

    async def _start_server_with_read_only_mounts(self) -> None:
        original_mounts = self._mounts
        command_mounts = []
        for mount in original_mounts:
            command_mount = dict(mount)
            if command_mount.get("type") == "bind" and command_mount.get(
                "read_only"
            ):
                command_mount["target"] = f"{command_mount['target']}:ro"
            command_mounts.append(command_mount)
        self._mounts = command_mounts
        try:
            await super()._start_server()
        finally:
            self._mounts = original_mounts

    def _collect_final_fault_state(self) -> None:
        if self._fault_condition is None or self._fault_scenario is None:
            return
        final: dict[str, object] = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "task": self.environment_name,
            "family": self._family.name,
            "condition": self._fault_condition,
            "scenario": self._fault_scenario,
        }
        if self._fault_target_host is not None:
            final["primary_input"] = {
                "selected_target": self._fault_target_container,
                "final": file_state(self._fault_target_host),
            }
        if self._fault_agent_env is not None:
            events_path = self.trial_paths.artifacts_dir / "fault-events.jsonl"
            event_count = 0
            if events_path.is_file():
                event_count = sum(
                    1 for line in events_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            final["dependency"] = {
                "name": self._family.dependency.name,
                "agent_env_key": self._fault_agent_env.split("=", 1)[0],
                "exposed": event_count > 0,
                "event_count": event_count,
            }
        self._write_json("fault-final-state.json", final)

    def _cleanup_family_runtime(self) -> None:
        runtime = self._family_runtime_dir
        if runtime is None or self._family_keep_runtime:
            return
        if runtime.exists():
            shutil.rmtree(runtime)
        self._family_runtime_dir = None

    async def start(self, force_build: bool) -> None:
        if sys.platform == "win32":
            raise RuntimeError("FamilySingularityEnvironment requires Linux")
        self._sif_path = Path(self._docker_image)
        if not self._sif_path.is_file():
            raise FileNotFoundError(
                f"Family SIF is missing: {self._sif_path}. Run "
                "containers/build_family_sifs.sh first."
            )

        runtime = self._prepare_runtime_layout()
        self._family_runtime_dir = runtime
        try:
            bootstrap_secret_keys, _ = await self._run_bootstrap(runtime)
            main_env_keys = self._set_main_environment()
            self._add_runtime_mounts(runtime)
            await self._prepare_fault(runtime)
            self._write_runtime_manifest(
                bootstrap_secret_keys=bootstrap_secret_keys,
                main_env_keys=main_env_keys,
            )
            await self._start_server_with_read_only_mounts()
        except Exception:
            self._cleanup_family_runtime()
            raise

    async def stop(self, delete: bool) -> None:
        try:
            try:
                self._collect_final_fault_state()
            except Exception:
                self.logger.exception("Failed to collect final family fault state")
            await super().stop(delete)
        finally:
            self._cleanup_family_runtime()
