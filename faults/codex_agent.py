"""Codex adapter for text-only local model backends."""

from __future__ import annotations

import shlex
import socket
import uuid
from pathlib import Path

from harbor.agents.base import AgentContext
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment


TEXT_ONLY_PROXY_SOURCE = Path(__file__).with_name("text_only_proxy.py")


# Reserve an unused loopback port shared by the host and Apptainer network namespace.
def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class TextOnlyCodex(Codex):
    """Run Codex without tools that attach image bytes to model requests."""

    # Route Codex through a local sanitizer because released CLIs ignore tools.view_image.
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        upstream_url = self._get_env("OPENAI_BASE_URL")
        if not upstream_url:
            raise ValueError("TextOnlyCodex requires OPENAI_BASE_URL")

        # Give every trial independent proxy files and a collision-resistant port.
        proxy_token = uuid.uuid4().hex
        proxy_port = _reserve_loopback_port()
        remote_script = f"/tmp/healthagentbench-text-only-{proxy_token}.py"
        remote_ready = f"/tmp/healthagentbench-text-only-{proxy_token}.ready"
        remote_pid = f"/tmp/healthagentbench-text-only-{proxy_token}.pid"
        remote_log = f"/tmp/healthagentbench-text-only-{proxy_token}.log"
        proxy_url = f"http://127.0.0.1:{proxy_port}/v1"

        # Upload the standard-library proxy into the isolated task environment.
        await environment.upload_file(TEXT_ONLY_PROXY_SOURCE, remote_script)
        await self.exec_as_root(
            environment,
            command=f"chmod 0644 {shlex.quote(remote_script)}",
        )

        # Start the sanitizer before Codex resolves its OpenAI-compatible endpoint.
        start_command = (
            f"nohup python3 {shlex.quote(remote_script)} "
            f"--upstream {shlex.quote(upstream_url)} "
            f"--listen-port {proxy_port} "
            f"--ready-file {shlex.quote(remote_ready)} "
            f">{shlex.quote(remote_log)} 2>&1 & "
            f"printf '%s\\n' $! >{shlex.quote(remote_pid)}"
        )
        await self.exec_as_agent(environment, command=start_command)

        # Fail before model inference if the filtering endpoint did not bind successfully.
        ready_command = (
            "for attempt in $(seq 1 100); do "
            f"if [ -s {shlex.quote(remote_ready)} ] && "
            f"curl --fail --silent --max-time 1 "
            f"http://127.0.0.1:{proxy_port}/_text_only_proxy/health >/dev/null; "
            "then exit 0; fi; sleep 0.1; done; "
            f"cat {shlex.quote(remote_log)} >&2; exit 1"
        )
        await self.exec_as_agent(environment, command=ready_command)

        # Override only this agent instance while preserving the caller's upstream URL.
        previous_url = self._extra_env.get("OPENAI_BASE_URL")
        self._extra_env["OPENAI_BASE_URL"] = proxy_url
        try:
            await super().run(instruction, environment, context)
        finally:
            if previous_url is None:
                self._extra_env.pop("OPENAI_BASE_URL", None)
            else:
                self._extra_env["OPENAI_BASE_URL"] = previous_url

            # Stop only this trial's sanitizer and remove its temporary runtime files.
            cleanup_command = (
                f"if [ -s {shlex.quote(remote_pid)} ]; then "
                f"kill $(cat {shlex.quote(remote_pid)}) 2>/dev/null || true; fi; "
                f"rm -f {shlex.quote(remote_script)} {shlex.quote(remote_ready)} "
                f"{shlex.quote(remote_pid)} {shlex.quote(remote_log)}"
            )
            try:
                await self.exec_as_agent(environment, command=cleanup_command)
            except Exception:
                pass
