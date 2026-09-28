from __future__ import annotations

import argparse
import base64
import hmac
import json
import select
import socket
import socketserver
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

MAX_HEADER_BYTES = 64 * 1024
BUFFER_SIZE = 64 * 1024


@dataclass(frozen=True)
class ProxyConfig:
    condition: str
    blocked_host: str
    username: str
    password: str
    events_path: Path

    @property
    def expected_authorization(self) -> str:
        token = base64.b64encode(
            f"{self.username}:{self.password}".encode("utf-8")
        ).decode("ascii")
        return f"Basic {token}"


class SelectiveProxyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        config: ProxyConfig,
    ) -> None:
        self.proxy_config = config
        self._event_lock = threading.Lock()
        super().__init__(server_address, SelectiveProxyHandler)

    def record_event(self, *, host: str, port: int, action: str, status: int) -> None:
        event = {
            "event": "recovery_http_request",
            "host": host,
            "port": port,
            "action": action,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.proxy_config.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self._event_lock:
            with self.proxy_config.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")


class SelectiveProxyHandler(socketserver.BaseRequestHandler):
    server: SelectiveProxyServer

    def handle(self) -> None:
        self.request.settimeout(15)
        try:
            raw_headers = self._read_headers()
            if not raw_headers:
                return
            request_line, headers = self._parse_headers(raw_headers)
            if not self._authorized(headers):
                self._respond(
                    407,
                    "Proxy Authentication Required",
                    headers=(
                        ("Proxy-Authenticate", 'Basic realm="healthagent-fault"'),
                    ),
                )
                return

            method, target, version = request_line.split(" ", 2)
            if method.upper() == "CONNECT":
                self._handle_connect(target)
            else:
                self._handle_plain_http(method, target, version, headers)
        except (OSError, ValueError):
            try:
                self._respond(400, "Bad Request")
            except OSError:
                pass

    def _read_headers(self) -> bytes:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = self.request.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_HEADER_BYTES:
                raise ValueError("Proxy request headers are too large")
        return bytes(data)

    @staticmethod
    def _parse_headers(raw: bytes) -> tuple[str, list[tuple[str, str]]]:
        head = raw.split(b"\r\n\r\n", 1)[0].decode("iso-8859-1")
        lines = head.split("\r\n")
        if not lines or len(lines[0].split(" ", 2)) != 3:
            raise ValueError("Malformed proxy request line")
        headers: list[tuple[str, str]] = []
        for line in lines[1:]:
            if not line:
                continue
            name, separator, value = line.partition(":")
            if not separator:
                raise ValueError("Malformed proxy header")
            headers.append((name.strip(), value.strip()))
        return lines[0], headers

    def _authorized(self, headers: list[tuple[str, str]]) -> bool:
        provided = next(
            (value for name, value in headers if name.lower() == "proxy-authorization"),
            "",
        )
        return hmac.compare_digest(
            provided, self.server.proxy_config.expected_authorization
        )

    def _is_blocked(self, host: str) -> bool:
        config = self.server.proxy_config
        normalized = host.rstrip(".").lower()
        blocked = config.blocked_host.rstrip(".").lower()
        return config.condition == "error" and (
            normalized == blocked or normalized.endswith(f".{blocked}")
        )

    @staticmethod
    def _host_and_port(target: str, default_port: int) -> tuple[str, int]:
        if target.startswith("["):
            host, separator, raw_port = target[1:].partition("]:")
            if not separator:
                return target.strip("[]"), default_port
            return host, int(raw_port)
        if ":" not in target:
            return target, default_port
        host, raw_port = target.rsplit(":", 1)
        return host, int(raw_port)

    def _handle_connect(self, target: str) -> None:
        host, port = self._host_and_port(target, 443)
        if self._is_blocked(host):
            self.server.record_event(host=host, port=port, action="blocked", status=403)
            self._respond(403, "Forbidden")
            return

        try:
            upstream = socket.create_connection((host, port), timeout=15)
        except OSError:
            if self._is_tracked_host(host):
                self.server.record_event(
                    host=host, port=port, action="upstream-error", status=502
                )
            self._respond(502, "Bad Gateway")
            return

        if self._is_tracked_host(host):
            self.server.record_event(
                host=host, port=port, action="forwarded", status=200
            )
        try:
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self._relay(self.request, upstream)
        finally:
            upstream.close()

    def _handle_plain_http(
        self,
        method: str,
        target: str,
        version: str,
        headers: list[tuple[str, str]],
    ) -> None:
        parsed = urlsplit(target)
        if not parsed.hostname:
            self._respond(400, "Absolute URL Required")
            return
        host = parsed.hostname
        port = parsed.port or 80
        if self._is_blocked(host):
            self.server.record_event(host=host, port=port, action="blocked", status=403)
            self._respond(403, "Forbidden")
            return

        path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        forwarded_headers = [
            (name, value)
            for name, value in headers
            if name.lower() not in {"proxy-authorization", "proxy-connection"}
        ]
        request = [f"{method} {path} {version}"]
        request.extend(f"{name}: {value}" for name, value in forwarded_headers)
        payload = ("\r\n".join(request) + "\r\n\r\n").encode("iso-8859-1")

        try:
            upstream = socket.create_connection((host, port), timeout=15)
        except OSError:
            self._respond(502, "Bad Gateway")
            return
        if self._is_tracked_host(host):
            self.server.record_event(
                host=host, port=port, action="forwarded", status=200
            )
        try:
            upstream.sendall(payload)
            self._relay(self.request, upstream)
        finally:
            upstream.close()

    def _is_tracked_host(self, host: str) -> bool:
        normalized = host.rstrip(".").lower()
        tracked = self.server.proxy_config.blocked_host.rstrip(".").lower()
        return normalized == tracked or normalized.endswith(f".{tracked}")

    @staticmethod
    def _relay(client: socket.socket, upstream: socket.socket) -> None:
        sockets = [client, upstream]
        while sockets:
            readable, _, exceptional = select.select(sockets, [], sockets, 30)
            if exceptional or not readable:
                return
            for source in readable:
                destination = upstream if source is client else client
                data = source.recv(BUFFER_SIZE)
                if not data:
                    return
                destination.sendall(data)

    def _respond(
        self,
        status: int,
        reason: str,
        *,
        headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        response_headers = [
            f"HTTP/1.1 {status} {reason}",
            *(f"{name}: {value}" for name, value in headers),
            "Content-Length: 0",
            "Connection: close",
            "",
            "",
        ]
        response = "\r\n".join(response_headers)
        self.request.sendall(response.encode("ascii"))


def create_proxy_server(
    *,
    condition: str,
    blocked_host: str,
    username: str,
    password: str,
    events_path: Path,
    port: int = 0,
) -> SelectiveProxyServer:
    if condition not in {"control", "error"}:
        raise ValueError(f"Unsupported proxy condition: {condition!r}")
    config = ProxyConfig(
        condition=condition,
        blocked_host=blocked_host,
        username=username,
        password=password,
        events_path=events_path,
    )
    return SelectiveProxyServer(("127.0.0.1", port), config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("control", "error"), required=True)
    parser.add_argument("--blocked-host", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--events-path", type=Path, required=True)
    parser.add_argument("--ready-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = create_proxy_server(
        condition=args.condition,
        blocked_host=args.blocked_host,
        username=args.username,
        password=args.password,
        events_path=args.events_path,
    )
    host, port = server.server_address
    args.ready_path.parent.mkdir(parents=True, exist_ok=True)
    args.ready_path.write_text(
        json.dumps({"host": host, "port": port, "status": "ready"}) + "\n",
        encoding="utf-8",
    )
    args.ready_path.chmod(0o600)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
