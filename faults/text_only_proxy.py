"""Remove image capabilities from Codex requests sent to text-only models."""

from __future__ import annotations

import argparse
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit


DISABLED_TOOL_NAMES = frozenset({"view_image"})
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
IMAGE_PLACEHOLDER = "[Image omitted by the text-only model adapter.]"


# Resolve function-style and direct tool names used by Responses API clients.
def tool_name(tool: dict[str, Any]) -> str | None:
    direct_name = tool.get("name")
    if isinstance(direct_name, str):
        return direct_name
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


# Replace residual image blocks without retaining their encoded pixel payloads.
def strip_image_blocks(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("type") == "input_image":
            return {"type": "input_text", "text": IMAGE_PLACEHOLDER}
        return {key: strip_image_blocks(child) for key, child in value.items()}
    if isinstance(value, list):
        return [strip_image_blocks(child) for child in value]
    return value


# Remove the image tool and any completed image-tool exchange from one request.
def sanitize_request_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    tools = payload.get("tools")
    if isinstance(tools, list):
        payload["tools"] = [
            tool
            for tool in tools
            if not isinstance(tool, dict) or tool_name(tool) not in DISABLED_TOOL_NAMES
        ]

    input_items = payload.get("input")
    if isinstance(input_items, list):
        disabled_call_ids = {
            item.get("call_id")
            for item in input_items
            if isinstance(item, dict)
            and item.get("type") in {"function_call", "custom_tool_call"}
            and item.get("name") in DISABLED_TOOL_NAMES
            and isinstance(item.get("call_id"), str)
        }
        payload["input"] = [
            strip_image_blocks(item)
            for item in input_items
            if not (
                isinstance(item, dict)
                and (
                    item.get("name") in DISABLED_TOOL_NAMES
                    or item.get("call_id") in disabled_call_ids
                )
            )
        ]

    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_name(tool_choice) in DISABLED_TOOL_NAMES:
        payload["tool_choice"] = "auto"
    return strip_image_blocks(payload)


# Encode a sanitized JSON body while passing non-JSON traffic through unchanged.
def sanitize_request_body(body: bytes) -> bytes:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    sanitized = sanitize_request_payload(payload)
    return json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")).encode()


class TextOnlyProxyServer(ThreadingHTTPServer):
    """HTTP server with immutable upstream configuration."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], upstream: SplitResult) -> None:
        self.upstream = upstream
        super().__init__(address, TextOnlyProxyHandler)


class TextOnlyProxyHandler(BaseHTTPRequestHandler):
    """Sanitize Responses requests and relay all HTTP response bytes."""

    server: TextOnlyProxyServer
    protocol_version = "HTTP/1.1"

    # Keep routine access logs out of experiment output.
    def log_message(self, format: str, *args: Any) -> None:
        return

    # Expose a local readiness endpoint without contacting the model server.
    def _serve_health(self) -> None:
        body = b'{"status":"ok"}\n'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Force Codex to use HTTP because this small proxy does not relay WebSockets.
    def _reject_websocket(self) -> None:
        body = b"WebSocket transport is not supported\n"
        self.send_response(426)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    # Preserve the upstream base path without duplicating an existing /v1 prefix.
    def _upstream_target(self) -> str:
        base_path = self.server.upstream.path.rstrip("/")
        if not base_path or self.path == base_path or self.path.startswith(f"{base_path}/"):
            return self.path
        return f"{base_path}{self.path}"

    # Forward one sanitized request and return the complete upstream response.
    def _proxy_request(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b""
        if self.command == "POST" and self.path.rstrip("/").endswith("/responses"):
            body = sanitize_request_body(body)

        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP_HEADERS
            and name.lower() not in {"content-length", "host"}
        }
        headers["Content-Length"] = str(len(body))

        connection_class = (
            http.client.HTTPSConnection
            if self.server.upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_class(
            self.server.upstream.hostname,
            self.server.upstream.port,
            timeout=3600,
        )
        try:
            connection.request(
                self.command,
                self._upstream_target(),
                body=body,
                headers=headers,
            )
            response = connection.getresponse()
            response_body = response.read()
            self.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                if name.lower() not in HOP_BY_HOP_HEADERS and name.lower() != "content-length":
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        finally:
            connection.close()

    # Route supported HTTP methods through the same sanitizing relay.
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/_text_only_proxy/health":
            self._serve_health()
        elif self.headers.get("Upgrade", "").lower() == "websocket":
            self._reject_websocket()
        else:
            self._proxy_request()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_PATCH(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._proxy_request()


# Parse the small runtime contract used by TextOnlyCodex.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    return parser


# Bind before publishing readiness so Codex never races proxy startup.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    upstream = urlsplit(args.upstream)
    if upstream.scheme not in {"http", "https"} or not upstream.hostname:
        raise ValueError(f"Unsupported upstream URL: {args.upstream}")
    server = TextOnlyProxyServer(("127.0.0.1", args.listen_port), upstream)
    args.ready_file.write_text(
        json.dumps({"listen_port": server.server_port}) + "\n",
        encoding="utf-8",
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
