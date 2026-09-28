"""Capture and forward OpenAI-compatible HTTP calls to a local vLLM server."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit

from faults.telemetry import SCHEMA_VERSION

# Keep transport-only headers out of the forwarded request and response.
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

# Redact credentials while preserving header names for reproducibility audits.
SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "openai-api-key",
        "proxy-authorization",
        "set-cookie",
        "x-api-key",
    }
)

# Bound individual request bodies to avoid accidental disk exhaustion.
DEFAULT_MAX_REQUEST_BYTES = 256 * 1024 * 1024
STREAM_CHUNK_BYTES = 64 * 1024


# Generate timezone-aware timestamps for every persisted event.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Produce stable content hashes without retaining secret header values.
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# Convert HTTP headers to a JSON-safe list while preserving duplicates and order.
def sanitize_headers(headers: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for name, value in headers:
        recorded_value = "<redacted>" if name.lower() in SENSITIVE_HEADERS else value
        sanitized.append({"name": name, "value": recorded_value})
    return sanitized


# Write JSON through a sibling temporary file so completed artifacts are atomic.
def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


# Decode JSON only when the payload is valid UTF-8 JSON.
def decode_json(payload: bytes) -> Any | None:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


# Extract JSON payloads from an SSE byte stream while retaining malformed events.
def parse_sse_events(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8", errors="replace")
    events: list[dict[str, Any]] = []
    for event_index, block in enumerate(text.replace("\r\n", "\n").split("\n\n")):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
        if not data_lines:
            continue
        raw_data = "\n".join(data_lines)
        parsed_data: Any = raw_data
        if raw_data != "[DONE]":
            try:
                parsed_data = json.loads(raw_data)
            except json.JSONDecodeError:
                parsed_data = raw_data
        events.append(
            {
                "event_index": event_index,
                "event": event_name,
                "data": parsed_data,
            }
        )
    return events


# Recursively collect response identifiers from regular JSON and SSE payloads.
def collect_response_ids(payload: Any) -> list[str]:
    response_ids: set[str] = set()

    # Walk nested response objects because vLLM may wrap the completed response.
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            candidate = value.get("id")
            object_name = value.get("object")
            event_type = value.get("type")
            if isinstance(candidate, str) and (
                candidate.startswith("resp_")
                or object_name == "response"
                or event_type in {"response.created", "response.completed"}
            ):
                response_ids.add(candidate)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    # Return a deterministic ordering for manifests and downstream joins.
    visit(payload)
    return sorted(response_ids)


@dataclass(frozen=True)
class ProxyConfig:
    """Immutable runtime configuration shared by all request handlers."""

    listen_host: str
    listen_port: int
    upstream: SplitResult
    output_dir: Path
    max_request_bytes: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CallContext:
    """Paths and timing information for one captured HTTP call."""

    sequence: int
    call_id: str
    call_dir: Path
    started_at: str
    started_monotonic: float


class CaptureRecorder:
    """Persist concurrent request records in independent, append-only call folders."""

    def __init__(self, config: ProxyConfig) -> None:
        self.config = config
        self.calls_dir = config.output_dir / "calls"
        self.index_path = config.output_dir / "calls.jsonl"
        self._lock = threading.Lock()
        self._next_sequence = self._discover_next_sequence()

        # Create the artifact root before accepting network traffic.
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            config.output_dir / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "upstream_url": urlunsplit(config.upstream),
                "listen_host": config.listen_host,
                "listen_port": config.listen_port,
                "metadata": config.metadata,
            },
        )

    # Continue sequence numbers when a stopped capture directory is reused.
    def _discover_next_sequence(self) -> int:
        calls_dir = self.config.output_dir / "calls"
        if not calls_dir.exists():
            return 1
        sequences: list[int] = []
        for child in calls_dir.iterdir():
            prefix = child.name.split("-", 1)[0]
            if prefix.isdigit():
                sequences.append(int(prefix))
        return max(sequences, default=0) + 1

    # Allocate a collision-resistant directory before contacting the upstream model.
    def begin_call(
        self,
        *,
        method: str,
        target: str,
        headers: Sequence[tuple[str, str]],
        body: bytes,
    ) -> CallContext:
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
        call_id = f"call_{uuid.uuid4().hex}"
        call_dir = self.calls_dir / f"{sequence:06d}-{call_id}"
        call_dir.mkdir(parents=False, exist_ok=False)
        started_at = utc_now()

        # Preserve exact request bytes and a readable JSON copy when possible.
        (call_dir / "request.body").write_bytes(body)
        parsed_body = decode_json(body)
        if parsed_body is not None:
            atomic_write_json(call_dir / "request-body.json", {"body": parsed_body})
        atomic_write_json(
            call_dir / "request.json",
            {
                "schema_version": SCHEMA_VERSION,
                "sequence": sequence,
                "call_id": call_id,
                "started_at": started_at,
                "method": method,
                "target": target,
                "headers": sanitize_headers(headers),
                "body_bytes": len(body),
                "body_sha256": sha256_bytes(body),
                "metadata": self.config.metadata,
            },
        )
        return CallContext(
            sequence=sequence,
            call_id=call_id,
            call_dir=call_dir,
            started_at=started_at,
            started_monotonic=time.monotonic(),
        )

    # Finalize one call and append its compact summary to the global index.
    def finish_call(self, context: CallContext, record: dict[str, Any]) -> None:
        completed_at = utc_now()
        duration_seconds = time.monotonic() - context.started_monotonic
        complete_record = {
            "schema_version": SCHEMA_VERSION,
            "sequence": context.sequence,
            "call_id": context.call_id,
            "started_at": context.started_at,
            "completed_at": completed_at,
            "duration_seconds": duration_seconds,
            "metadata": self.config.metadata,
            **record,
        }
        atomic_write_json(context.call_dir / "call.json", complete_record)

        # Serialize index writes so concurrent rollouts never interleave JSON lines.
        with self._lock:
            with self.index_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(complete_record, ensure_ascii=False) + "\n")


class TelemetryHTTPServer(ThreadingHTTPServer):
    """Threaded reverse proxy with a shared capture recorder."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: ProxyConfig) -> None:
        self.proxy_config = config
        self.recorder = CaptureRecorder(config)
        super().__init__((config.listen_host, config.listen_port), TelemetryHandler)


class TelemetryHandler(BaseHTTPRequestHandler):
    """Forward one HTTP request while recording exact request and response bodies."""

    server: TelemetryHTTPServer
    protocol_version = "HTTP/1.1"

    # Route every OpenAI-compatible HTTP method through the same capture path.
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/_telemetry/health":
            self._serve_health()
            return
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self._reject_websocket()
            return
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

    # Keep routine access logs concise because full request data lives on disk.
    def log_message(self, format_string: str, *args: object) -> None:
        sys.stderr.write(
            f"[telemetry-proxy] {self.address_string()} {format_string % args}\n"
        )

    # Expose a local readiness endpoint that never reaches vLLM or the capture index.
    def _serve_health(self) -> None:
        payload = json.dumps(
            {"status": "ready", "schema_version": SCHEMA_VERSION}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    # Record unsupported WebSocket attempts and force Codex onto inspectable HTTP.
    def _reject_websocket(self) -> None:
        headers = list(self.headers.items())
        context = self.server.recorder.begin_call(
            method=self.command,
            target=self.path,
            headers=headers,
            body=b"",
        )
        payload = b"WebSocket capture is disabled; use HTTP Responses streaming.\n"
        self.send_response(426, "Upgrade Required")
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        (context.call_dir / "response.body").write_bytes(payload)
        self.server.recorder.finish_call(
            context,
            {
                "method": self.command,
                "target": self.path,
                "status": 426,
                "outcome": "websocket-rejected",
                "response_bytes": len(payload),
                "response_sha256": sha256_bytes(payload),
                "response_ids": [],
            },
        )

    # Read fixed-length or chunked request bodies with an explicit safety limit.
    def _read_request_body(self) -> bytes:
        transfer_encoding = self.headers.get("Transfer-Encoding", "").lower()
        if "chunked" in transfer_encoding:
            return self._read_chunked_body()
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        content_length = int(raw_length)
        if (
            content_length < 0
            or content_length > self.server.proxy_config.max_request_bytes
        ):
            raise ValueError("request body exceeds the configured size limit")
        body = self.rfile.read(content_length)
        if len(body) != content_length:
            raise ValueError("request body ended before Content-Length bytes were read")
        return body

    # Decode HTTP/1.1 chunk framing while preserving the exact semantic body bytes.
    def _read_chunked_body(self) -> bytes:
        body = bytearray()
        while True:
            size_line = self.rfile.readline(128)
            if not size_line:
                raise ValueError("unexpected EOF in chunked request")
            chunk_size = int(size_line.split(b";", 1)[0].strip(), 16)
            if chunk_size == 0:
                while self.rfile.readline(8192) not in {b"\r\n", b"\n", b""}:
                    pass
                break
            if len(body) + chunk_size > self.server.proxy_config.max_request_bytes:
                raise ValueError("request body exceeds the configured size limit")
            body.extend(self.rfile.read(chunk_size))
            if self.rfile.read(2) != b"\r\n":
                raise ValueError("malformed chunked request")
        return bytes(body)

    # Build the upstream path while retaining the request's query string.
    def _upstream_target(self) -> str:
        upstream = self.server.proxy_config.upstream
        request = urlsplit(self.path)
        base_path = upstream.path.rstrip("/")
        request_path = (
            request.path if request.path.startswith("/") else f"/{request.path}"
        )
        combined_path = f"{base_path}{request_path}" or "/"
        return urlunsplit(("", "", combined_path, request.query, ""))

    # Remove hop-by-hop fields and set a deterministic upstream Host header.
    def _forward_headers(self, body: bytes) -> dict[str, str]:
        upstream = self.server.proxy_config.upstream
        forwarded: dict[str, str] = {}
        for name, value in self.headers.items():
            if name.lower() in HOP_BY_HOP_HEADERS | {"content-length", "host"}:
                continue
            forwarded[name] = value
        forwarded["Host"] = upstream.netloc
        forwarded["Content-Length"] = str(len(body))
        forwarded["X-Telemetry-Proxy"] = SCHEMA_VERSION
        return forwarded

    # Stream the upstream response to Codex and to the immutable capture artifact.
    def _proxy_request(self) -> None:
        context: CallContext | None = None
        connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = (
            None
        )
        try:
            body = self._read_request_body()
            request_headers = list(self.headers.items())
            context = self.server.recorder.begin_call(
                method=self.command,
                target=self.path,
                headers=request_headers,
                body=body,
            )
            upstream = self.server.proxy_config.upstream
            connection_class = (
                http.client.HTTPSConnection
                if upstream.scheme == "https"
                else http.client.HTTPConnection
            )
            connection = connection_class(upstream.hostname, upstream.port, timeout=600)
            forwarded_headers = self._forward_headers(body)
            atomic_write_json(
                context.call_dir / "request-forwarded.json",
                {
                    "method": self.command,
                    "upstream_target": self._upstream_target(),
                    "headers": sanitize_headers(list(forwarded_headers.items())),
                    "body_bytes": len(body),
                    "body_sha256": sha256_bytes(body),
                },
            )
            connection.request(
                self.command,
                self._upstream_target(),
                body=body,
                headers=forwarded_headers,
            )
            response = connection.getresponse()
            response_headers = response.getheaders()

            # Use chunked transfer to preserve streaming behavior for SSE responses.
            self.send_response(response.status, response.reason)
            for name, value in response_headers:
                if name.lower() in HOP_BY_HOP_HEADERS | {"content-length"}:
                    continue
                self.send_header(name, value)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()

            # Record raw response bytes and per-chunk latency metadata incrementally.
            response_hash = hashlib.sha256()
            response_bytes = 0
            chunk_index = 0
            client_connected = True
            chunks_path = context.call_dir / "response-chunks.jsonl"
            response_path = context.call_dir / "response.body"
            with (
                response_path.open("wb") as response_stream,
                chunks_path.open("w", encoding="utf-8") as chunk_stream,
            ):
                # read1 returns currently buffered bytes instead of waiting for 64 KiB or EOF.
                read_available = getattr(response, "read1", response.read)
                while True:
                    chunk = read_available(STREAM_CHUNK_BYTES)
                    if not chunk:
                        break
                    received_offset = time.monotonic() - context.started_monotonic
                    response_stream.write(chunk)
                    response_hash.update(chunk)
                    response_bytes += len(chunk)
                    chunk_stream.write(
                        json.dumps(
                            {
                                "chunk_index": chunk_index,
                                "offset_bytes": response_bytes - len(chunk),
                                "length_bytes": len(chunk),
                                "received_after_seconds": received_offset,
                                "sha256": sha256_bytes(chunk),
                            }
                        )
                        + "\n"
                    )
                    chunk_stream.flush()
                    if client_connected:
                        try:
                            self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                            self.wfile.write(chunk)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            client_connected = False
                    chunk_index += 1
                if client_connected:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()

            # Parse convenience artifacts only after the exact raw stream is durable.
            response_payload = response_path.read_bytes()
            content_type = next(
                (
                    value
                    for name, value in response_headers
                    if name.lower() == "content-type"
                ),
                "",
            )
            parsed_payload: Any = decode_json(response_payload)
            if "text/event-stream" in content_type.lower():
                parsed_payload = parse_sse_events(response_payload)
                with (context.call_dir / "response-events.jsonl").open(
                    "w", encoding="utf-8"
                ) as event_stream:
                    for event in parsed_payload:
                        event_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            elif parsed_payload is not None:
                atomic_write_json(
                    context.call_dir / "response-body.json",
                    {"body": parsed_payload},
                )
            response_ids = collect_response_ids(parsed_payload)
            atomic_write_json(
                context.call_dir / "response.json",
                {
                    "status": response.status,
                    "reason": response.reason,
                    "headers": sanitize_headers(response_headers),
                    "body_bytes": response_bytes,
                    "body_sha256": response_hash.hexdigest(),
                    "chunk_count": chunk_index,
                    "response_ids": response_ids,
                    "client_connected_through_completion": client_connected,
                },
            )
            self.server.recorder.finish_call(
                context,
                {
                    "method": self.command,
                    "target": self.path,
                    "upstream_target": self._upstream_target(),
                    "status": response.status,
                    "outcome": "completed",
                    "response_bytes": response_bytes,
                    "response_sha256": response_hash.hexdigest(),
                    "response_ids": response_ids,
                    "chunk_count": chunk_index,
                    "client_connected_through_completion": client_connected,
                },
            )
        except (OSError, ValueError, http.client.HTTPException) as error:
            self._handle_proxy_error(context, error)
        finally:
            if connection is not None:
                connection.close()

    # Return a controlled gateway error and preserve the failure as an experiment event.
    def _handle_proxy_error(
        self,
        context: CallContext | None,
        error: Exception,
    ) -> None:
        payload = json.dumps(
            {
                "error": {
                    "type": "telemetry_proxy_error",
                    "message": str(error),
                }
            }
        ).encode("utf-8")
        try:
            self.send_response(502, "Bad Gateway")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        if context is not None:
            (context.call_dir / "response.body").write_bytes(payload)
            atomic_write_json(
                context.call_dir / "error.json",
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            self.server.recorder.finish_call(
                context,
                {
                    "method": self.command,
                    "target": self.path,
                    "status": 502,
                    "outcome": "proxy-error",
                    "response_bytes": len(payload),
                    "response_sha256": sha256_bytes(payload),
                    "response_ids": [],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )


# Validate and normalize the upstream origin supplied on the command line.
def parse_upstream(value: str) -> SplitResult:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise argparse.ArgumentTypeError("upstream URL must use http or https")
    if parsed.hostname is None:
        raise argparse.ArgumentTypeError("upstream URL must include a hostname")
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError(
            "upstream URL must not contain query or fragment"
        )
    return parsed


# Parse repeatable KEY=VALUE metadata without executing shell-like syntax.
def parse_metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        key, separator, item_value = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"metadata must use KEY=VALUE syntax: {value!r}")
        metadata[key.strip()] = item_value
    return metadata


# Define an explicit CLI so all experimental settings can be captured verbatim.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture OpenAI-compatible calls and forward them to vLLM."
    )
    parser.add_argument("--upstream", type=parse_upstream, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8100)
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=DEFAULT_MAX_REQUEST_BYTES,
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--ready-file", type=Path)
    return parser


# Run the threaded proxy until Slurm, Ctrl-C, or SIGTERM requests shutdown.
def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.listen_port <= 65535:
        raise SystemExit("--listen-port must be between 1 and 65535")
    if args.max_request_bytes <= 0:
        raise SystemExit("--max-request-bytes must be positive")
    try:
        metadata = parse_metadata(args.metadata)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    config = ProxyConfig(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream=args.upstream,
        output_dir=args.output_dir.resolve(),
        max_request_bytes=args.max_request_bytes,
        metadata=metadata,
    )
    server = TelemetryHTTPServer(config)

    # Publish readiness only after the listening socket and artifact root exist.
    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            args.ready_file,
            {
                "status": "ready",
                "host": args.listen_host,
                "port": args.listen_port,
                "upstream_url": urlunsplit(args.upstream),
                "output_dir": str(config.output_dir),
            },
        )

    # Convert process signals into a clean server shutdown from another thread.
    def request_shutdown(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(
        f"[telemetry-proxy] listening on "
        f"http://{args.listen_host}:{args.listen_port}; "
        f"upstream={urlunsplit(args.upstream)}; output={config.output_dir}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
