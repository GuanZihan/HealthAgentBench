from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from faults.telemetry.extract_hidden_states import (
    resolve_layer_selection,
    select_calls,
)
from faults.telemetry.proxy import ProxyConfig, TelemetryHTTPServer
from faults.telemetry.responses import (
    CapturedCall,
    discover_response_calls,
    reconstruct_context,
)


class FakeVLLMHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Return a minimal streaming Responses sequence that exercises SSE capture.
    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        self.server.request_body = self.rfile.read(content_length)  # type: ignore[attr-defined]
        events = [
            {
                "type": "response.created",
                "response": {"id": "resp_test", "object": "response", "output": []},
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "OK"}],
                        }
                    ],
                },
            },
        ]
        payload = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # Keep test output quiet because assertions inspect persisted access records.
    def log_message(self, _format: str, *_args: object) -> None:
        return


# Start a server in a daemon thread and return its bound ephemeral port.
def start_server(server: ThreadingHTTPServer) -> tuple[threading.Thread, int]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread, int(server.server_address[1])


def test_proxy_captures_streaming_response_and_redacts_credentials(
    tmp_path: Path,
) -> None:
    # Run a fake upstream and the telemetry proxy on independent ephemeral ports.
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLMHandler)
    upstream_thread, upstream_port = start_server(upstream)
    config = ProxyConfig(
        listen_host="127.0.0.1",
        listen_port=0,
        upstream=urlsplit(f"http://127.0.0.1:{upstream_port}"),
        output_dir=tmp_path / "capture",
        max_request_bytes=1024 * 1024,
        metadata={"condition": "error"},
    )
    proxy = TelemetryHTTPServer(config)
    proxy_thread, proxy_port = start_server(proxy)

    # Send a realistic Responses body and consume the proxy's chunked response.
    request_body = json.dumps(
        {"model": "qwen-coder", "input": "Reply with OK", "stream": True}
    ).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=5)
    connection.request(
        "POST",
        "/v1/responses",
        body=request_body,
        headers={
            "Authorization": "Bearer secret-value",
            "Content-Type": "application/json",
        },
    )
    response = connection.getresponse()
    response_body = response.read()
    connection.close()

    # Shut down both local servers before reading the final call manifest.
    proxy.shutdown()
    upstream.shutdown()
    proxy_thread.join(timeout=5)
    upstream_thread.join(timeout=5)
    proxy.server_close()
    upstream.server_close()

    assert response.status == 200
    assert b"response.completed" in response_body
    call_dir = next((tmp_path / "capture" / "calls").iterdir())
    request_record = json.loads((call_dir / "request.json").read_text())
    response_record = json.loads((call_dir / "response.json").read_text())
    recorded_headers = {
        item["name"].lower(): item["value"] for item in request_record["headers"]
    }
    assert recorded_headers["authorization"] == "<redacted>"
    assert (call_dir / "request.body").read_bytes() == request_body
    assert (call_dir / "response.body").read_bytes() == response_body
    assert response_record["response_ids"] == ["resp_test"]
    assert len((call_dir / "response-events.jsonl").read_text().splitlines()) == 2
    assert len((call_dir / "response-chunks.jsonl").read_text().splitlines()) >= 1
    assert upstream.request_body == request_body  # type: ignore[attr-defined]


def test_reconstruct_context_resolves_previous_response_chain(tmp_path: Path) -> None:
    # Model a parent tool call followed by a child function result.
    parent = CapturedCall(
        sequence=1,
        call_id="call_parent",
        call_dir=tmp_path / "parent",
        request={
            "instructions": "Be precise.",
            "input": "Inspect the file.",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Run a command.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
        response={
            "id": "resp_parent",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_tool",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                }
            ],
        },
        response_ids=("resp_parent",),
    )
    child = CapturedCall(
        sequence=2,
        call_id="call_child",
        call_dir=tmp_path / "child",
        request={
            "instructions": "Be precise.",
            "previous_response_id": "resp_parent",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_tool",
                    "output": "No such file or directory",
                }
            ],
            "tools": parent.request["tools"],
        },
        response={"id": "resp_child", "output": []},
        response_ids=("resp_child",),
    )

    # Verify semantic order and normalized Qwen tool schemas.
    context = reconstruct_context(child, [parent, child])
    assert [message["role"] for message in context.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert context.messages[-1]["content"] == "No such file or directory"
    assert context.messages[2]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert context.messages[2]["tool_calls"][0]["function"]["arguments"] == {
        "cmd": "ls"
    }
    assert context.tools[0]["function"]["name"] == "exec_command"
    assert context.response_chain == ("resp_parent",)
    assert context.warnings == ()


def test_discovery_and_layer_selection_are_deterministic(tmp_path: Path) -> None:
    # Build one minimal completed artifact and one irrelevant GET artifact.
    capture_dir = tmp_path / "capture"
    response_dir = capture_dir / "calls" / "000002-call_response"
    response_dir.mkdir(parents=True)
    (response_dir / "call.json").write_text(
        json.dumps(
            {
                "sequence": 2,
                "call_id": "call_response",
                "method": "POST",
                "target": "/v1/responses",
                "outcome": "completed",
                "response_ids": [],
            }
        )
    )
    (response_dir / "request-body.json").write_text(
        json.dumps({"body": {"model": "qwen-coder", "input": "hello"}})
    )

    # Resolve fractional, positive, and negative layer specifications without duplicates.
    calls = discover_response_calls(capture_dir)
    selection = resolve_layer_selection("0.25,0.5,1.0,-1", layer_count=40)
    assert [call.sequence for call in calls] == [2]
    assert selection.indices == (9, 19, 39)
    assert select_calls(calls, [2], select_all=False) == calls
