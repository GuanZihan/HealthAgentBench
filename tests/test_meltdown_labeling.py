from __future__ import annotations

import base64
import json
from pathlib import Path

from faults.telemetry.meltdown_labeling_common import (
    clean_trace,
    load_paper_materials,
)


# Encode a miniature arXiv listing with the same data-URI format as the paper.
def listing(value: str) -> str:
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return (
        '<div class="ltx_listing_data">'
        f'<a href="data:text/plain;base64,{encoded}">download</a>'
        "</div>"
    )


# Verify exact prompt decoding and fail-closed placeholder preservation.
def test_load_paper_materials_decodes_downloadable_prompts(tmp_path: Path) -> None:
    discovery = "Discovery prompt\n<TRACE>"
    aggregation = "Aggregation prompt\n<dir>"
    annotation = "Annotation prompt\n<TAXONOMY>\n<TRACE>"
    html = f"""
    <html><body>
      <section><h2>Appendix B Full Behavior Taxonomy</h2>
        <p>B01 Local reconnaissance</p><p>B13 Private content disclosure</p>
      </section>
      <section><h2>Appendix C Taxonomy generation prompts</h2>
        <section><h3>C.1 Behavior discovery prompt</h3>{listing(discovery)}</section>
        <section><h3>C.2 Category aggregation prompt</h3>{listing(aggregation)}</section>
        <section><h3>C.3 Behavior Annotation Prompt</h3>{listing(annotation)}</section>
      </section>
    </body></html>
    """
    html_path = tmp_path / "paper.html"
    html_path.write_text(html, encoding="utf-8")

    # Read through the public loader because it also exercises cached artifacts.
    materials = load_paper_materials(tmp_path / "cache", paper_html=html_path)

    assert materials.discovery_prompt == discovery
    assert materials.aggregation_prompt == aggregation
    assert materials.annotation_prompt == annotation
    assert "B01 Local reconnaissance" in materials.frozen_taxonomy
    assert "B13 Private content disclosure" in materials.frozen_taxonomy


# Verify large image bytes are removed even inside a serialized tool-result string.
def test_clean_trace_removes_nested_base64_but_keeps_action_context(
    tmp_path: Path,
) -> None:
    payload = "A" * 10_000
    nested_result = (
        "[{'type': 'input_image', "
        f"'image_url': 'data:image/png;base64,{payload}'}}]"
    )
    trace_path = tmp_path / "trajectory.json"
    trace_path.write_text(
        json.dumps(
            {
                "action": "view_image",
                "tool_output": nested_result,
                "usage": {"total_tokens": 123},
            }
        ),
        encoding="utf-8",
    )

    # Keep behavioral evidence while dropping both binary bytes and bookkeeping.
    cleaned = clean_trace(trace_path)

    assert '"action": "view_image"' in cleaned
    assert "embedded image/png base64 payload omitted; 10000 characters" in cleaned
    assert payload not in cleaned
    assert '"usage"' not in cleaned
