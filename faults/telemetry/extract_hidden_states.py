"""Extract pooled Qwen hidden states from captured vLLM Responses calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from faults.telemetry.responses import (
    CapturedCall,
    discover_response_calls,
    reconstruct_context,
    render_prompt,
)


@dataclass(frozen=True)
class LayerSelection:
    """Resolved zero-based transformer block indices and their original specification."""

    specification: str
    indices: tuple[int, ...]


# Import GPU-heavy dependencies only after argument validation and call selection.
def import_model_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "hidden-state extraction requires numpy, torch, and transformers; "
            "run the script inside the pinned vLLM SIF"
        ) from error
    return numpy, torch, AutoModelForCausalLM, AutoTokenizer


# Generate a stable SHA-256 for rendered prompts and token sequences.
def sha256_payload(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# Read package versions without assuming every optional runtime package is installed.
def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


# Resolve dotted attribute paths without relying on one Qwen implementation layout.
def resolve_attribute(root: Any, path: str) -> Any | None:
    current = root
    for component in path.split("."):
        if not hasattr(current, component):
            return None
        current = getattr(current, component)
    return current


# Locate the ordered transformer blocks across common Hugging Face causal-LM layouts.
def find_transformer_layers(model: Any, torch_module: Any) -> Sequence[Any]:
    candidate_paths = (
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "transformer.h",
        "gpt_neox.layers",
    )
    for candidate_path in candidate_paths:
        candidate = resolve_attribute(model, candidate_path)
        if isinstance(candidate, torch_module.nn.ModuleList) and len(candidate) > 0:
            return candidate

    # Fall back to the largest ModuleList named like transformer blocks.
    candidates: list[tuple[int, str, Sequence[Any]]] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, torch_module.nn.ModuleList) or len(module) == 0:
            continue
        final_name = module_name.rsplit(".", 1)[-1]
        if final_name in {"layers", "h", "blocks"}:
            candidates.append((len(module), module_name, module))
    if not candidates:
        raise RuntimeError("could not locate transformer layer ModuleList")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


# Translate quartile/fraction/index specifications into unique layer indices.
def resolve_layer_selection(specification: str, layer_count: int) -> LayerSelection:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    normalized = specification.strip().lower()
    if normalized == "quartiles":
        tokens = ["0.25", "0.5", "0.75", "1.0"]
    else:
        tokens = [token.strip() for token in specification.split(",") if token.strip()]
    if not tokens:
        raise ValueError("layer specification is empty")
    indices: list[int] = []
    for token in tokens:
        if any(marker in token for marker in (".", "e")):
            fraction = float(token)
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"layer fraction must be in (0, 1]: {token}")
            index = math.ceil(fraction * layer_count) - 1
        else:
            index = int(token)
            if index < 0:
                index += layer_count
            if not 0 <= index < layer_count:
                raise ValueError(f"layer index is out of range: {token}")
        if index not in indices:
            indices.append(index)
    return LayerSelection(specification=specification, indices=tuple(indices))


# Select explicit call sequences or require --all to prevent accidental GPU-heavy runs.
def select_calls(
    calls: Sequence[CapturedCall],
    sequences: Sequence[int],
    select_all: bool,
) -> list[CapturedCall]:
    if select_all and sequences:
        raise ValueError("use either --all or --call-sequence, not both")
    if not select_all and not sequences:
        raise ValueError(
            "select calls with --all or one or more --call-sequence values"
        )
    if select_all:
        return list(calls)
    by_sequence = {call.sequence: call for call in calls}
    missing = [sequence for sequence in sequences if sequence not in by_sequence]
    if missing:
        raise ValueError(f"captured Responses call sequences not found: {missing}")
    return [by_sequence[sequence] for sequence in sequences]


# Tokenize a rendered context and optionally append a standardized probe suffix.
def tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    probe_text: str,
    max_length: int | None,
    truncate_left: bool,
    torch_module: Any,
) -> tuple[Any, int, int, bool]:
    context_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    probe_ids = (
        tokenizer(probe_text, add_special_tokens=False)["input_ids"]
        if probe_text
        else []
    )
    token_ids = list(context_ids) + list(probe_ids)
    truncated = False
    if max_length is not None and len(token_ids) > max_length:
        if not truncate_left:
            raise ValueError(
                f"rendered context has {len(token_ids)} tokens, exceeding "
                f"--max-length={max_length}; pass --truncate-left only if this "
                "matches the generation-time policy"
            )
        removed = len(token_ids) - max_length
        token_ids = token_ids[removed:]
        removed_from_context = min(removed, len(context_ids))
        removed_from_probe = max(0, removed - len(context_ids))
        context_ids = context_ids[removed_from_context:]
        probe_ids = probe_ids[removed_from_probe:]
        truncated = True
    if not token_ids:
        raise ValueError("rendered context produced no tokens")
    input_ids = torch_module.tensor([token_ids], dtype=torch_module.long)
    return input_ids, len(context_ids), len(probe_ids), truncated


# Identify the device hosting input embeddings under a Hugging Face device map.
def input_device(model: Any) -> Any:
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is None:
        raise RuntimeError("model input embeddings do not expose a weight tensor")
    return weight.device


# Pool context-last and optional probe representations directly inside each hook.
def build_layer_hook(
    *,
    layer_index: int,
    context_length: int,
    probe_length: int,
    captured: dict[int, dict[str, Any]],
    torch_module: Any,
) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch_module.is_tensor(hidden):
            raise TypeError(f"layer {layer_index} hook did not receive a tensor")
        representations: dict[str, Any] = {}
        if context_length > 0:
            representations["context_last"] = (
                hidden[0, context_length - 1].detach().float().cpu()
            )
        if probe_length > 0:
            probe_hidden = hidden[0, context_length : context_length + probe_length]
            representations["probe_mean"] = (
                probe_hidden.mean(dim=0).detach().float().cpu()
            )
            representations["probe_last"] = probe_hidden[-1].detach().float().cpu()
        captured[layer_index] = representations

    return hook


# Run the transformer backbone once so the LM head does not materialize full-sequence logits.
def run_backbone(
    model: Any, input_ids: Any, attention_mask: Any, torch_module: Any
) -> None:
    backbone = getattr(model, "model", None)
    target = backbone if backbone is not None and backbone is not model else model
    signature = inspect.signature(target.forward)
    forward_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
    }
    if "output_hidden_states" in signature.parameters:
        forward_kwargs["output_hidden_states"] = False
    if target is model and "logits_to_keep" in signature.parameters:
        forward_kwargs["logits_to_keep"] = 1
    with torch_module.inference_mode():
        target(**forward_kwargs)


# Save numeric arrays and provenance sidecars for one captured model call.
def save_call_artifacts(
    *,
    output_dir: Path,
    call: CapturedCall,
    prompt: str,
    token_ids: Any,
    representations: dict[int, dict[str, Any]],
    metadata: dict[str, Any],
    numpy_module: Any,
) -> dict[str, Any]:
    stem = f"{call.sequence:06d}-{call.call_id}"
    prompt_dir = output_dir / "prompts"
    state_dir = output_dir / "states"
    metadata_dir = output_dir / "metadata"
    for directory in (prompt_dir, state_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Keep the rendered prompt beside the state vectors for audit and reruns.
    prompt_path = prompt_dir / f"{stem}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    arrays: dict[str, Any] = {
        "token_ids": token_ids.detach().cpu().numpy().astype("int32", copy=False)
    }
    for layer_index, pooled_states in representations.items():
        for state_name, tensor in pooled_states.items():
            arrays[f"layer_{layer_index:04d}_{state_name}"] = tensor.numpy().astype(
                "float32", copy=False
            )
    state_path = state_dir / f"{stem}.npz"
    numpy_module.savez_compressed(state_path, **arrays)

    # Record hashes and paths so every row can be independently verified.
    token_bytes = arrays["token_ids"].tobytes(order="C")
    complete_metadata = {
        **metadata,
        "sequence": call.sequence,
        "call_id": call.call_id,
        "capture_call_dir": str(call.call_dir.resolve()),
        "prompt_path": str(prompt_path.resolve()),
        "state_path": str(state_path.resolve()),
        "prompt_sha256": sha256_payload(prompt.encode("utf-8")),
        "token_ids_sha256": sha256_payload(token_bytes),
        "saved_arrays": sorted(arrays),
    }
    metadata_path = metadata_dir / f"{stem}.json"
    metadata_path.write_text(
        json.dumps(complete_metadata, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return complete_metadata


# Define a reproducible CLI that keeps model and capture selection explicit.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay captured Responses contexts and extract Qwen hidden states."
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--layers", default="quartiles")
    parser.add_argument("--probe-text", default="")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--truncate-left", action="store_true")
    parser.add_argument("--call-sequence", type=int, action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


# Load the pinned model once and extract selected representations call by call.
def main() -> None:
    args = build_parser().parse_args()
    calls = discover_response_calls(args.capture_dir.resolve())
    try:
        selected_calls = select_calls(calls, args.call_sequence, args.all)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not selected_calls:
        raise SystemExit("no completed POST /responses calls were found")
    numpy_module, torch_module, auto_model, auto_tokenizer = import_model_dependencies()

    # Pin tokenizer provenance separately because tokenizer and weights may diverge.
    tokenizer_id = args.tokenizer or args.model
    tokenizer_revision = args.tokenizer_revision or args.revision
    tokenizer = auto_tokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    # Load across visible GPUs through Accelerate unless an explicit single device is requested.
    model_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "dtype": args.dtype,
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = auto_model.from_pretrained(args.model, **model_kwargs)
    if args.device_map.lower() == "none":
        model.to(args.device)
    model.eval()
    layers = find_transformer_layers(model, torch_module)
    try:
        layer_selection = resolve_layer_selection(args.layers, len(layers))
    except ValueError as error:
        raise SystemExit(str(error)) from error

    # Create an append-only manifest after every successful extraction.
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "extractions.jsonl"
    for call in selected_calls:
        context = reconstruct_context(call, calls)
        prompt = render_prompt(tokenizer, context)
        input_ids, context_length, probe_length, truncated = tokenize_prompt(
            tokenizer,
            prompt,
            args.probe_text,
            args.max_length,
            args.truncate_left,
            torch_module,
        )
        device = input_device(model)
        input_ids = input_ids.to(device)
        attention_mask = torch_module.ones_like(input_ids, device=device)

        # Register only selected layer hooks to keep activation memory bounded.
        captured: dict[int, dict[str, Any]] = {}
        handles = []
        for layer_index in layer_selection.indices:
            handles.append(
                layers[layer_index].register_forward_hook(
                    build_layer_hook(
                        layer_index=layer_index,
                        context_length=context_length,
                        probe_length=probe_length,
                        captured=captured,
                        torch_module=torch_module,
                    )
                )
            )
        try:
            run_backbone(model, input_ids, attention_mask, torch_module)
        finally:
            for handle in handles:
                handle.remove()
        missing_layers = sorted(set(layer_selection.indices) - set(captured))
        if missing_layers:
            raise RuntimeError(
                f"hooks did not observe selected layers: {missing_layers}"
            )

        # Persist all configuration needed to reproduce this exact replay decision.
        metadata = save_call_artifacts(
            output_dir=output_dir,
            call=call,
            prompt=prompt,
            token_ids=input_ids,
            representations=captured,
            metadata={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "revision": args.revision,
                "tokenizer": tokenizer_id,
                "tokenizer_revision": tokenizer_revision,
                "dtype": args.dtype,
                "device_map": args.device_map,
                "layer_count": len(layers),
                "layer_specification": layer_selection.specification,
                "layer_indices": list(layer_selection.indices),
                "context_token_count": context_length,
                "probe_token_count": probe_length,
                "total_token_count": int(input_ids.shape[1]),
                "probe_text": args.probe_text,
                "truncated": truncated,
                "reconstruction_warnings": list(context.warnings),
                "response_chain": list(context.response_chain),
                "software": {
                    "accelerate": package_version("accelerate"),
                    "numpy": package_version("numpy"),
                    "torch": package_version("torch"),
                    "transformers": package_version("transformers"),
                },
            },
            numpy_module=numpy_module,
        )
        with manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        print(
            f"[hidden-state] sequence={call.sequence} "
            f"tokens={input_ids.shape[1]} layers={list(layer_selection.indices)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
