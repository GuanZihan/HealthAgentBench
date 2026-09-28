from __future__ import annotations

import os
import subprocess
from pathlib import Path


QWEN_VLLM_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = QWEN_VLLM_DIR / "launch_qwen_coder.sh"
BENCHMARK_ROOT = QWEN_VLLM_DIR.parents[1]


def test_dry_run_uses_isolated_versioned_sif_without_recording_token(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    model_cache = tmp_path / "model-cache"
    secret = "secret-must-not-be-recorded"
    env = os.environ | {
        "MODEL_ID": "Qwen/example-coder",
        "MODEL_REVISION": "example-commit",
        "TENSOR_PARALLEL_SIZE": "4",
        "OUTPUT_DIR": str(output_dir),
        "MODEL_CACHE_DIR": str(model_cache),
        "RECORD_SIF_SHA256": "0",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "HF_TOKEN": secret,
    }

    completed = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        cwd=BENCHMARK_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    command = (output_dir / "command.sh").read_text()
    run_config = (output_dir / "run-config.env").read_text()
    codex_config = (output_dir / "codex-config.toml").read_text()

    assert "apptainer exec --cleanenv --nv" in command
    assert "vllm-openai-v0.29.0-cu129.sif" in command
    assert 'APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"' in command
    assert 'APPTAINERENV_HF_TOKEN="$HF_TOKEN"' in command
    assert secret not in command
    assert secret not in run_config
    assert "supports_websockets = false" in codex_config
    assert "dry run; the service was not started" in completed.stdout
