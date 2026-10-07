# 在 Rivanna 上用 vLLM 启动 Qwen Coder

这套小工具做一件事：在 GPU 计算节点上，用 Apptainer 启动一个
Qwen Coder 服务，并把它暴露成 Codex 能访问的 OpenAI Responses API。
它不依赖 Docker，也不会偷偷修改你的 `~/.codex/config.toml`。

启动脚本是 `launch_qwen_coder.sh`。每次运行都会留下服务日志、完整命令、
SIF checksum、GPU 信息和最终使用的参数。这样几周后回头看实验结果时，
不会只剩下一句“当时应该用了某个 Qwen”。

## 先说最重要的限制

Qwen Coder 不是只要能生成文本就够了。Codex 会要求模型正确地产生 shell、
文件编辑等 tool calls，所以 vLLM 必须同时满足下面两点：

1. 提供 `/v1/responses`；
2. 为模型启用正确的 tool-call parser。

当前脚本默认使用 `qwen3_coder` parser。vLLM 的 Qwen3-Coder recipe 表明，
这项 tool-call 支持需要 vLLM 0.10.0 或更高版本；实际实验建议固定一个较新的、
已经验证过 Responses API 的稳定版本，而不是使用浮动的 `latest`。

参考资料：

- [vLLM 的 Codex 集成说明](https://docs.vllm.ai/en/latest/serving/integrations/codex/)
- [vLLM 的 Qwen3-Coder recipe](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3-Coder-480B-A35B.html)
- [Codex 自定义 provider 配置](https://learn.chatgpt.com/docs/config-file/config-advanced)

## 1. 准备 vLLM SIF

这一步只需要做一次。下面固定使用官方单独发布的 CUDA 12.9 artifact
`v0.29.0-cu129`，也避免了浮动的 `latest` 在不同日期指向不同环境。当前本地
`v0.30.0-cu129` 镜像已经观察到 torch/torchvision CUDA build 不匹配，启动时报
`RuntimeError: operator torchvision::nms does not exist`。

相关记录：

- [vLLM v0.29.0 release artifacts](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [vLLM 上游同类 CUDA wheel mismatch 报告](https://github.com/vllm-project/vllm/issues/56829)

```bash
cd HealthAgentBench
module load apptainer/1.5.0

pull_tmp="$(mktemp -d "/tmp/${USER}-vllm-pull.XXXXXX")"
export APPTAINER_IGNORE_PROOT=1
export APPTAINER_TMPDIR="$pull_tmp"
export APPTAINER_CACHEDIR="$PWD/.apptainer-cache"
mkdir -p "$APPTAINER_CACHEDIR"

VLLM_TAG=v0.29.0-cu129
VLLM_SIF_PATH="$PWD/containers/vllm-openai-${VLLM_TAG}.sif"
apptainer build \
  "$VLLM_SIF_PATH" \
  "docker://vllm/vllm-openai:${VLLM_TAG}"

sha256sum "$VLLM_SIF_PATH"
rm -rf -- "$pull_tmp"
```
  --mksquashfs-args "-processors 2 -mem 8G -single-reader" \

`APPTAINER_IGNORE_PROOT=1` 是 Rivanna 当前 Apptainer 1.5.0 环境的重要兼容
设置。若不设置，OCI layers 已经下载完以后，`mksquashfs` 仍可能因为节点禁止
`ptrace` 而以 `proot error: ptrace(TRACEME): Operation not permitted` 失败。

这里有意使用 `apptainer build`，而不是更短的 `apptainer pull`。两者都会从
同一个 OCI image 创建 SIF，但 `build` 可以把参数传给 `mksquashfs`。Rivanna
节点的物理内存可能明显大于当前 Slurm job 获得的内存；若沿用
`mksquashfs` 的默认 25% cache 和全部可见 CPU，它可能超过 job memory limit，
然后只留下含义不太明显的 `command failed: signal: killed`。限制为 2 个压缩
线程和 8 GB cache 会慢一些，但更适合交互式构建，也不改变最终容器内容。

如果之后有意升级 vLLM，请换一个新的明确 release tag，并把它作为新的实验
环境记录；不要静默覆盖旧实验使用的版本。

`containers/*.sif` 已经被 git 忽略，不会把几 GB 的镜像误提交进仓库。
脚本默认会再次记录这个 SIF 的 SHA-256；如果 GPFS 上计算大镜像 checksum
太慢，可以临时设置 `RECORD_SIF_SHA256=0`，不过正式结果最好保留它。

模型权重可以让 vLLM 从 Hugging Face 下载，也可以提前放到本地共享目录。
第一次下载通常很慢，脚本默认把缓存放在：

```text
HealthAgentBench/.cache/qwen-vllm
```

也可以用 `MODEL_CACHE_DIR` 指向容量更大的持久化目录。

## 2. 申请 GPU

先根据模型大小申请一个单节点 GPU allocation。下面的 partition、GPU 类型和
数量只是占位符，需要换成 Rivanna 上实际可用的资源：

```bash
salloc \
  --nodes=1 \
  --partition=gpu-partition \
  --gres=gpu:gpu-type:gpu-count \
  --time=08:00:00
```

`Qwen3-Coder-480B-A35B-Instruct` 很大。vLLM 官方示例使用 8 张 141 GB 的
H200/H20 级 GPU；不要默认拿它在单卡上试。可以选择更小或量化的 Qwen Coder，
但需要把准确的模型 ID、revision、量化方式和 GPU 配置记录下来。

## 3. 启动服务

进入计算节点后运行。下面保留两套已经固定 revision 的配置，实验记录里不要只写
`qwen-coder`，还要保存具体的 `MODEL_ID` 和 `MODEL_REVISION`。

### Qwen3-Coder-30B-A3B-Instruct

```bash
cd HealthAgentBench

MODEL_ID=Qwen/Qwen3-Coder-30B-A3B-Instruct \
MODEL_REVISION=b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
VLLM_SIF="$PWD/containers/vllm-openai-v0.29.0-cu129.sif" \
PORT=8010 \
TENSOR_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=32768 \
GPU_MEMORY_UTILIZATION=0.88 \
RUN_ID=qwen3-coder-rep1 \
  bash llm_backends/qwen_vllm/launch_qwen_coder.sh -- \
    --enable-expert-parallel
```

### Qwen3-Coder-Next-FP8

当前本地缓存对应的模型 revision 是
`da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`。这个模型的 `config.json` 已经声明
FP8 quantization，所以让 vLLM 自动读取模型配置即可，不需要再额外添加
`--quantization fp8`。

```bash
cd HealthAgentBench

MODEL_ID=Qwen/Qwen3-Coder-Next-FP8 \
MODEL_REVISION=da6e2ed27304dd39abadd9c82ef50e8de67bdd4c \
VLLM_SIF="$PWD/containers/vllm-openai-v0.29.0-cu129.sif" \
SERVED_MODEL_NAME=qwen-coder \
PORT=8010 \
TENSOR_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=32768 \
DTYPE=auto \
GPU_MEMORY_UTILIZATION=0.88 \
RUN_ID=qwen3-coder-next-fp8-rep1 \
  bash llm_backends/qwen_vllm/launch_qwen_coder.sh -- \
    --enable-expert-parallel
```

`Qwen3-Coder-Next-FP8` 的权重显著大于上面的 30B-A3B 模型。只有单卡显存足够时
才使用 `TENSOR_PARALLEL_SIZE=1`；使用多张 GPU 时，把它改成实际 GPU 数量，并在
同一个 Slurm allocation 内启动服务。模型服务仍然使用 API alias
`qwen-coder`，所以 benchmark 侧的 `MODEL=qwen-coder` 不需要改变。

这条命令会保持在前台。看到下面的信息，才说明服务真的可以接请求了：

```text
[qwen-vllm] service is ready: http://127.0.0.1:8010/v1
```

这里的端口来自 `PORT`。如果改回脚本默认的 `8000`，后续 curl 和 benchmark
命令里的 URL 也要一起改成 `8000`。

如果模型是本地目录，`MODEL_ID` 使用绝对路径即可，脚本会把它只读挂进
Apptainer。此时 `MODEL_REVISION` 只作为本地 snapshot 的记录标签，不会作为
vLLM 的 `--revision` 参数：

```bash
MODEL_ID=/path/to/Qwen-Coder-model \
MODEL_REVISION=local-snapshot-2026-09-23 \
VLLM_SIF="$PWD/containers/vllm-openai-v0.29.0-cu129.sif" \
TENSOR_PARALLEL_SIZE=4 \
  bash llm_backends/qwen_vllm/launch_qwen_coder.sh
```

如果想先检查解析后的命令，不启动模型：

```bash
MODEL_ID=Qwen/replace-with-your-qwen-coder-model \
VLLM_SIF="$PWD/containers/vllm-openai-v0.29.0-cu129.sif" \
TENSOR_PARALLEL_SIZE=4 \
  bash llm_backends/qwen_vllm/launch_qwen_coder.sh --dry-run
```

额外的 vLLM 参数全部放在 `--` 后面。它们会原样进入命令，并被记录在
`command.sh` 中。例如：

```bash
bash llm_backends/qwen_vllm/launch_qwen_coder.sh -- \
  --quantization fp8 \
  --enable-expert-parallel
```

## 4. 服务启动后会留下什么

默认输出目录是 `results/vllm/{run-id}/`：

| 文件 | 用途 |
| --- | --- |
| `server.log` | vLLM 的完整 stdout/stderr |
| `command.sh` | 完整、可 shell-safe 重放的启动命令 |
| `run-config.env` | 模型、revision、SIF checksum 和推理参数 |
| `python-packages.txt` | SIF 内 vLLM、torch、torchvision 和 torchaudio 版本 |
| `container-preflight.log` | torch/torchvision 启动前兼容性检查 |
| `gpus.csv` | GPU 型号、UUID、驱动和显存 |
| `models.json` | readiness check 返回的模型列表 |
| `codex-config.toml` | 普通 Codex CLI 可用的 provider 配置快照 |
| `endpoint.env` | HealthAgentBench/Harbor 使用的 endpoint 变量 |
| `READY` | 只有 `/v1/models` 成功后才会出现 |

这些文件不包含 Hugging Face token。`endpoint.env` 里的 API key 是
localhost 服务使用的 dummy 值。

## 5. 先做一个轻量检查

在同一个计算节点的另一个 shell 中，检查模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

再检查 Responses API，而不只是 Chat Completions：

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-coder",
    "input": "Reply with exactly READY."
  }'
```

能返回普通文本只是第一关。正式跑 benchmark 前，仍然要跑一次真实 Codex
smoke test，确认模型能生成工具调用并读写 workspace。

## 6. 用普通 Codex CLI 测试

启动脚本会生成 `codex-config.toml`。为了不碰个人配置，可以创建一个临时
`CODEX_HOME`：

```bash
run_dir=results/vllm/qwen3-coder-rep1
codex_home="$(mktemp -d /tmp/qwen-codex.XXXXXX)"

cp "$run_dir/codex-config.toml" "$codex_home/config.toml"
export CODEX_HOME="$codex_home"
export VLLM_API_KEY=dummy

codex exec --model qwen-coder \
  'List the files in the current directory, then summarize README.md.'
```

这个测试必须观察到真实的 shell/file tool call；只有最终文字回答、没有工具调用，
不能证明它适合跑 HealthAgentBench。

## 7. 接到 HealthAgentBench

当前 Harbor 0.8.0 的 Codex adapter 会为每个 rollout 创建独立的临时
`CODEX_HOME`。因此这里不复制上面的 TOML，而是显式传入 endpoint。服务和
benchmark 必须位于同一个 Slurm 节点；Apptainer 默认共享宿主机网络，所以
可以继续使用 `127.0.0.1:8000`。

先跑一个 control smoke test：

```bash
MODEL=qwen-coder \
N_ATTEMPTS=1 \
N_CONCURRENT=1 \
JOB_NAME=ct-qwen-smoke-control \
  bash faults/scripts/run_family_control.sh \
    ct_abnormality_valid_81_a_1 file-not-found \
    --agent-env OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    --agent-env OPENAI_API_KEY=dummy
```

确认 trajectory 中确实出现成功的 shell 和文件操作以后，再跑 matched pair：

```bash
MODEL=qwen-coder N_ATTEMPTS=1 N_CONCURRENT=1 \
JOB_NAME=ct-qwen-file-not-found-rep1-control \
  bash faults/scripts/run_family_control.sh \
    ct_abnormality_valid_81_a_1 file-not-found \
    --agent-env OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    --agent-env OPENAI_API_KEY=dummy

MODEL=qwen-coder N_ATTEMPTS=1 N_CONCURRENT=1 \
JOB_NAME=ct-qwen-file-not-found-rep1-error \
  bash faults/scripts/run_family_error.sh \
    ct_abnormality_valid_81_a_1 file-not-found \
    --agent-env OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    --agent-env OPENAI_API_KEY=dummy
```

如果要跑普通任务而不是 fault pair，同样把两个 `--agent-env` 参数附加到
`containers/run_family_task.sh` 后面即可。

### Qwen Coder 的图像工具策略

`qwen-coder` 是纯文本服务别名。现有 fault runners 检测到
`--agent codex --model qwen-coder` 时，会自动使用
`faults.codex_agent:TextOnlyCodex`。由于多个已发布 Codex CLI 版本不会可靠执行
`tools.view_image=false`，adapter 会启动一个仅属于当前 trial 的本地过滤层。它在
请求到达 telemetry 和 vLLM 前删除 `view_image` 工具定义、对应的历史调用以及
残留的 `input_image` 内容。其他模型继续使用 Harbor 原始 Codex adapter，不受影响。

关闭 `view_image` 只会避免文本模型收到不支持的图像输入，并不会让 Qwen Coder
获得视觉能力。CT、X-ray 和 pathology 任务仍只能依靠 Python、图像统计和其他
命令行分析；需要可靠的视觉判读时应换用兼容的多模态模型。

## 8. 实验记录建议

为了让 Qwen 结果能和 GPT 结果公平比较，至少固定并记录：

- Qwen 的完整模型 ID 与 revision；
- vLLM SIF SHA-256 和 vLLM 版本；
- dtype、quantization、tensor parallel、context length 和 seed；
- GPU 型号与数量；
- Codex CLI 和 Harbor 版本；
- control/error 的 task、scenario、replicate 和执行顺序。

同一个 matched pair 应使用同一个服务配置。第一轮建议保持
`N_CONCURRENT=1`，先排除并发调度、KV cache 压力和 OOM 对行为的干扰。
如果每个 replicate 都重启服务，那么 control 和 error 都要遵循同样的重启规则。

## 常见问题

### `RuntimeError: operator torchvision::nms does not exist`

这是 vLLM SIF 内部的 torch/torchvision 二进制不匹配，不是 Qwen 权重损坏。
先查看旧镜像实际包含的版本，无需导入 torchvision：

```bash
apptainer exec --cleanenv containers/vllm-openai.sif python3 -c \
  'from importlib.metadata import version; print({p: version(p) for p in ("vllm", "torch", "torchvision", "torchaudio")})'
```

若旧镜像来自 `v0.30.0-cu129`，按第 1 节另建
`containers/vllm-openai-v0.29.0-cu129.sif`，再把 `VLLM_SIF` 指向新文件。
不要直接在原 SIF 上临时 `pip install`：SIF 本身只读，而 writable overlay 会让
真实依赖脱离 SIF checksum，后续实验很难复现。launcher 现在会在加载模型前
执行兼容性检查，并把结果写入 `python-packages.txt` 和
`container-preflight.log`。

旧的 `containers/vllm-openai.sif` 可以暂时保留；等新镜像通过 preflight 和
smoke test 后，再决定是否清理。

### launcher 说没有 GPU，或者可见 GPU 数量不足

先运行 `nvidia-smi -L`。命令必须列出至少 `TENSOR_PARALLEL_SIZE` 张 GPU。
例如设置 `TENSOR_PARALLEL_SIZE=4` 就必须在当前 Slurm allocation 中真正分到
4 张 GPU；登录节点或 `SLURM_JOB_GPUS=0` 的作业不能运行 vLLM。

### `/v1/chat/completions` 可以，但 Codex 连不上

先直接测试 `/v1/responses`。Codex 自定义 provider 使用的是 Responses API，
Chat Completions 正常并不能说明 Codex 路径正常。

### 能回答问题，但从不调用工具

检查启动日志和最终命令里是否同时存在：

```text
--enable-auto-tool-choice --tool-call-parser qwen3_coder
```

如果使用的不是 Qwen3-Coder，需要按该模型的 vLLM 文档更换
`TOOL_CALL_PARSER`，不能机械地沿用 `qwen3_coder`。

### Harbor 报 model not found

HealthAgentBench 的 `MODEL` 必须和 `SERVED_MODEL_NAME` 一致。默认二者都是
`qwen-coder`。可以查看 `results/vllm/{run-id}/models.json` 确认服务实际暴露的
名字。

### 在另一个节点访问不到服务

默认只监听 `127.0.0.1`，这是有意的：简单，也不会把无认证的模型端口暴露到
集群网络。最稳妥的做法是让 vLLM 与 Harbor 位于同一个单节点 allocation，
而不是把端口开放到其他节点。
