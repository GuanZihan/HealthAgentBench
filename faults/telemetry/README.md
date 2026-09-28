# Qwen rollout telemetry

这个目录做两件彼此独立的事：

1. `proxy.py` 在 Codex 和 vLLM 之间记录每一次 HTTP 调用；
2. `extract_hidden_states.py` 在 rollout 结束后重建 context，并离线提取 Qwen
   的 hidden states。

采集阶段不计算 control/error 差分，也不要求两个 rollout 在步骤上对齐。每条
rollout 都是独立的研究样本。

## 目录结构

```text
faults/telemetry/
├── proxy.py                       # 透明 HTTP/SSE capture proxy
├── responses.py                   # Responses state chain 重建
├── extract_hidden_states.py       # 离线 hidden-state extraction
└── run_fault_with_capture.sh      # 启动 proxy 并运行一个 fault condition
```

一次 capture 的输出大致如下：

```text
results/telemetry/<run-id>/
├── manifest.json
├── calls.jsonl
├── proxy.log
└── calls/
    └── 000012-call_<uuid>/
        ├── request.json
        ├── request.body
        ├── request-body.json
        ├── request-forwarded.json
        ├── response.json
        ├── response.body
        ├── response-events.jsonl
        ├── response-chunks.jsonl
        └── call.json
```

`request.body` 和 `response.body` 是代理实际收到的 body bytes。`request.json`
会保留 header 名称，但会把 `Authorization`、cookie 和 API key 的值替换为
`<redacted>`。SSE response 另外展开成 `response-events.jsonl`；每个网络 chunk
的到达时间、offset、长度和 hash 则写入 `response-chunks.jsonl`。

## 1. 启动 vLLM

先按照 `llm_backends/qwen_vllm/README.md` 在 Rivanna GPU 节点启动 vLLM。默认
endpoint 是：

```text
http://127.0.0.1:8000/v1
```

Telemetry proxy 使用另一个端口，不需要重启或修改 vLLM：

```text
Codex → 127.0.0.1:8100 → 127.0.0.1:8000 → Qwen
```

## 2. 跑一个带 capture 的任务

在同一个 GPU 节点的另一个 shell 中进入 `HealthAgentBench`：

```bash
cd HealthAgentBench

MODEL=qwen-coder \
JOB_NAME=ct-81-file-not-found-rep1-error \
TELEMETRY_RUN_ID=ct-81-file-not-found-rep1-error \
  bash faults/telemetry/run_fault_with_capture.sh \
    ct_abnormality_valid_81_a_1 file-not-found error
```

Control 组只改变最后一个参数和 run ID：

```bash
MODEL=qwen-coder \
JOB_NAME=ct-81-file-not-found-rep1-control \
TELEMETRY_RUN_ID=ct-81-file-not-found-rep1-control \
  bash faults/telemetry/run_fault_with_capture.sh \
    ct_abnormality_valid_81_a_1 file-not-found control
```

EHR-to-MEDS 使用同一个 wrapper：

```bash
MODEL=qwen-coder \
TELEMETRY_RUN_ID=ehr-meds-checksum-rep1-error \
  bash faults/telemetry/run_fault_with_capture.sh \
    ehr_to_meds_etl checksum-mismatch error
```

下面这些变量可以覆盖默认值：

```bash
VLLM_UPSTREAM_URL=http://127.0.0.1:8000
TELEMETRY_PORT=8100
TELEMETRY_DIR=/path/to/capture-directory
```

一个 proxy process 最好只对应一个 Harbor job。这样 `task`、`scenario`、
`condition`、`model` 和 `run_id` 会固定写入每条调用，不需要事后猜测某次请求
属于哪条 rollout。

如果不用 wrapper，也可以直接启动代理：

```bash
python -m faults.telemetry.proxy \
  --upstream http://127.0.0.1:8000 \
  --listen-port 8100 \
  --output-dir results/telemetry/manual-run \
  --metadata task=ct_abnormality_valid_81_a_1 \
  --metadata condition=error
```

然后把 Harbor 的 Agent endpoint 指向：

```bash
--agent-env OPENAI_BASE_URL=http://127.0.0.1:8100/v1 \
--agent-env OPENAI_API_KEY=dummy
```

代理会拒绝 WebSocket upgrade 并记录该尝试，使正式生成统一经过可审计的 HTTP
Responses 路径。`launch_qwen_coder.sh` 生成的独立 Codex provider 配置也显式设置
了 `supports_websockets = false`。

## 3. 先检查 capture

查看所有完成的 vLLM 调用：

```bash
jq -r '[.sequence, .method, .target, .status, .duration_seconds] | @tsv' \
  results/telemetry/<run-id>/calls.jsonl
```

只查看 Responses 调用：

```bash
jq -r 'select(.target | contains("/responses")) | .sequence' \
  results/telemetry/<run-id>/calls.jsonl
```

建议先选一个 sequence 做 hidden-state smoke test，再对整个 rollout 使用
`--all`。Qwen3-Coder-Next-FP8 很大，重复 forward 所需时间和 GPU 资源不能忽略。

## 4. 离线提取 hidden states

提取脚本使用 `transformers` 和 forward hooks。现有 vLLM SIF 已经包含
`torch`、`transformers`、`accelerate` 和 `numpy`，所以可以继续使用同一个
容器，不需要在登录节点安装第二套 CUDA 环境。

下面的命令需要在 GPU allocation 中运行：

```bash
cd HealthAgentBench

capture_dir="$PWD/results/telemetry/ct-81-file-not-found-rep1-error"
state_dir="$PWD/results/hidden-states/ct-81-file-not-found-rep1-error"
model_cache="$PWD/.cache/qwen-vllm"

module load apptainer/1.5.0

apptainer exec --cleanenv --nv \
  --bind "$PWD:$PWD" \
  --bind "$model_cache:/model-cache" \
  --pwd "$PWD" \
  --env HF_HOME=/model-cache/huggingface \
  --env PYTHONPATH="$PWD" \
  containers/vllm-openai-v0.29.0-cu129.sif \
  python3 -m faults.telemetry.extract_hidden_states \
    --capture-dir "$capture_dir" \
    --output-dir "$state_dir" \
    --model Qwen/Qwen3-Coder-Next-FP8 \
    --revision da6e2ed27304dd39abadd9c82ef50e8de67bdd4c \
    --layers quartiles \
    --probe-text $'\nCurrent agent state:' \
    --max-length 32768 \
    --device-map auto \
    --call-sequence 12
```

上面的 revision 是当前本地 `Qwen/Qwen3-Coder-Next-FP8` snapshot。正式实验时它
必须和启动 vLLM 时记录的 `MODEL_REVISION` 完全一致；如果以后更新权重，也要同时
更新这里，而不是继续复用旧 commit。

确认单次 forward 正常后，可以把最后一行替换为：

```bash
    --all
```

默认的 `quartiles` 提取 25%、50%、75% 和最后一个 transformer block。也可以
显式选择层：

```bash
--layers 0,15,31,-1
```

每个 `.npz` 保存：

- `token_ids`：实际用于离线 forward 的 token IDs；
- `layer_XXXX_context_last`：原 context 最后一个 token 的 hidden state；
- `layer_XXXX_probe_mean`：固定 probe tokens 的 mean-pooled hidden state；
- `layer_XXXX_probe_last`：固定 probe 最后一个 token 的 hidden state。

如果不需要固定 probe，省略 `--probe-text`，输出中只会有
`context_last`。所有向量都以 float32 保存，完整 per-token activations 不会写盘。

## 5. 输出与可复现性

Hidden-state 输出包括：

```text
results/hidden-states/<run-id>/
├── extractions.jsonl
├── prompts/
├── states/
└── metadata/
```

每个 metadata 文件都会记录：

- model、revision、tokenizer 和 tokenizer revision；
- PyTorch、Transformers、Accelerate 和 NumPy 版本；
- 选中的层、token 数量、probe 和 truncation 状态；
- rendered prompt 和 token IDs 的 SHA-256；
- Responses chain，以及所有无法精确重建的 item warning。

默认情况下，context 超过 `--max-length` 会直接报错，不会静默截断。只有确认
generation-time 也使用 left truncation 时，才应该显式加入 `--truncate-left`。

## 6. 精确性边界

代理保存的是精确的 HTTP request/response body。Hidden-state script 则是离线
重放：它解析 `previous_response_id`，拼接历史 input/output，再调用当前 Qwen
tokenizer 的原生 chat template。

因此，以下内容是精确记录的：

- Codex 发出的 request JSON；
- vLLM 返回的 JSON/SSE bytes；
- request order、response ID、时延和 streaming chunk；
- 离线重放使用的 rendered prompt 与 token IDs。

以下内容属于可审计的 reconstruction，而不是声称与历史 KV cache byte-identical：

- vLLM 内部 Responses-to-chat 的私有转换细节；
- server-side KV cache；
- generation 当时未保存的内部 activation；
- 无法映射到 Qwen chat template 的非 function tool item。

脚本不会隐藏这些差异；它会把 reconstruction warnings 写进每个 metadata 文件。
正式分析时应按相同 model revision、tokenizer、chat template、dtype 和截断策略重放。

## 7. 数据安全

虽然认证 header 已脱敏，request body 仍可能包含完整 prompt、工具返回和任务数据。
因此 `results/telemetry/` 应视为受控研究数据：不要提交到 Git，不要上传包含真实
患者信息或 access token 的 capture，并按项目的数据治理要求设置目录权限。
