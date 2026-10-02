# tiny-vLLM

一个以简洁代码实现高性能推理的学习型引擎，当前支持单 GPU、dense Qwen3。

已实现分页 KV、连续混合批处理、chunked prefill、CUDA attention、采样与 EOS、文本 API/CLI，以及有界异步请求和 HTTP/SSE。模型计算与调度、缓存、协议层分离，后续按实际需求扩展模型。

项目仍在开发；v0.1 的目标是在明确的共同负载中，性能至少比肩启用 CUDA Graph 的 nano-vLLM，同时保持正确性与代码简洁。当前尚未达到这一发布目标。

## 使用

当前验证环境为 Linux/WSL、Python 3.12、PyTorch 2.9.1+cu128、NVIDIA CUDA 12.8。CUDA 后端需要匹配的本地 Toolkit 和 C++ 编译器，权重使用本地 Qwen3 目录。项目依赖见 `pyproject.toml`。

在已准备好匹配 PyTorch/CUDA 的环境中安装：

```bash
python -m pip install -e '.[kernels,server]'
```

文本生成示例：

`example.py` 展示 Python API；也可直接使用 CLI：

```bash
python -m tinyvllm \
  --model "$HOME/models/Qwen3/Qwen3-0.6B" \
  --attention-backend cuda \
  --prompt "Explain paged KV caching briefly." \
  --max-new-tokens 64
```

追加 `--stream` 可以即时打印文本；此时只接受一个 `--prompt`，也支持 `--messages` 聊天输入。

服务入口集中在 `tinyvllm/serving/`。公开 Python API 保持 `from tinyvllm import LLM, AsyncLLM`：
`LLM.generate/chat` 返回完整结果；`AsyncLLM.generate/chat` 是异步完整结果，`stream/chat_stream` 是请求上下文中的增量文本流。
流先产生 `TextDelta`，最后产生一个 `TextOutput`；提前退出应使用 `async with`，及时取消请求并回收 KV。

启动 HTTP 服务（默认仅监听本机）：

```bash
python -m tinyvllm.server \
  --model "$HOME/models/Qwen3/Qwen3-0.6B" \
  --attention-backend cuda
```

另一个终端请求聊天流：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"简述 KV Cache。"}],"max_tokens":64,"stream":true}'
```

也支持 `/v1/completions`、`/v1/models`、`/health`。本项目的 SSE 子集在结束帧中提供 `finish_reason` 和 `usage`，随后发送 `[DONE]`；客户端必须检查 `error`，不能只凭 `[DONE]` 判定成功。服务端提供请求/输出容量限制、超时和断连取消。

本地多轮聊天使用 `python -m tinyvllm --model "$HOME/models/Qwen3/Qwen3-0.6B" --attention-backend cuda --chat`；连接已启动的 HTTP 服务使用 `python -m tinyvllm.chat`。两种入口支持 `--system`、`/help`、`/reset`、`/exit`，生成期间 Ctrl-C 取消当前轮；失败或取消不加入历史。历史只在客户端内存保存，每轮重新提交，尚无跨轮 KV 复用。

尚不支持 TP/多 GPU、量化、多模态、工具调用或投机解码；HTTP 接口只实现明确的文本请求子集，不声明完整兼容其他服务的全部参数。

## 与 nano-vLLM 的开发阶段比较

2026-09-29，RTX 5060 Laptop 8 GiB，Qwen3-0.6B BF16，PyTorch 2.9.1+cu128。
nano-vLLM 固定为 [bb823b3e](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)。tiny 基于 b3f34f7 的引擎和当时本地测量工具，完整源码与结果身份在本地报告留档。

共同负载：4 个同时到达的请求，输入长度 64/512/64/512，各输出 32 token；temperature=0.6、无 top-k/top-p 截断、ignore_eos；KV page=256、token budget=256、max_num_seqs=8。每进程预热 1 次、测量 5 次，正逆顺序各执行一轮。只计 token 层推理，排除模型加载、tokenizer 与 HTTP。

| 实现 | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms/token) | TPOT p95 (ms/token) | 输出 token/s |
|---|---:|---:|---:|---:|---:|
| tiny CUDA | 87.08 | 153.40 | 21.73 | 22.71 | 156.98 |
| nano CUDA Graph OFF | 97.17 | 191.13 | 30.19 | 33.55 | 118.64 |
| nano CUDA Graph ON | 93.78 | 187.71 | 9.88 | 12.80 | 299.95 |

数字为 10 次回放中每轮统计值的中位数；4 请求的 p95 等于最慢请求，不构成可靠的服务尾延迟估计。nano Graph OFF 仍使用上游编译算子。双方采样实现不同，不要求输出 token 相同；固定输出长度控制工作量。

GPU 与桌面共享，时钟未锁定，启动前有约 12%–15% 外部利用率。KV 分配策略不相同（tiny 实际 504 MiB，nano 约 3.6 GiB），因此不能用这次数据比较等显存容量、最大并发或普遍性能。tiny 的首 token 在本场景较早，nano Graph ON 的持续生成与吞吐明显更好；项目尚未全面超过 nano-vLLM。

接下来先做小算子融合与每步 RoPE 数据复用，紧接固定 decode 缓冲区与 CUDA Graph、批量采样，再依据长上下文和并发测量调整 attention 与调度。

公开仓库保留引擎、构建配置、简短示例与此说明；开发测试、教程、测量脚本和原始结果在本地维护。
