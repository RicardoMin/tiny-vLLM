# S3：HTTP、SSE 与增量文本

更新：2026-09-26。用户基线 main `2b9914b`，S2 与 v0.1 架构收敛已提交。
本模块先在 `~/project/tiny-vllm-codex` 实现验证；用户生产源码依照聊天教程复现。
测试与本文直接同步不代表用户生产代码已完成。

## 职责与数据流

`aiohttp handler -> bounded tokenizer worker -> AsyncEngine.submit -> owner thread -> Engine/Scheduler/Executor`

结果经过 `TokenEvent/CompletionEvent -> RequestStream -> request-local StreamingDecoder -> JSON/SSE -> client`。
HTTP 与 tokenizer 不直接访问 Scheduler 或 GPU。AsyncEngine 启动时发布实际 scheduler 的不可变 EOS ID 快照；
`is_running` 在桥的 condition 下读取启动与关闭状态，health 不依据陈旧的“曾启动成功”标记。

- aiohttp 负责 HTTP 解析、TCP 连接、异步写入。仅增加 `server` 可选依赖，不进入 Engine/Runner。
- 单 CPU worker 串行执行 HF encode/chat/decode；HTTP 取消不能抢占已经在原生库执行的 CPU 调用。
  shield 保持 future 存活，实际完成回调才归还 pending 额度、取走异常，避免取消洪峰造成无界排队。
- GPU owner 保持现有构造/step/close 线程归属。所有请求共享一个 AsyncEngine。
- Qwen3 ByteLevel 字节词表启动预热一次；增量 push 只查 bytes 表并推进独立 UTF-8 decoder。

## 服务与协议范围

`python -m tinyvllm.server --model ~/models/Qwen3/Qwen3-0.6B`

默认 `127.0.0.1:8000`，served model 名称默认 `Qwen3-0.6B`，可用 `--served-model-name` 显式设置。

| 接口 | 行为 |
| --- | --- |
| GET /health | owner 可用为 200，否则 503 |
| GET /v1/models | 返回唯一配置的服务模型名称 |
| POST /v1/completions | model + 单字符串 prompt；JSON 或 SSE |
| POST /v1/chat/completions | model + messages；JSON 或 SSE |

共同字段：model、stream、n、max_tokens、temperature、top_p、top_k、seed、ignore_eos。
chat 额外支持 enable_thinking；消息仅 system/user/assistant 文本，最后一条为 user。
只支持 n=1；未知字段拒绝。stop strings、logprobs、tools、images、stream_options、批量 prompt 不在本模块范围。
这是一组明确的 OpenAI 风格文本接口子集，不宣称完整兼容 OpenAI SDK 的全部能力。

SSE 每个事件为 `data: <JSON>\n\n`。chat 先发 assistant role，再发文本 delta。
成功时 final chunk 带 finish_reason 和 usage，随后唯一 `data: [DONE]`。
空文本 token 不必发送空 delta，但计入 usage；EOS、special 和 padded vocab ID 同样按原始 token 计数。
stop 集合来自 engine，不能拿 generation_config.json 中可能不同的列表替代。
thinking 文本原样输出，不拆分 reasoning_content。

SSE 已发头后不能改 HTTP 状态码：运行错误通过 error 对象与 DONE 结束；客户端必须识别 error。
网络已中断或 final chunk 已开始发送后发生异常，可直接断开而没有完整 DONE；客户端不能把残缺流视为成功。

## 字符正确性

一个 token 不一定是完整 Unicode 字符。ByteLevel 的 Unicode 字母表是 bytes 的可打印编码，
需要先逆映射再交给 `codecs.getincrementaldecoder('utf-8')(errors='replace')`。
每请求保存尚未凑齐的字节；最终 finish 明确 flush。这样真实 U+FFFD 与无效字节不会混淆。
special token 跳过，`<think>` 等非 special 保留；padded vocab ID 与 HF full decode 一致为空文本。
其它 decoder 或非 ByteLevel-compatible 词表在服务启动预热时报错；未来模型需要适配这一边界。

## 资源和关闭

三类上限独立：

1. HTTP active：覆盖生成 handler 的 body 读取、CPU、排队、生成及响应发送。
2. CPU pending：覆盖真正未完成的 tokenizer 工作，HTTP 取消后也不能提前减计数。
3. AsyncEngine：请求数、reserved token 和每请求 token 输出缓冲，沿用 S2 生命周期。

body 上限 1 MiB；过载 429，错误输入 400，超大 body 413，未发头超时 504，引擎不可用 503。
request_timeout 覆盖正常请求执行与发送；失败后的错误响应另有最多 1 秒补发窗口。
JSON 必须在 handler 内显式 prepare/write_eof，不能 return 后才让 aiohttp 在预算之外发送。
write 等待期间不会阻塞 owner；token 缓冲满时 AsyncEngine 将慢消费者失败并取消。
最终 HTTP handler 的 finally 总会 aclose RequestStream，取消在 GPU step 边界被处理。

`force_close` 只关闭 keep-alive；发生发送超时而要中止当前传输时使用 transport.abort，
避免 aiohttp 在 handler 返回后再次补写 EOF 导致超时边界失效。
成功 terminal 开始发送后不再追加第二组 error/terminal/DONE。

CLI 设置 handler_cancellation=True，真实 TCP 断连会取消 handler；自建 AppRunner 也必须设置。
shutdown 先停止接收新连接并设置 closing，等待已有 handler，再取消超时 handler，最后关闭 owner 与 CPU pool。
shutdown_timeout=5 是网络处理器的关闭等待窗口，不是进程退出的硬上限；正在执行的 CPU/GPU 操作仍需实际返回。
CUDA 不可恢复错误不会自动重建引擎；health 为 503，详细错误留服务端日志，客户端不收到内部 traceback。

## 回归与后续

test_streaming_tokenizer.py：字节边界、flush、special/padded ID、隔离状态、真实 HF 一致性。
test_server.py：真实 loopback TCP，CPU executor 下验证协议、输入、过载、断连、timeout、fault、关闭；
发送背压与 EOF 故障通过受控 await 注入验证，不能当作真实慢网卡压测。
test_server_model.py：真实 Qwen3，reference/CUDA/FlashInfer 三后端的 HTTP JSON 与 SSE 文本、reason、usage 对照。

实际运行数字写在本轮交付报告；这些属于正确性测试，没有测量 HTTP TTFT/TPOT 或可持续吞吐。
下一步 S4 为安装/支持范围/示例/发布验收及严谨共同能力 benchmark。全部性能优化不是 v0.1 功能交付前置。

参考：本轮实查 aiohttp 3.14.3 的 RequestHandler.finish_response/shutdown、StreamResponse.force_close
以及 handler_cancellation 实现；[aiohttp peer disconnection 文档](https://docs.aiohttp.org/en/stable/web_advanced.html#peer-disconnection)。
