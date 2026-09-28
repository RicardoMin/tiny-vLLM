# tiny-vLLM v0.1 交付路线

更新：2026-09-26。用户已提交 S2 与 v0.1 架构收敛（2b9914b）。本轮交付 S3 HTTP/SSE；助手验证与用户实施分别记录，详见 MODULE_S3_HTTP.md。
本文定义近期发布顺序；TINY_VLLM_V1_REFACTOR_PLAN.md保留长期技术路线。
pyproject.toml中的0.1.0目前只是开发版本号，不代表下面的发布验收已经完成。

## 调整原因与版本边界

现有token级引擎已足以承载应用层：统一混合batch、chunked prefill、连续调度、分页KV、
采样/EOS/取消/失败清理，以及本地benchmark。继续先完成所有P4/P5再补HTTP，会延迟验证
真实请求入口、排队、断连、输出与背压。因此先形成完整可用的v0.1，再按profile推进性能。
长期目标仍是在简洁代码前提下全面改善相对nano-vLLM的TTFT/TPOT/吞吐/并发/显存，
并与vLLM/SGLang严谨对比。v0.1功能发布与达到全部性能目标是两个验收维度，不混淆。

v0.1范围：单进程、单GPU、dense Qwen3、文本请求、单候选输出n=1，
同步Python API、CLI、异步多请求、HTTP非流式/流式、有限资源与明确关闭语义。
prefix cache/共享页、抢占重计算、TP、多GPU、量化、speculation、工具调用、多模态、
第二模型和所有kernel优化不作为v0.1功能发布前置；保持后续扩展边界。

## 已实现与尚未实现

| 层 | 当前状态 |
|---|---|
| Qwen3权重、RMSNorm/RoPE、模型forward | 已有真实模型门禁；不同计算形状/后端不承诺位级一致 |
| Engine/Scheduler/BlockManager | 已有submit/step/cancel/generate、批量和混合调度、保守准入 |
| KV与attention | 分页存储，CUDA单Query、融合store；可选FI历史多Query，默认reference |
| 采样与输出 | 已有greedy/temperature/top-k/top-p/seed/EOS和TokenEvent/CompletionEvent |
| 文本层 | S1已完成：tokenizer、chat template、完整detokenize、结构化输出、CLI |
| 服务并发桥 | S2已提交：单owner线程、有界token异步流、取消与关闭 |
| HTTP/SSE | S3本轮候选：aiohttp、JSON/SSE、ByteLevel增量解码；用户待按教程复现 |
| v0.1验收/正式对比 | S4待实现；现有benchmark主要是token层本地诊断 |

## S1：文本API与CLI（用户已提交5ae8a9b）

Tokenizer仅从本地目录加载，不执行远程自定义代码；raw prompt不套模板、不额外加special tokens。
chat使用Qwen3仓库模板，直接tokenize，添加generation prompt，默认enable_thinking=False。
支持system/user/assistant文本消息、末条user；不静默接受tool/image等未支持结构。
LLM.generate(str|list[str])一次调用原Engine.generate，按输入顺序返回TextOutput；
LLM.chat返回单个TextOutput。先完整tokenize全批，再由原Engine做原子校验/入队。
保存完整output_token_ids、prompt/completion/total counts和finish_reason；输出text只含新生成部分。
EOS从可见文本隐藏但计入原始tokens和usage；EOS/长度同时满足时对外分类为stop，ignore_eos时为length。
不做stop strings，不解析/丢弃thinking内容；enable_thinking=True时生成的推理文本保持原样。
本轮只完整输出，增量detokenize/网络流式留S3，不把逐token decode后拼接当作正确流式。
增加Engine.close与LLM上下文管理器，清理模型/KV/planner/RNG；close是永久终止，不是请求cancel。

完成标志：中文/emoji/空白round trip、模板一致、文本API与同后端token API输出ID一致、
原始token计数、输入失败原子性、关闭/失败实际对象释放和真实CLI JSON可用。

## S2：单owner工作线程与有界异步桥（用户已提交）

HTTP事件循环只处理连接和协议。GPU Engine在专用线程内创建、执行和销毁，
持续消费submit/cancel/shutdown命令，在step之间接收新请求；不为每个请求调用阻塞generate。
多请求GPU并发依靠现有batching；多线程并不等于多个线程同时forward一个模型。
FI绑定创建stream，必须连模型/runner也在owner线程内初始化，不能主线程创建后随意迁移。
等待队列和输出桥有上限；等待请求数/token预算不能只靠KV active准入限制。
不能让慢消费者阻塞GPU线程；输出溢出有明确取消/错误策略，且只终结一次。
CPU tokenizer/detokenizer先采用可控的串行所有权或有界执行器，确认线程安全与瓶颈后再加并行。
shutdown先通知/终结所有外部等待者，再关闭底层engine；错误不留下永远pending的future/queue。

本轮API为AsyncEngine与RequestStream，直接接收token IDs并交付TokenEvent/CompletionEvent。
max_requests和max_reserved_tokens覆盖排队/执行/完成未读的全生命周期，max_buffered_tokens限制每请求输出。
通知通过dirty IDs合并，最多一个待执行唤醒回调。终态独立于token缓冲；溢出失败而非静默成功。
RequestStream.aclose立即放弃输出并请求取消，不等待GPU回收；AsyncEngine.aclose等待owner完成清理并退出。
S3再将S1 tokenizer接入有界CPU处理与HTTP协议，不在S2重复实现AsyncLLM或网络协议。

完成标志：交错到达形成mixed batch、取消在步骤边界生效、队列溢出、慢消费者、
worker异常广播、重复关闭、所有等待者完成、blocks回收；无跨线程直接改scheduler状态。

## S3：HTTP、SSE与增量文本（当前交付模块）

薄协议层复用S2桥：GET /health、GET /v1/models，
POST /v1/completions与POST /v1/chat/completions的明确文本子集。
先支持n=1及已实现采样参数；拒绝未支持字段/能力，不宣称完整OpenAI协议兼容。
支持完整JSON与stream=true SSE；处理UTF-8字节片段、跨token字符、空文本delta、
最终flush、EOS/长度finish_reason、usage与唯一[DONE]，不能丢同一步的末token。
客户端断连/超时转换为cancel；有界背压、并发限额、错误响应、worker启动/关闭和本地默认绑定。
本阶段才增加必要server可选依赖与serve命令，不把网络库嵌入Engine/Runner。

完成标志：真实HTTP客户端并发请求、长短混合、断连取消、停止后无额外token、
流式拼接等于非流式结果（同配置/同轨迹）、慢客户端隔离、服务关闭无悬挂请求。

## S4：v0.1发布与基准收口

冻结代码/依赖/模型/workload与公开支持范围。干净环境安装、帮助与示例、异常/关闭/资源回收通过。
完整测试覆盖S1-S3和原核心数值门禁，记录GPU条件、跳过和失败，不用修改预算制造全绿。
先完成可复现的nano-vLLM共同能力矩阵，再接vLLM/SGLang，记录版本与功能配置差异。
引擎内计时和HTTP客户端计时分开；后者包含tokenizer/排队/传输/输出。
报告TTFT/ITL/TPOT分布、吞吐、完成率、SLO下可持续并发、KV/allocator峰值、等待队列与错误率。
包含长prefill、稳定decode、长短混合、并发阶梯、open-loop；prefix cache单列，不进行不公平混比。
基准可复现并如实揭示差距是v0.1发布条件；全面超过其他框架仍是后续优化验收目标。

## 性能路线如何继续

P4 metadata复用、融合、批量采样、decode CUDA Graph，P5抢占/公平性/前缀复用仍保留。
依据S2/S3真实profile选择最高收益项，必要时穿插一个性能模块；不以完成所有优化作为HTTP前置。
当前FI局部更快但短整引擎更慢、长负载收益小，保持显式选项；没有新的全面性能领先结论。

## 教程协作

助手先在tiny-vllm-codex实现并验证，窗口提供按用户实际基线的精确生产代码教程；
测试/文档验证后复制到tiny-vllm，用户在main自行复现并提交。普通模块不自动变成面试专题。
