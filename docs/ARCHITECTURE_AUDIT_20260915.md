# tiny-vllm 全仓审查与 v1 路线建议

审查时间：2026-09-15。用户源码 `/home/ricardo/project/tiny-vllm`，提交 `175975c`，main，工作树干净。
本记录不修改用户或助手 WSL 推理源码，不代表已实施修复；此前 P1 已由用户完成并提交。

## 结论

控制面和数据面的分离方向正确，无需再整体重写 BlockPool/Engine。
当前是统一 flat batch + 全局分页 KV + gather/SDPA reference，尚未具有直接分页 attention kernel。
存在已复现的调度进展、输入隔离、异常资源保留和事件交付问题，应在性能后端前修复。
“全面超越”保留为目标，验收限定模型、硬件、负载矩阵、内存预算和延迟 SLO；目前未运行对手性能对比。

## 当前调用链

Engine.submit -> AddCommand -> Engine.step -> Scheduler.apply_commands/begin_step
-> BlockManager.begin_plan -> StepPlan + BlockPlanTxn -> BatchBuilder.prepare
-> PreparedBatch -> mark_dispatched -> Executor.execute -> ModelRunner.run
-> ModelInput lowering -> Qwen3 flat forward -> 各层 AttentionBackend
-> selected hidden rows -> LM head -> TorchSampler -> StepResult
-> Scheduler.commit_step -> TokenEvent / CompletionEvent。

- Scheduler：到达/等待/运行队列、请求与 token 预算、chunked prefill、decode、公平性、终止事件。
- BlockManager：请求有序块表、有效 KV 长度、当前追加事务；BlockPool：物理 ID、generation、free list。
- BatchBuilder：CPU token/context/slot/采样行快照；Runner：检查快照并构造真正消费的设备张量。
- PagedKVCache：唯一的全局物理 K/V 张量；不复制 CPU 请求所有权。
- Qwen3/layers：模型计算与权重映射；AttentionBackend：KV 写入、attention 实现。
- Executor：Runner 与 Sampler 组合；Engine：命令和结果边界、失败后永久关闭。
- Config/Registry/Checkpoint 已存在，但 Config 尚未接成统一构造入口。

保留：Batch=1 同路径、一次 Transformer forward、QKV/gate-up packed projection、只对采样行计算 logits、CPU 块所有权、GPU 纯存储、dispatch 前 rollback/后 fail-stop。
正常请求结束回收块编号，不释放全局池。`_last_completed_txn` 仅保留最近一步，服务于 KV complete 后调度 commit 失败清理，不是无限历史泄漏。

## 已确认问题

### A1. 高优先级：KV fallback 只检查预选 items，可能误报 BLOCKED

位置：scheduler.py:495-507、532-536、806-869；engine.py:283-287。
请求行/token 预算先截断候选，随后 `_fit_kv_capacity(items)` 只缩减已选项，不补选遗漏候选。
复现：block_size=2，num_blocks=2，max_step_requests=1，prefill_chunk_size=1，三个 prompt 长度 3/2/1，max_new_tokens=1。
三步后 request0 有 2 个有效 token，request1 有 1 个，free_blocks=0。
下一步 request0 要新块，调度器报错；request1 可以在已有尾块追加 1 个 token 并完成、释放块。
仅在复现进程内旋转 ready queue 后，不修改实现即可完成全部请求，证明并非物理容量必然不足。
修复：将 KV 增量块预算并入候选选取，跳过不合适候选后继续补选，保留行/token/active-slot 与公平性约束。
测试：头部候选不可执行但后续尾块可执行；验证全部完成，不只检查某一步产生计划。

### A2. 高优先级：真实 KV 耗尽时没有前进策略

两个长度 3 的 prompt，max_new_tokens=1，block_size=2/blocks=2，每轮各 prefill 1 token。
两轮后各占满一块，双方还需一块才能产出，永久 BLOCKED；每个请求单独执行均能完成。
这是与 A1 不同的容量管理缺口。需要最小 admission/backpressure；v1 加重计算抢占以支持 KV 紧张下并发。
重计算需保留输出 token、RNG、first_token_ns，只清理 victim 的 KV 并重建已知前缀，不能重复交付 token。
单请求都无法容纳的负载应提前拒绝，避免重计算循环；记录抢占数和重算 token 数。

### A3. 高优先级：可提前拒绝的 token 越界会使整个 Engine fail-stop

位置：engine.py:164-170、315-324；runtime/runner.py:84-85。
submit 仅检查非负整数，vocab 上界在 dispatch 后的 Runner 才检查。
CPU 小模型复现 vocab_size=41：正常请求已生成首 token，再 submit([41]) 被接受，下一步使 Engine 永久失败，正常请求也不能继续。
修复：统一 bootstrap 传入不可变模型能力/限制，入口校验词表与长度；Runner 保留内部契约检查。
max_model_len、dtype、block_size、num_blocks、EOS 也应在构造时对齐，不拖到执行后才报错。

### A4. 高优先级：异常 traceback 持有 KV 张量，clear 不等于真正释放

位置：engine.py:146、428-448；runtime/runner.py:128-136、144-146。
`self._failed = cause` 保留异常 traceback，traceback 中 Runner/model/attention 栈帧又持有 cache 和中间张量。
注入最后一层异常，退出 except 并 gc.collect 后：runner.kv_cache is None，但 cache/K 张量 weakref 仍活。
只在复现进程删除保存的异常引用后，二者释放。
CUDA 小模型复现：KV 张量 3,145,728 bytes；删除诊断异常引用后 allocated bytes 下降 3,148,288。
修复：长期只保存字符串/结构化错误信息或 TracebackException(capture_locals=False)，首次异常正常向调用方抛出；关闭时清理无后续消费者的命令/请求状态。
勿通过直接清除 `_failed` 恢复生产 Engine；复现这样做仅为定位引用来源。
当前失败测试只验证指针 None/free block count，需增加实际弱引用释放检查。

### A5. 中优先级：历史请求元数据无限保留

位置：engine.py:144、183-199；scheduler.py:952-982、1020-1032。
100 个请求全部完成后，scheduler.request_ids=0，但 `_submitted_ids` 长度仍为 100。
100 个远期请求 submit/step/cancel/step 后，无活跃请求，但 `_pending_arrivals` 留 100 个条目直到远期到达。
这是 host 内存保留，不能与正常 GPU pool 常驻混称。
修复：明确 ID 语义，优先内部单调 ID + 仅维护活跃外部 ID；若坚持终身拒绝任意历史自定义 ID 重用，就必须承担历史存储成本。
未来到达 heap 使用惰性删除加周期压缩/头部清理；不要每次取消都 O(N) 重建。

### A6. 中优先级：generate 消费并丢弃其他请求的结果

位置：engine.py:343-383。
先 submit(A)，再 generate([B])；A/B 均完成，generate 仅返回 B，A 的 completion 已被消费，之后无法再取。
建议最小方案：generate 作为独占批量便利接口，入口拒绝已有未完成请求，并预检本次全部输入；流式混合提交统一用 submit/step。
现有 generate 对未来 arrival_ns 不等待，会忙轮询 step；基准回放器应感知下一到达时间并等待，不能用忙轮询伪造负载。

### A7. 高优先级的架构缺口：显存预算配置未生效

位置：config.py:15-21；scheduler.py:150-153；runtime/runner.py:36-44；paged_kv.py:53-65。
EngineConfig 仅定义/导出，没有生产构造消费者，gpu_memory_utilization 不控制实际 KV 分配。
Scheduler 默认容量=max_active_requests*ceil(max_model_len/block_size)，Runner 会物理分配全部对应存储。
真实 Qwen3-0.6B BF16：2*28*8*128*2=114688 bytes/token=112 KiB；block16=1.75 MiB。
64 请求×4096 token 默认上界意味着仅 KV=28 GiB，远超当前 8 GiB GPU。
修复：统一配置入口显式设置 device/dtype，加载权重后测峰值临时内存，预留 graph/activation，按剩余 KV 字节预算计算 num_blocks。
区分 GPU 存储容量、admitted active 上限、每步请求数、每步 token 数，不把它们联动成“并发数×最大长度”的物理预分配。

## 其他边界与清理项

- 当前 serial Engine 不是线程安全 service：submit 的 seq/ID/command 更新非原子，多 step 不受安全串行化保证。保留单 owner event loop，服务入口用队列；多请求 batch 并发不等于多线程调用 Engine。
- 现在单 CUDA stream 的顺序性支持先提交后续工作再复用块；未来多 stream/异步 overlap 必须用完成事件延迟回收与复用。
- SamplingParams 接受 temperature=1e-40，scores/temperature 在 FP32 溢出后 multinomial 报错。给采样规定数值域并稳定实现，禁止静默把随机采样改成 greedy。
- 模型签名仍有 attention=None/positions=None，但实际无可选语义：qwen3.py:281、486、710；应收紧。
- batch.token_request_slots 在生产中仅构造/存储，无消费者；可删除。num_decode_tokens 目前未使用，但可作为未来 backend 分组所需派生属性，按实际消费者决定。
- qwen3.py:687 错误消息打印 tuple(tensor) 而非 tensor.shape；错误诊断可非常庞大。
- tied embedding 模型仍先分配完整 lm_head 后覆盖 weight：qwen3.py:531-542。0.6B BF16 多出约296.75 MiB 短暂分配；启动入口可以顺手修复。
- load_model(dtype=None) 使用 PyTorch 默认 dtype 创建模型；checkpoint 的 BF16 不会自动成为计算 dtype。统一 bootstrap 必须显式设置。
- pyproject 要求 torch>=2.6,<2.7，而当前验证环境2.9.1+cu128；旧 L4/cu124 lock 不是 RTX5060 环境复现锁。
- benchmarks/tools/collect_env.py:42-47 只哈希 git status 文本，不能标识 dirty 文件内容。正式跑分使用干净提交，或额外记录 diff/untracked 内容 hash。

## 性能优化优先级

1. 直接分页 attention：attention.py:44-60 每层每请求 gather KV 并 SDPA；paged_kv.py:118-126 实际复制完整历史。
   现状 attention 调用数为层数×请求数，长上下文额外产生历史 KV 读写。选成熟 FlashAttention/FlashInfer 或小型 Triton backend，先验证当前 GPU、dtype、GQA、block size 和 chunk mask。
   保持一次模型前向，backend 对 decode 前缀和 prefill 后缀使用合适 kernel；混合批仍是一份计划/模型数据流。
2. metadata 复用：contracts.py:57-64 每步每请求构造 arange/div/index_select/read_slots；有历史 chunk 的 mask 大小 query_len×context_len。
   优化 backend 直接消费 block_tables/cu_seqlens/context_lens，不展开所有历史 token 的 read slots，也不物化因果矩阵。
   CPU pinned staging/device metadata buffer 复用，减少零碎传输与分配。
3. 小算子：layers.py:153、228、250、264。共享本步 RoPE gather/cast；融合 RMSNorm/residual、SiLU×up、QK norm/RoPE、KV store。
   当前 cos/sin 表已跨层共享，不存在28份表。先局部 compile 或小型融合实现，对齐数值契约后测收益。
4. decode CUDA Graph：固定地址、少量 batch buckets、安全 dummy slots，capture模型与适合的计算，CPU调度留外；图显存计入预算。
5. 批量 sampler：greedy 已批处理；随机采样逐行多 kernel，top_p 无 top_k 时完整词表排序。按采样参数分组/成熟采样 kernel；保留 request RNG 隔离和重排稳定性。
   executor.py:114 的 tolist 是本串行实现每批输出回 CPU 的边界，不应为了避免同步马上重写异步引擎。
6. Prefix cache：共享前缀工作负载收益高；完整块不可变共享、私有尾块、引用计数与受限淘汰优先，后续才评估复杂 radix/COW。
   需同步修改 BlockPool ACTIVE/引用计数语义与 runner.py:79-81 的全表唯一约束；共享只读合法，写 slots 仍必须独占。

同模型/dtype/unshared full KV，单位 token 字节数相同。显存领先应来自碎片、临时拷贝、共享与图预算优化；增加 KV 容量提升并发与降低常驻显存并非自动同时达成。

## 代码简洁性的量化

同一统计程序扫描双方生产 package 所有 .py，排除 tests/docs/benchmarks。
tiny：20文件，4736物理行，3793非注释/docstring代码行，1924条AST语句。
nano：19文件，1385物理行，1195代码行，958条AST语句。
行拆分影响很大，AST语句仅是辅助复杂度指标，不能视为功能/质量的等价单位。
结果说明“只比 nano 多一点代码”目前尚未做到；不能靠删除空行掩盖。
优先精简 scheduler 选取+fallback 双遍逻辑、无消费者字段、过度通用能力层和重复参数来源；保留关键事务/公共语义测试。

## 下一步模块路线

1. P2-A：资源与调度收敛。修A1/A3/A4/A5/A6，建立统一bootstrap/显存规划；对A2至少提供保证前进的 admission/backpressure。
2. P2-B：可复现 runner/metrics/profile 与 BF16 数值契约。现在建立 baseline，每模块复测，不等v1全部完成才量。
3. P3：paged decode + varlen/paged prefill backend + KV store，复用现有AttentionBackend。
4. P4：metadata buffer、小算子融合、decode Graph、批量采样；逐项消融记录代码增量和端到端收益。
5. P5：重计算抢占/调度公平性；prefix cache 在包含共享前缀的正式目标矩阵前补齐。单模型单GPU不引入复杂分布式系统。
6. v1收口：统一from_config入口、tokenizer/generate/streaming、有限队列与关闭语义、冻结环境与baseline；严谨对比nano/vLLM/SGLang。
7. 第二模型：小型 ModelSpec/InferenceModel 协议提供 vocab/max_len/KV几何、forward、compute_logits、权重加载。选择第二个dense模型检验无需修改Scheduler；不先造大型插件框架。

TP、PD分离、多节点、speculation、复杂层级缓存、全面异步overlap暂不进入核心路线。它们的维护和验证成本在当前阶段高于收益。

## Benchmark 契约

- token-level engine 和 HTTP/text service 分开测，不把tiny直接token输入与对手完整服务端数据混合排名。
- 固定权重/tokenizer/config/dtype/停止条件/负载；每个框架固定SHA与环境，记录kernel/graph/prefix配置，允许各自兼容依赖但完整披露。
- nano固定SHA bb823b3e06983d71485a8e1f23715ebd87d98ef8。该版 SamplingParams 明确拒绝greedy，不能将tiny默认greedy与nano随机采样对比；共同性能轨用相同正温度、无top_k/p限制、固定输出长度，质量轨另测。
- nano step仅返回完成请求，TTFT/ITL需要最小token观测插桩；保存插桩diff，不能把完成时刻当首token时刻。
- TTFT=t_first-t_arrival；ITL_i=t_i-t_(i-1)；TPOT=(t_last-t_first)/(n_output-1)，n<2为N/A；输出吞吐=实际交付输出token/测量窗口。
- 长prefill/短decode、短prefill/长decode、混合长短、不同并发和open-loop到达率；唯一前缀与重复前缀分组。
- 报p50/p95/p99、完成/拒绝/超时率、SLO内goodput和队列；可持续并发由内存预算、到达率、SLO及足够长稳态窗口定义。
- 报权重、KV池容量/实际占用、activation、graph、allocated/reserved/进程显存峰值；同KV容量开销与同总显存预算容量均可分别比较。
- 预热、编译/捕获与steady-state分开；多轮交错运行和变异范围；笔记本功耗/温度与WSL条件需记录，正式多框架比较在同一支持的Linux/GPU条件。
- 历史BF16混合batch对HF跨GEMM形状有超出旧容差的元素，之前归因首差异在第一层GEMM。现GPU exact门禁证明相同flat batch下分页存储等价，不证明跨后端质量契约已通过。
  保留FP32 HF独立基线，补长上下文/混合批的logit分布误差、greedy稳定性和语料log-likelihood；不能静默放宽容差或要求随机采样跨框架同seed同token。

## 本轮验证与复现

- 用户仓库pytest：185 passed, 1 skipped in5.53s。
- 真实Qwen3-0.6B GPU gate：1 passed in5.38s；5步9个采样行，hidden/logits/28层逻辑KV与相同batch dense oracle逐元素一致。
- git diff --check通过；用户仓库仍干净。
- 环境：PyTorch2.9.1+cu128、Transformers4.51.3、NVIDIA GeForce RTX5060 Laptop GPU。
- 工作区 `.audit_20260915/lifecycle_repros.py`：A1/A2/A4/A5/A6；`--cuda-retention`独立验证CUDA张量保留。
- `.audit_20260915/runtime_failures.py`：A3及极小正温度边界。
- `.audit_20260915/inventory.py`与inventory.json：双方代码计数、用户源码SHA256。
- 没有跑nano/vLLM/SGLang性能，因此不存在已领先或提速百分比结论。

## 对照资料

- nano源码固定版本：https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8
  已核对attention.py、model_runner.py、scheduler.py、block_manager.py、layernorm.py、sampler.py、llm_engine.py。
  已有FlashAttention分页读、varlen/chunked prefill、Graph、compile融合、前缀缓存及重计算；该版本仍prefill优先独立于decode，不进行tiny式混合计划。
- vLLM官方V1优化文档（2026-09-15获取main快照）：https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md
- SGLang官方调优文档固定快照：https://github.com/sgl-project/sglang/blob/1895cabfa8d0c7b171d72717812abe0c5d831791/docs/docs/advanced_features/hyperparameter_tuning.mdx
  取其内存分项预算、retract、graph、chunked prefill和prefix-aware调度机制；不照搬大卡建议的绝对内存/批量参数到8GiB笔记本。
