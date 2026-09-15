# tiny-vLLM v1 架构决策与路线

更新：2026-09-14；用户仓库整理已提交 f929cda，助手已验证 P1 分页 reference，待用户复现。
替代旧 Stage 0–6、独立 block_pool.py 和 fake BlockTable 路线。

## 固定目标与职责

Batch=1 使用统一 batch 契约，当前只支持 dense Qwen3。追求简洁代码以及相对
nano-vLLM 的 TTFT、TPOT、吞吐和可持续并发全面改善；性能结论限于实测条件。

```text
Engine -> Scheduler + BlockManager -> StepPlan + BlockPlanTxn
       -> BatchBuilder -> immutable PreparedBatch
       -> ModelRunner -> ModelInput -> Qwen3 + AttentionBackend
       -> selected logits -> sampler -> StepResult -> Scheduler commit

BlockManager -> BlockPool       CPU id / generation / free list
ModelRunner  -> PagedKVCache    全局 device K/V tensors
```

Scheduler 管请求、预算和计划；BlockManager 独占请求 block table 与有效 KV 长度。
Pool 只管物理编号，不持有 CUDA tensor。Runner 消费快照并 lowering，不复制持久
请求所有权表。按逻辑顺序的 tuple[BlockRef, ...] 已足够，不新增独立可变 BlockTable。
容量与有效长度分开，dispatch 之后、complete 之前允许表已扩容而 KV 长度未推进。
模型只依赖 attention 契约；prefill/decode 可用不同 kernel，共用完整模型 batch 路径。

## P0：唯一 block 所有权与生命周期

直接改造 cache.py 并沿现有调度调用链统一 Block 命名，不新增第二套 allocator。
Pool 使用 FREE/RESERVED/ACTIVE，单 open reservation，全 batch 容量预检。
BlockRef 是 pool-local (block_id, generation)，每次分配递增 generation，rollback
恢复可用性但不回退 generation。ACTIVE 表示已分配，complete_writes 才推进有效长度。
无新增 block 的尾块写入仍需事务，无需空 reservation。块编号发放顺序不是契约。
teardown 关闭 open metadata；仅当前步骤句柄可以触发清理，旧事务不得影响后续请求。

验收：容量/输入失败原子性、多请求隔离、尾块/跨块 slots、stale ref、dispatch 前
rollback、执行后 fail-stop、过期事务拒绝及完成/取消回收。本阶段不宣称 attention 分页。

## P1：分页 reference 接线与生产路径收敛

MODULE_REPO_CLEANUP.md 已完成：submit 入队前验证、总序列长度限制、
Scheduler.shutdown 统一 fail-stop 清理、权重 shape 校验，以及删除无消费者的写块列表。
max_model_len 限制 prompt + output 总长度；KV valid length 仅统计已执行 token。
shutdown 是永久失败清理，不是恢复调度；不撤销已经完成步骤的语义。

1. Runner 创建全局 PagedKVCache，与 BlockManager 使用相同容量和 block_size。
2. CPU batch 边界检查 block 范围、slots 与表一致性及写入 slots 唯一性。
3. reference index_copy/index_select 统一 int64。ModelInput 只保留 input_ids、positions、
   sample_token_indices、write slots 和各请求 read slots；未消费的设备 metadata 不构造。
   优化 kernel 可以引入实际消费的 int32 表，转换必须明确。
4. 每层按 slots 写 K/V，按 block table 读取历史。reference 可先 gather 后 SDPA，
   chunked prefill 的 causal mask 必须带历史位置偏移。
5. 独立小模型/连续张量 oracle 对照 hidden、selected logits、每层逻辑 K/V。
   固定随机种子和独立数值基线。FP32 对 HF 完整前缀对照 hidden/logits/各层 KV；
   真实 BF16 用相同 flat batch、相同 SDPA 计算模式的测试侧 dense KV oracle 隔离存储变化。
   5 步含混合批、decode、重排、复用的 hidden/logits/各层 KV 已逐元素一致。
   RMSNorm/RoPE 已对齐 HF 舍入。真实混合批对 HF 分请求增量计算仍有超出原容差的元素；
   首差异在第一层 GEMM，源于批量行数形状。保留记录，不宣称跨计算形状位级一致。
6. 同模块验收通过后删除 ContiguousKVCache 生产引用、cache= 分支、executor _caches
   兼容属性及旧测试；oracle 留在测试侧，不依赖另一份生产 executor。

覆盖 Batch=1、混合批、纯 decode、partial prefill、不采样行、请求重排、跨块/尾块、
回收复用和失败清理。无逐请求完整 Transformer forward 循环。
gather+SDPA 证明分页语义，但仍复制历史 KV，不能据此宣称高性能 paged kernel 已完成。

当前接口：EagerExecutor/ModelRunner 显式接收 block_size、num_blocks、max_model_len；
构造方从 Scheduler 读取这些值。Batch 快照包含 block_size/num_blocks，Runner 检查几何
一致性；BatchBuilder 检查事务请求集合/顺序和每个 append 区间。Runner 不复制请求所有权。
设备 write/read 是内部接口，只消费已校验/转换的 indices；不每层回读 GPU 数值做范围校验。
调度器回收请求 blocks，executor.release 仅回收 RNG；全局 tensor 不因单请求结束而释放。

## P2：统一路径审计、基线与 profile

基准协议和 nano 固定版本现在即可准备；P1 端到端可运行后开始同条件实测。
先明确 BF16 与不同 batch/后端间允许的数值漂移及采样稳定性标准；不能用当前两个短
prompt 或同计算形状的位级一致结果代替跨负载数值验收。
清理重复校验、无用 metadata 与兼容层，保留公共语义。
负载包含长 prefill、稳定 decode、混合长短请求、逐级并发、open-loop 到达；重复前缀
单列并说明双方 prefix caching 设置。报告 TTFT/ITL/TPOT p50/p95、tokens/s、完成率、
KV 峰值内存。可持续并发须给定延迟 SLO、到达率和窗口，不能只比 max_active_requests。
说明 cold/warm、预热、重复运行分布和失败情况，定位 CPU/GPU 实际瓶颈后再优化。

## P3：有证据的性能 backend

候选为 varlen prefill、kernel 按表直接读取的 paged decode、Triton KV store；之后
评估 decode CUDA Graph、融合算子和 metadata 复用。选型依据硬件、数值与维护成本。
每项优化有 reference 对照、开关实验、端到端指标和生产代码增量。

## 后续与测试

第二模型复用 runner/attention 契约，不先造大型插件系统。Prefix Cache、共享/COW、
TP、量化、speculation 和异步 overlap 在基础数据面稳定后排期；异步前重审回收时机。
测试按 block、attention、runner、engine/scheduler、模型数值组织，不限制五个文件或
每功能一条；保留 sampling、权重、取消、失败等关键公共语义，删除私有布局绑定与重复。
助手验证后同步测试/文档，生产源码提供窗口教程，用户在 main 自行复现提交。
