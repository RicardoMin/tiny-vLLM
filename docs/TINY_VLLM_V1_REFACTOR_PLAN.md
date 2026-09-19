# tiny-vLLM v1 架构决策与路线

更新：2026-09-19；用户 P3-A 已提交 8653e1f。
当前交付 P3-B1 融合KV写入与首段Prefill去Gather；助手验证与用户实施分别记录。
替代旧 Stage 0–6、独立 block_pool.py 和 fake BlockTable 路线。

## 固定目标与职责

Batch=1 使用统一 batch 契约，当前只支持 dense Qwen3。追求简洁代码以及相对
nano-vLLM 的 TTFT、TPOT、吞吐和可持续并发全面改善；性能结论限于实测条件。
自定义算子优先采用CUDA C++满足用户学习目标，GEMM等继续使用成熟库，并允许评估FlashInfer等成熟attention实现；不把CUDA语言本身视为性能收益，不要求为替换语言而重写所有算子。

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

当前接口：Executor/ModelRunner 显式接收 block_size、num_blocks、max_model_len；
构造方从 Scheduler 读取这些值。Batch 快照包含 block_size/num_blocks，Runner 检查几何
一致性；BatchBuilder 检查事务请求集合/顺序和每个 append 区间。Runner 不复制请求所有权。
设备 write/read 是内部接口，只消费已校验/转换的 indices；不每层回读 GPU 数值做范围校验。
调度器回收请求 blocks，executor.release 仅回收 RNG；全局 tensor 不因单请求结束而释放。

## P2-A：资源与调度收敛

对应 MODULE_P2A_ADMISSION.md，审计证据见 ARCHITECTURE_AUDIT_20260915.md。
按最大实际KV需求做逻辑准入，物理块仍按step增量分配；不维护第二份所有权表。
删除容量fallback二次计划与BLOCKED分支；不可容纳单请求入队前拒绝。
入口验证词表/上下文，构造检查块几何；active IDs有界、显式ID完成后可复用，远期heap按submit_seq防止旧项误激活。
generate独占且整批原子准备，未来到达等待；执行中取消碰到同一步完成时退休失效取消并重新编号未消费命令。
Engine仅保存错误摘要，shutdown清理CPU请求与KV，测试实际对象释放。
Engine.from_config接通dtype/EOS/块几何，采用显式KV字节预算(默认2GiB)与加载权重后的CUDA实际用量检查。
该上限不代表运行时activation/Graph峰值保证；自动profile定容进入P2-B，避免增加特制生产attention模拟器。
保守准入可降低实际提前EOS负载的并发；保证有限工作负载完成，不宣称持续到达时无饥饿。P5再用重计算与公平性提高利用率。

## P2-B1：本地基准回放与测量（已完成）

见MODULE_P2B_MEASUREMENT.md。复用Engine.step事件，不向推理热路径加入计时分支。
按预定arrival计算TTFT，以step返回后的host单调时钟观测token；保存完整token IDs和时间，
核对completion完整序列及EOS/长度语义。记录ITL、每请求TPOT、E2E、submit lag与吞吐，
每轮独立nearest-rank p50/p95，不平均percentile。
benchmark executor包装器在dispatch后execute前记录KV block高水位；真实负载前后同步，
记录PyTorch allocator baseline/peak与完整KV池分配。无每step额外CUDA同步。
内容hash包含Git index与实际文件，workload固定LF及hash；报告实际packages/模型hash/config。
P2-A名称统一为Executor；移除unused bootstrap import，修复shutdown计数器和拼写。

## P2-B2：Nsight Systems与可解释的时间线（已完成）

见MODULE_P2C_NSYS.md。只在benchmark侧增加可选NVTX与cudaProfilerApi capture。
默认不调用NVTX/profiler CUDA API；仅第0个measurement回放捕获，预热和权重加载排除。
measurement/step/execute-mode三级范围；静态标签不加入request id或变动shape。
observed时间仍紧跟step返回，位于NVTX pop之前；没有逐step额外同步。
异常时尝试drain与stop，保留原始错误。profile报告标diagnostic，不能用于正式端到端对比。
新增可重复长短混合profile负载16/128/256/512，各生成16，max_model_len使用1024。
用户已成功采集 .local/profiles/20260916-183722/tiny.nsys-rep，25步包含9个mixed。
该成功记录与此前中断的partial trace分开。现有短负载只作诊断，不作为完整v1性能结论。

## P2-B3：随模块的数值门禁与 v1 对标准备

现有测量基础已足够支持核心开发。每个性能模块补必要数值对照与局部A/B；
完整负载矩阵、nano适配和多框架排名放到v1收口，不作为推进P3的串行前置任务。
先明确 BF16 与不同 batch/后端间允许的数值漂移及采样稳定性标准；不能用当前两个短
prompt 或同计算形状的位级一致结果代替跨负载数值验收。
清理重复校验、无用 metadata 与兼容层，保留公共语义。
负载包含长 prefill、稳定 decode、混合长短请求、逐级并发、open-loop 到达；重复前缀
单列并说明双方 prefix caching 设置。报告 TTFT/ITL/TPOT p50/p95、tokens/s、完成率、
KV 峰值内存。可持续并发须给定延迟 SLO、到达率和窗口，不能只比 max_active_requests。
说明 cold/warm、预热、重复运行分布和失败情况，定位 CPU/GPU 实际瓶颈后再优化。P2-B1的显存profile只覆盖所跑负载，自动KV定容须另做峰值/余量实验；不能根据一个短smoke推算全配置安全容量。

## P3：直接分页 attention 数据面

P3-A（用户已提交）：保持Engine/Scheduler/BlockManager/统一模型forward不变。
按query_len==1选择直接分页kernel，包含decode和单token prefill；多token query继续SDPA。
SingleQueryMetadata仅传被选请求的query行、压紧block表/offset和有效context长度，
跨所有层复用；不为这些请求展开read_slots。每层统一写KV后读取，尾块mask必须覆盖表读取。
一CTA/请求/query head，显式实现线程分工、warp归约和共享内存同步；分段执行FP32 online softmax。
最终CUDA优化保留原接口与精度策略：二次幂页使用shift/mask、P/V分为8组并行累计、
对齐时使用16B加载及共享内存float4搬运，实际D128采用编译特化和K/V 8-token预取；
其他head维度、非二次幂页和未对齐地址保留通用安全路径。这是CUDA Core实现，不是自写Tensor Core kernel。
P乘V前转为输入dtype，FP32累加后写回原dtype。这是显式精度策略，不承诺位级复现FlashAttention。
显式reference/cuda配置，默认reference供数值和A/B基线；不自动吞异常回退。
Python loader按需编译paged_decode.cu并注册torch.ops；选择cuda时构造阶段完成编译，
避免在首个计量step引入编译工作。CPU/reference不编译扩展。kernel使用设备guard和当前CUDA stream，
检查host可见的shape/dtype/device/stride，不逐层读取GPU索引值或新增全局同步。
CUDA BF16/FP16，当前支持head_dim<=256、整数GQA分组；SingleQueryMetadata契约不变。
本机使用隔离的CUDA Toolkit12.8.93匹配torch cu128，系统13.2不变；g++14/gcc14、MAX_JOBS=1、
TORCH_CUDA_ARCH_LIST=12.0为当前RTX5060本地构建设置。分发包包含.cu源码，打包检查已通过。
kernel对独立FP32 oracle，整模型对同flat-batch reference；不把新门禁混称HF质量门禁。
test_cuda_decode.py/test_cuda_model.py替换原test_triton_*，保留host张量契约和current-stream/event测试；
新增block_size3/head_dim80、head_dim129和storage_offset1未对齐张量三项数值case，未放宽原阈值。
最终CUDA常规281 passed/22 skipped，GPU专项27 passed，真实7步11采样行重新通过原预算，
max row mean_abs=.045703、max_abs=.281250、max TV=.018456、greedy11/11。
本机9个shape的算子中位耗时均低于旧Triton候选；五轮交替短混合负载中位吞吐为reference86.77、
旧Triton95.27、最终CUDA98.27 tok/s，仍不是完整框架对标。最初较慢CUDA基线仅归档为历史。
FlashInfer0.3.1 FA2 Tensor Core候选mean_abs=.066726>.06，CUDA Core候选TV=.027327>.02，
虽局部快但原整模型门禁未通过，本轮不引入项目依赖；这不等于判定FlashInfer实现错误。
本轮检查的FlashAttention官方main paged KV接口要求页大小为256的倍数，不能直接套入当前16-token页。
后续成熟库接入仍需通过相同数值与性能验收，不为候选库放宽本轮预算。
详见MODULE_P3A_PAGED_DECODE.md。这一版未做split-KV/GQA共享加载，不保证长上下文最优。

P3-B1（本轮助手已验证、用户待复现）：融合K/V store成一次launch；无历史且多query的
首段Prefill直接读本轮K/V，省去read_slots和cache gather。参考backend、历史chunk的offset
mask与decode算法保持原状；类名CudaAttention覆盖写入/prefill/decode职责。
slot范围/唯一性由Runner CPU边界验证，kernel结构检查与source/cache storage隔离。
验证常规282 passed/46 skipped、GPU52 passed及wheel源文件。实模沿用固定预算。
微基准改善，短整引擎A/B噪声明显，无稳定整引擎收益结论。见MODULE_P3B1_KV_STORE_PREFILL.md。

P3-B2（下一模块）：优化有历史的多token/chunked prefill attention，评估成熟
varlen/paged/Tensor Core backend与16-token页适配；覆盖历史位置offset、GQA和混合批。
继续同一模型路径，保留reference；通过固定数值预算后再比较收益和代码复杂度。
只有P3-B2也落地直接分页路径后，才称prefill/decode均具备直接分页性能backend。
若decode长上下文出现并行度瓶颈，再添加有明确收益的split-KV；不先堆所有优化。

## P4/P5 与 v1 收口

P4：metadata复用、局部融合、decode Graph、批量采样；每项保留数值对照、消融、端到端收益与代码增量。
P5：重计算抢占/公平性，在共享前缀纳入正式目标矩阵前完成完整块复用、引用计数和淘汰。
v1：统一入口、tokenizer/streaming与关闭/背压语义，冻结版本后完整对比nano-vLLM、vLLM、SGLang。
第二个dense模型用小型ModelSpec/InferenceModel契约检验无需修改Scheduler，不先造大型插件框架。

## 后续与测试

第二模型复用 runner/attention 契约，不先造大型插件系统。Prefix Cache、共享/COW、
TP、量化、speculation 和异步 overlap 在基础数据面稳定后排期；异步前重审回收时机。
测试按 block、attention、runner、engine/scheduler、模型数值组织，不限制五个文件或
每功能一条；保留 sampling、权重、取消、失败等关键公共语义，删除私有布局绑定与重复。
助手验证后同步测试/文档，生产源码提供窗口教程，用户在 main 自行复现提交。
