> 2026-09-15 状态更新：用户 P1 已提交175975c。当前全仓审计见 ARCHITECTURE_AUDIT_20260915.md，
> P2-A实现契约见 MODULE_P2A_ADMISSION.md；以下P1以前的状态描述作为历史理由保留。
> P2-A采用保守逻辑准入与显式KV字节预算，自动profile定容留P2-B。助手验证不代表用户已复现。

# tiny-vLLM 架构复查

更新：2026-09-14；P0 历史复查如下，最新全仓整理依据用户 0b89ba8，见 MODULE_REPO_CLEANUP.md。
有效路线见 [TINY_VLLM_V1_REFACTOR_PLAN.md](TINY_VLLM_V1_REFACTOR_PLAN.md)。

## 保留与纠正

保留控制面/模型解耦、flat immutable batch、混合预算、已计算 KV 与采样 token 分离、
dispatch 前 rollback 和执行后 fail-stop。ModelRunner 已一次调用完整模型；逐请求
SDPA 是 reference attention 的限制，不能继续称 executor 逐请求执行 Transformer。

不再新增与 cache.py 平行的 block_pool.py。Manager 独占资源表，Pool 管 CPU 编号，
PagedKVCache 管设备存储；generation 是同池复用检测，不替代异步 GPU 生命周期。
先发大编号不是必要约束，物理编号顺序由 block table 隔离。本轮测试不绑定分配顺序。

## 本轮修复的真实问题

1. teardown() 无显式 txn 时，PREPARED reservation 已释放，旧事务却仍可读 metadata。
   对捕获的 open txn 总是写 FAILED 终态。
2. 旧 COMMITTED txn.fail() 能误清空后续请求。Scheduler 可能在 KV complete 后、请求
   状态提交时失败，所以仍允许当前 COMMITTED fail，但通过 _last_completed_txn 验证
   对象身份；下一次成功 begin 或 teardown 清除，不授权历史句柄清理。
3. required_new_blocks 与 begin_plan 复用 append 验证，避免 token_start/growth 漂移。

## 当前边界

每 engine 一个串行 scheduler、一个 manager/pool、最多一个进行中步骤；多请求并发
不等于多个 Python 调度事务并行。BlockRef 只在产生它的 pool 内有效，不验证跨池身份。
请求不共享 blocks，因此清理无需 prefix-cache refcount 或去重释放集合。
ACTIVE 表示所有权，成功 complete 才推进有效长度。
普通输入/容量拒绝应无部分修改；不承诺任意 MemoryError、进程崩溃、CUDA 设备故障
后的继续执行。现有 dispatch 提交边界尚未覆盖任意中途分配异常。

## 下一阶段的真实接线工作

以下是 P0 时的接线缺口，已由助手 P1 实现解决，用户是否实施须读最新源码确认。
当前有效接口和验收见 MODULE_P1_PAGED_REFERENCE.md。

当前 runner 仍用 ContiguousKVCache，分页存储未接模型。P1 需接容量、block table、
slots lowering、历史 KV、causal mask 和回收；改名不完成这些工作。
runtime slot_mapping 为 int32，而 PagedKVCache 要求 torch.long，需显式统一。
现有 bool(torch.any(...)) 范围校验接 GPU 后会同步，不能每层反复回读；应在 CPU
metadata 边界验证，再明确 reference/debug 与设备热路径的检查职责。
分页存储、gather+SDPA reference、kernel 直接按块读取是不同完成度，不混称性能完成。

## 简洁与性能

保留用户全面性能改进目标，固定 nano checkout 和协议，报告各负载结果及代码增量。
消除重复生产路径；不为目录整齐造 wrapper，不用缩排版或删除必要测试冒充精简。

## 2026-09-14 全仓整理结论

原源码全量 159 passed，但新增案例暴露了 Engine.submit 非法输入污染命令队列、
总序列长度多生成一个 token、失败清理漏掉 PREPARED/完成事件/时钟异常、Qwen3
非 packed 权重广播复制，以及默认 EOS 集合可变、基准工具接受 arrival_ns=-1。
修复和精简在助手工作区完成，184 passed；源码由用户按窗口教程实施。

删除 BlockPlanTxn._written_refs：Scheduler 忽略 complete_writes 的返回值，物理存储
写入由 slots 表示，不需要每步再保存一份写块列表。complete_writes 只提交有效 KV 长度。
保留 _last_completed_txn：Scheduler 在 KV complete 后仍可能失败，需要当前事务身份校验。
Engine 删除 _terminal_ids；cancel 已能根据待提交命令与 Scheduler 中的请求是否存在/
是否终态决定结果。_submitted_ids 暂留以禁止重复 ID，仍随历史请求数增长；未来若要
支持无限服务生命周期，应明确 ID 复用协议再处理，而非直接删去去重语义。

真实 Qwen3-0.6B / RTX 5060 Laptop / BF16 / 两请求 prefill 的 HF 对照 greedy 一致，
但 logits 最大绝对差 0.1796875、平均绝对差约 0.03138884，atol=0.15、rtol=0.02
检查未通过。保留此发现，P1 需要独立数值归因和验收；不据此推断分页或性能结论。
相同真实权重样例下，本轮修改前后 logits 逐元素完全一致；HF 差异在用户原提交已存在。
当前可见开销还包括未使用 metadata 的设备张量构造、逐请求 SDPA 与 sampler、逐层
KV 范围校验同步。这些需跟随 P1 接线和 P2 profile 分别处理，避免再造临时旁路。

## P1 实现结果（助手工作区）

全局分页 tensor 已取代逐请求连续缓存。ModelInput 删除未消费的设备表，只生成需要的
int64 write/read slots，一次生成后跨层复用；范围/唯一性/映射检查留在 CPU Runner 边界。
旧 PagedKVCache.logical_slots 与逐层 GPU 数值检查删除，底层张量接口不是外部验证入口。
BatchBuilder 验证完整 txn 请求列表与 append 区间，不能让未执行 append 被一起 complete。
Runner 不维护请求有效长度，不能独立重放历史 batch；可信 Engine 按当前事务串行派发。
Snapshot 并非可供外部任意伪造/重放的服务接口，异步执行前需重审其有效期和回收屏障。

RMSNorm 和 RoPE 低精度计算顺序已对齐 HF，原两个 BF16 短 prompt 的差异消失。
混合批仍可能由于 GEMM 行数不同产生差异：第一层 qkv 输入一致，输出 3/20480
元素不同，最大 3.814697265625e-06，发生在 attention 之前。它会在后续层放大；
不将跨 batch BF16 数值一致性标为完成，也不通过调宽原容差隐藏该现象。
同 flat batch、相同 SDPA 模式的独立 dense 存储 oracle 与 paged 5 步全量逐元素一致，
证明当前已测路径的分页读写未引入额外变化。详细记录见 P1_NUMERICAL_ATTRIBUTION.md。
