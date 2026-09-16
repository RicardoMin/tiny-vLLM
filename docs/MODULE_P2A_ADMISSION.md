# P2-A：请求准入、生命周期与显式 KV 预算

更新：2026-09-15。教程基线：用户 main 提交 `175975c`。
用户已在ad7c81c完成本模块并将EagerExecutor改名Executor。2026-09-16复核：233 passed, 2 skipped；2项真实GPU门禁通过。原教程基线/验证记录保留。

## 本模块结果

- Scheduler 按请求最大实际 KV 需求做逻辑准入，物理块仍由 BlockManager 按 step 增量分配。
- 删除 KV fallback 二次计划、BLOCKED/BlockedReason/SchedulerResourceError；准入保证下容量不足是内部 invariant 失败。
- 统一的 Scheduler.validate_request 与 ExecutorCapabilities 将容量、词表、模型长度和块几何错误拦在入队/构造前。
- 请求 ID 只保留活跃集合；显式 ID 在 completion 交付后可复用；自动 ID 单调增加。
- 远期 heap 项使用 request_id + submit_seq + PENDING 判断有效性，取消残留项按阈值压缩。
- generate 是独占便利 API，整批原子准备后发布；未来到达使用 sleep，不忙轮询。
- execute 中排队的取消若与当步完成相遇，会在 ID 释放前退休，并重新连续编号未消费命令。
- Engine 仅保存错误摘要；shutdown 清空 CPU 请求状态；实际 KV 张量可释放，不被异常 traceback 长期持有。
- Engine.from_config -> bootstrap.create_engine：显式 CUDA dtype、EOS、长度、块几何与 KV 字节预算接通。

## 核心容量契约

设 P=prompt_len，N=max_new_tokens，L=max_model_len，B=block_size。
N=0 时无需 KV。否则：

```text
G = min(N, L - P)
KV tokens = P + max(0, G - 1)
blocks = ceil(KV tokens / B)
```

最后生成的 token 直接返回，不再执行下一次 forward，因此不写 KV。
当 P=L 时保留原语义：执行整个 prompt，不再输出 token，需求仍为 P。
活跃请求的最大需求之和不超过总块数；候选预算只在本次选取中计算，rollback 不留下额外额度状态。
没有新增持久请求所有权表，没有提前占有未来全部物理块。

这是重计算抢占前的保守策略：可能因较大的 max_new_tokens 降低并发，即使实际会提前 EOS。
有限工作负载可以完成；持续小请求到达仍可能延迟大请求，不能宣称无饥饿或提高性能已被证明。

## KV 字节预算与边界

本模块采用显式 kv_cache_memory_bytes，默认 2 GiB；kv_block_size 默认 16。
根据模型层数、KV 头数、head_dim、dtype 换算完整块，尾部不足一个块的字节不用。
gpu_memory_utilization 在权重加载后根据 CUDA 实际 total/free 检查总占用上限，包含设备上已有占用。
超预算直接报错，不静默缩小池。

当前 Qwen3-0.6B BF16：每块 1,835,008 bytes；2 GiB 得到 1170 块、18720 token 槽位，实际 2,146,959,360 bytes。
此比例不是运行时显存硬上限：尚未测量 activation/attention workspace，也没有 Graph。
自动峰值 profiling 与定容进入 P2-B；不要为了模拟峰值在生产里再引入一套特制 attention 或伪所有权路径。

## 教程文件范围

- 修改 tinyvllm/scheduler.py、engine.py、executor.py、config.py。
- 新增 tinyvllm/bootstrap.py。
- 不修改 BlockPool、PagedKVCache、模型数学或 AttentionBackend。
- 用户在 main 按窗口教程复现；不用应用补丁、复制助手源码或创建模块分支。
- 同一个文件按旧行号从下往上操作；片段采用正常空行、缩进、分段；不使用省略号替代核心实现。
- 测试由助手验证后直接同步到用户 tests/；窗口只给测试简报与命令。

## 验证与测试

默认回归：233 passed, 2 skipped in 5.95s；两项真实权重GPU门禁单独执行通过。
真实 Qwen3 paged/dense 数值门禁：1 passed。
新增真实 bootstrap GPU gate：1 passed；3个prompt长度1/80/130，各输出8 token，1170块全部归还。
注入最后一层失败后，cache/K tensor weakref均失效，约2GiB KV不再被Engine异常状态保留。
保持模型权重由Runner持有；这轮不新增完整模型卸载/服务关闭 API。

测试文件：

- test_scheduler.py：保留现有调度/事务覆盖，替换过时 fallback 测试，覆盖容量准入、跳过候选、rollback、堆与ID复用。
- test_engine.py：更新显式ID可复用语义，fail-stop后CPU状态应为空。
- test_engine_admission.py：原子提交、独占generate、非忙等、越界输入隔离、构造匹配、取消命令退休与实际KV释放。
- test_config.py / test_bootstrap.py：参数、预算换算、总显存占用、dtype/几何/EOS接线与提前拒绝。
- test_bootstrap_gpu.py：真实配置入口、混合生成、块回收和失败释放。

## 后续

P2-B 建立数值/测量协议、环境冻结、workload runner、逐token metrics及profile，评估自动显存定容。
保留已有BF16跨计算形状漂移记录；本模块通过不代表该数值议题已关闭。
之后按证据推进 paged/varlen kernel、metadata复用、融合、Graph、采样、重计算与prefix cache。

Windows复现资料：`.module_p2a_20260915/baseline`、`expected`、`line_map.json`、各文件patch。
这些是助手用于保证教程一致的材料；用户不需要照着patch学习或手工搬运测试。
