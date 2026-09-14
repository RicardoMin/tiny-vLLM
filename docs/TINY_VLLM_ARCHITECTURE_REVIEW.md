# tiny-vLLM 架构复查

更新：2026-09-13；依据用户 7185079、助手 cb16aaa 及本轮 P0 工作树。
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

当前 runner 仍用 ContiguousKVCache，分页存储未接模型。P1 需接容量、block table、
slots lowering、历史 KV、causal mask 和回收；改名不完成这些工作。
runtime slot_mapping 为 int32，而 PagedKVCache 要求 torch.long，需显式统一。
现有 bool(torch.any(...)) 范围校验接 GPU 后会同步，不能每层反复回读；应在 CPU
metadata 边界验证，再明确 reference/debug 与设备热路径的检查职责。
分页存储、gather+SDPA reference、kernel 直接按块读取是不同完成度，不混称性能完成。

## 简洁与性能

保留用户全面性能改进目标，固定 nano checkout 和协议，报告各负载结果及代码增量。
消除重复生产路径；不为目录整齐造 wrapper，不用缩排版或删除必要测试冒充精简。
