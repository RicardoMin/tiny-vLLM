# P0：唯一 BlockPool 与完整事务生命周期

## 2026-09-14 续写定位更新

用户已经完成名称替换，并继续手写 cache.py。最新已核对快照是 332 行，不再用
7185079 的行号指导剩余代码。BlockPool 和 BlockPlanTxn 基本完成，commit/rollback
已正确分开；不应继续重复旧的“commit 错误且无 rollback”诊断。

剩余改动仅涉及 cache.py 的实现：132 行 stable 改 stale；236 行注释 blcok 改 block；
251 行后增加 _last_completed_txn；278 行 _block_for_tokens 改 _blocks_for_tokens；
286 行 begin_plan 至 EOF 重新给出完整实现，随后补齐全部 manager 生命周期方法。
教程必须在当前窗口展开代码，不能只指向助手文件；以函数锚点配合上述快照行号。

已用此用户快照在助手工作区逐项应用续写改动，重新验证 159 passed，compileall 和
助手 git diff --check 通过。用户生产文件本轮未修改。两边现有 tests 内容一致，
本轮无需新增或重复迁移测试。用户五个生产文件仍为 CRLF，最终教程应包含统一 LF、
去尾空格和补文件尾换行的格式步骤，否则用户 git diff --check 会报行尾问题。

此前的设计说明如下，7185079 仅保留为模块起点记录。

更新：2026-09-13。教程起点为用户 7185079，不是助手 cb16aaa；替代前两版 P0。
生产代码只在助手工作区实现，用户按当前窗口教程复现。

## 实现与契约

改造 cache.py，以及 batch.py、scheduler.py、paged_kv.py、runtime/contracts.py 的
Block 命名，不新增 block_pool.py。Pool 管 id/generation/free list，Manager 管请求表
和有效长度，PlanTxn 管 staged metadata；复用已有 Scheduler，不改模型数值路径。

```text
FREE -> RESERVED --commit_before_dispatch--> ACTIVE -> release -> FREE
          |
       rollback -> FREE

PREPARED -> DISPATCHED -> COMMITTED
    |           |
 ROLLED_BACK   FAILED（engine fail-stop）
```

ACTIVE 不等于 K/V 已写完，complete_writes 才推进 valid_token_count。
generation 每次分配增加，rollback 不回退。BlockRef 是 pool-local 引用。
新块数量 = ceil((token_start + query_len) / block_size) - len(committed_table)。
无新增块仍需要写入事务，但不需要空 reservation。

block_size=4、已算 6、追加 3：先填尾块剩余 2 个位置，再加 1 块。staged table 长度为
3，dispatch 时 valid_token_count 仍为 6，complete 后为 9。物理编号无须连续；
slot = table[position // block_size].block_id * block_size + position % block_size。

## 相对上版的改进

- 不再规定大/小编号先发；测试验证逻辑顺序、容量和复用失效。
- 单 open reservation，不维护 txn_id 字典；容量预检在任何修改之前。
- required_new_blocks 与 begin_plan 共享 append 验证。
- teardown 无 txn 参数时也关闭 PREPARED metadata。
- _last_completed_txn 保留当前步骤的清理资格，拒绝旧完成句柄干扰后续请求。
- 无共享块，teardown 按请求释放，不用 seen 集合静默跳过重复所有权。

## 验证与交付

助手工作区全量 159 passed；BlockPool 核心 19 passed（13 个函数含参数化）。
test_block_pool.py 覆盖容量/输入原子性、reservation 身份、尾块/跨块、多请求隔离、
长度发布、回滚、generation、过期事务拒绝、teardown 和回收。
test_batch.py 验证 metadata 按计划顺序传递；test_paged_kv.py 用显式不连续块表测试
逻辑顺序；其余 scheduler/engine/executor/runtime 测试同步命名并保留公共语义。
旧 test_cache.py 已被替代。测试直接迁移，用户不手工复制长篇测试。

历史反测：内存加载 cb16aaa 的 cache.py，运行两个目标测试函数的 5 个 case，
3 failed / 2 passed。失败明确命中 PREPARED teardown 未终结，以及后续事务完成或
回滚后旧句柄未被拒绝；没有改写生产文件。这补充证明新测试能识别本次修复的问题。

P1 接全局 PagedKVCache、runner 和 reference attention，统一索引 dtype，对照
hidden/logits/逐层 KV，通过后删除 contiguous 生产路径。本模块不报告性能提升。
