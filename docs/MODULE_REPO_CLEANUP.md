# P0 后仓库整理：输入、长度、失败清理与无消费者状态

日期：2026-09-14。教程基线：用户 main `0b89ba8`。
助手在 `/home/ricardo/project/tiny-vllm-codex` 实现和验证；用户框架源码不直接修改。

## 修复与删除

1. Engine.submit：先验证完整 prompt/时间/ID，再发布命令及计数器。非法输入不进入队列。
2. Scheduler：max_model_len 限制 prompt + output，不以 KV computed_tokens 代替序列长度。
3. Scheduler.shutdown：永久停机，释放整个 Manager 所有权并关闭 open 事务；fail_step
   保留句柄检查后复用 shutdown。Engine 将不可恢复错误统一交给该入口，并清理 executor。
4. Engine.step：准备阶段的构建、能力检查、时钟获取共用 rollback；dispatch、执行、
   完成时钟、commit、完成事件的不可恢复失败都 fail-stop；idle 完成事件也覆盖。
5. Engine._fail_stop：先标记 engine failed，再分别尝试两侧清理；原始异常保持为 cause，
   二次清理错误使用 exception note 记录。任意硬件/进程级故障不承诺恢复执行。
6. Qwen3.load_weight：非 packed 权重必须 shape 完全相同后才能 copy_，仍支持 dtype 转换。
7. 删除 _written_refs 及其返回值、重复 _terminal_ids、尚未实现抢占的两个计数字段。
   preemption 的假测试改为真正验证跨 prompt/output 边界取 token 的测试。
8. 默认 EOS 始终 frozenset；工作负载 arrival_ns 同时验证非负和单调。

## 验证记录

- 用户 0b89ba8 原测试：159 passed，工作树干净。
- 新 Engine/Qwen3 案例与原有对应文件在旧源码运行：20 failed / 19 passed。
  参数化失败不是 20 个不同根因；覆盖输入、长度、清理、shape 和默认 EOS。
- 工作负载新增两例在旧源码：1 failed / 1 passed（负到达时间被接受）。
- 助手修复后全量：184 passed；compileall、git diff --check 通过。
- GPU 使用真实 Qwen3-0.6B、RTX 5060 Laptop、BF16，两请求 prompt=(1,4,7)/(11,12)。
  HF SDPA greedy token 为 [606,16]，tiny 一致；全词表 logits 在 atol=0.15/rtol=0.02
  下未完全通过，最大绝对差 0.1796875，平均绝对差约 0.03138884。此项不可写成 PASS。
  验证仅是两请求 prefill，未替代 P1 多步分页数值测试，也不是性能对标。
- 后续在未修改的用户 0b89ba8 上重跑同样 GPU 对照，基线与本轮输出逐元素完全相同
  （atol=0、rtol=0）。因此上述 BF16/HF 差异为原有现象，本轮未引入该样例数值变化。

## 测试交付

更新 test_engine.py、test_qwen3.py、test_block_pool.py、test_request.py；新增 test_workload.py。
只在助手验证后同步以上文件到用户 tests/。用户实施教程前，新回归失败是预期现象。
不复制生产源码，不自动 commit。用户实施后运行 pytest、compileall 和 diff check。

## 后续边界

P1 全局分页物理存储接线，完整覆盖混合批/重排/尾块复用/带历史的 causal mask，
同模块删除连续 KV 生产分支及相关兼容接口。先分析现存 BF16 对照误差，再设明确门槛。
P2 profile 再选择 metadata lowering、sampler、Python 调度开销等优化；不以源码行数变化
宣称 TTFT/TPOT 改善。目前没有相对 nano-vLLM 的同条件性能证据。

## 教程定位原则

窗口教程基于修改前行号；同文件从后往前修改，行号不会被之前的插入/删除扰动。
Engine: _fail_stop 460–473、_terminal_ids.add 420–422、step 246–355、
cancel 的终态集合分支 212–214、submit 147–199、终态字段 142、validation import 7。
Scheduler: fail_step 727–740、长度停止条件 702、EOS 初始化 171。
Qwen3: load_weight 非 packed 分支 822–826。
cache: complete 返回列表 465、返回类型 457、written_refs 构建 358/376–381/398、
公开 complete_writes 225–226、字段 169/181/190；修正两个 docstring 拼写。
RequestState: 两个抢占字段 93–94。基准校验器: arrival 条件 159。
