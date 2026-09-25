# S2：单GPU owner线程与有界异步token桥

用户基线：5ae8a9b95394dfe4d8ea277dfed74b84acb7599e，S1已完成并提交。
助手实现及验证不代表用户源码已复现。源码按窗口教程交付，测试与文档直接迁移。

## 范围与数据流

新增tinyvllm/async_engine.py，包根导出AsyncEngine/RequestStream和三种可辨识错误。
复用Engine.submit/cancel/step/close；不改Scheduler、BlockManager、模型、attention或CUDA算子。

```text
asyncio loop: submit -> bounded slots + pending deque
                             | Condition wake
GPU owner: create Engine -> drain commands -> Engine.step -> publish events
                             | coalesced loop notification
asyncio loop: RequestStream -> TokenEvent ... -> CompletionEvent
```

Engine.from_config连同模型、runner和FI planner均在owner中创建。FI创建/执行/关闭stream约束保持。
start可共同等待同一次启动；绑定单个事件循环，拒绝跨loop调用。Engine不暴露给调用方。
worker空闲用Condition等待，活跃时每步前处理入队/取消；无轮询sleep、无每请求线程。

## 三种边界与所有权

max_requests：排队、执行、完成未消费的slot总数，默认64。
max_reserved_tokens：所有slot的min(prompt_len+max_new_tokens,max_model_len)之和，默认65536。
这是保守逻辑token预算，不等价于精确CPU字节或GPU KV占用。CompletionEvent的完整输出也计入该生命周期。
max_buffered_tokens：每个slot的待消费TokenEvent上限，默认128；另有一个terminal槽。
先检查prompt长度与额度再复制入队；词表/实际KV容量仍由owner上的原Engine校验。
slot仅保存传输状态，owner入队后清除prompt副本，不复制scheduler或KV所有权。

正常流按token顺序消费，独立terminal不会占据最后一个token位置。
worker退休(retired)且consumer结束(released)同时满足才释放准入额度。
完成但未读结果仍占额度；读取最后一个token尚未读取terminal也仍占额度。
输出缓冲溢出会丢弃该请求未交付缓冲并显式SlowConsumerError，owner下一边界取消它；不能当成功截断。
超出请求/token预算同步抛OverloadedError，不通过无限排队缓解。
所有状态共享同一Condition锁；dirty request IDs受slot数量约束，全局最多一个待执行通知回调。
asyncio.Event仅由事件循环set/clear；worker用call_soon_threadsafe合并通知。

## 取消、失败与关闭

RequestStream只有一个reader；通过async with包住消费，以便break、异常、处理任务取消时aclose。
anext等待自身被取消时也自动aclose。aclose立即放弃输出、唤醒reader，并单独登记cancel，不被满队列阻挡。
request.aclose不等待正在执行的GPU步；额度可能暂未恢复，紧随其后的新提交仍可能Overloaded。
若执行已自然完成，放弃消费不会再修改原Engine完成结果；不会给已结束请求额外制造ABORTED。
本层放弃后不再交付token/terminal。S3的断连处理会复用这个入口。

AsyncEngine.aclose永久停止接收、终结未完成消费者、请求owner停止，等待engine.close和thread.join。
重复关闭共享一个清理task；取消某个关闭等待者不会取消实际清理。不能强行中断正在运行的CUDA调用。
启动等待被取消同样触发清理；factory返回后仍在owner线程close。线程创建失败不会留下等待中的伪线程。
submit的TypeError/ValueError隔离为单请求错误；step故障广播WorkerError并停止整个worker。
保存完整异常链的格式化文字，不持有原异常/traceback对象，避免间接保留GPU张量。
服务关闭不覆盖已发布成功终态；slot中的错误与终态只交付一次。上下文已有异常时cleanup附注，不覆盖原异常。

## 使用规则

先async with AsyncEngine(config)完成启动，再submit token IDs。每个RequestStream用async with管理。
async for依次获得TokenEvent及最后一个CompletionEvent；非流式调用也应消费该流，不能等一个不消费的result()。
S1 Tokenizer可用于准备输入与对最后的完整IDs解码；本轮没有增量文本、HTTP或AsyncLLM包装。
S3负责有界CPU tokenizer/增量detokenize、HTTP JSON/SSE、断连取消与协议边界。

## 验证与证据

完整常规回归345 passed/68 skipped；新增CPU并发测试18项包含在其中。
GPU组合16 passed，含新异步三后端、S1文本、原2GiB bootstrap和FI专项。
常规skip来自显式GPU/模型开关，专项另行开启；以上不是重跑了所有历史GPU门禁的声明。
本轮结果见suite.log、gpu.log和verification.json；async-cpu.log记录较早15项专项通过。
CPU测试用真实Engine+fake executor，通过Event屏障验证动态混批、背压、双边回收、启动/退出竞态与通知上界。
真实模型测试分别对reference/cuda/flashinfer：同一owner内先用原token API取得期望，再对比完整异步流IDs与终态。
检查所有Engine调用同一worker线程、最终join完成、模型/KV/FI workspace weakref释放；不是跨后端位级一致承诺。
GPU组合另覆盖S1文本、原2GiB bootstrap和FI页表/生命周期门禁。未修改旧数值预算，未宣称吞吐/延迟提升。
Windows证据目录：.module_s2_async_20260922，含基线、候选、日志、教程回放和迁移哈希。
