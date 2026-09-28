# v0.1 核心回归测试

本目录按当前推理引擎的行为契约组织，不为旧模块教程、过渡实现或测试数量保留测试。
2026-09-26 收敛基于用户工作区实际源码快照；测试不是性能结果，CPU 测试也不能证明 CUDA 或真实模型正确。

## 运行

在仓库根目录激活项目环境后：

```bash
source .venv-tiny/bin/activate
python -m pytest -q -ra
```

默认只运行 CPU 契约和小模型参考测试。GPU 和本地模型采用显式 opt-in，机器是否安装 GPU 不改变默认测试集。

只验证本地 tokenizer 资源：

```bash
TINYVLLM_MODEL="$HOME/models/Qwen3/Qwen3-0.6B" \
python -m pytest -q -ra tests/test_text_model.py
```

真实 CUDA、FlashInfer 和本地模型：

```bash
TINYVLLM_RUN_CUDA=1 \
TINYVLLM_RUN_FLASHINFER=1 \
TINYVLLM_MODEL="$HOME/models/Qwen3/Qwen3-0.6B" \
python -m pytest -q -ra
```

CUDA 编译器路径、架构与依赖必须在运行前按本机环境配置。显式启用 CUDA 而设备不可用会报错；启用后缺少依赖不会静默跳过。该命令用于正确性回归，不用于声称 TTFT、TPOT 或吞吐提升。

## 保留的保障

| 责任边界 | 文件 | 防止的实际回归 |
| --- | --- | --- |
| 外部输入 | `test_inputs.py`、`test_engine_admission.py`、`test_text_api.py` | 参数越界、无效 prompt 污染已有请求、批量准入只成功一半、错误请求消耗 ID |
| 调度与请求 | `test_scheduler.py`、`test_engine.py` | future-arrival/FIFO 错误、prefill 预算预留失效、KV 承诺过量导致死锁、取消后输出、EOS/长度错误、零输出却执行 GPU 工作 |
| KV 所有权 | `test_block_pool.py`、`test_batch.py`、`test_paged_kv.py` | 陈旧块复用、跨请求别名、提前发布、失败未回滚、结束前后资源滞留、错误 slot/table 映射 |
| 异步桥接 | `test_async_engine.py`、`test_async_model.py` | 跨线程使用 Engine、未读结果提前释放额度、慢消费者影响其他请求、取消/启动/关闭挂起、异常丢失 |
| 执行与模型 | `test_executor.py`、`test_model_runner.py`、`test_qwen3.py`、`test_layers.py`、`test_checkpoint.py` | flatten/sample 行错位、HF 数学结果不一致、权重错误、tied head 重复存储、失败后持有模型/KV |
| CUDA/FlashInfer | `test_cuda_*.py`、`test_flashinfer_*.py`、`test_qwen3_gpu.py` | 数值/因果错误、strided/offset 存储错误、stream 次序错误、直接指针调用不安全、workspace 生命周期错误 |
| 文本与装载 | `test_text_model.py`、`test_bootstrap*.py` | tokenizer 编码策略漂移、chat template 重编码、显存预算超标、文本与 token API 不一致 |
| 测量与采样 | `test_benchmark_*.py`、`test_profiling.py`、`test_workload.py`、`test_sampling.py` | TTFT/TPOT 口径失真、事件缺失仍报告成功、EOS 序列异常、采集污染报告、随机流互相干扰 |

## 本次收敛原则

- `test_request.py` 与 `test_config.py` 合并为 `test_inputs.py`。删除手工给 RequestState 赋值后验证属性、简单 getter/枚举测试；状态转换在 Scheduler/Engine 真实路径验证。
- BatchBuilder 的测试只使用真实 BlockManager 事务。混合批次 flatten、partial-prefill 不采样、final-prefill 最后一行、page-tail 跨页映射继续覆盖。
- Scheduler 零输出/full-context/单步状态重复用例删除；Engine 的零执行断言和 Scheduler 完整 chunk→decode→回收路径承担保障。prepare/rollback/重试和到达顺序分别集中到单一场景。
- 删除独立 `test_model_registry.py`，对应删除未接入实际 bootstrap 的 registry 生产代码。Qwen3 装载与数学集成仍覆盖。
- 删除 RMSNorm/RoPE 的简易公式复述和 SiLU helper 用例；保留独立 HF oracle、多个 dtype、非零位置和整模型 logits/KV 对比，不放宽数值误差。
- 删除历史 `PreviousCudaAttention` 测试基线，将其 512-token chunk、1152-token prompt 场景合入当前 ReferenceAttention 对比，并覆盖 CUDA/FlashInfer。保留长上下文算子的独立 FP32 oracle 和真实模型固定输出阈值。新长场景的 baseline 已改变，须实际启用 GPU 验证，不把旧实验结果当作新门禁通过证据。
- Benchmark 删除特殊文件名、软链接、FIFO 等文件系统角落用例；保留真实源码内容/index 身份、精确指标、损坏事件序列、清理及失败报告。profile workload 的重复精确常量改为生成结果与已提交资源一致，并验证 manifest。
- 保留三个直接服务多条核心路径的 helper：`engine_helpers.py`、`paged_helpers.py`、`attention_helpers.py`。移除 fake executor 对生产参数校验函数的依赖；参考数学 helper 仍不得复用被测计算结果。

测试收敛数量不是覆盖充分性的证明。新出现的真实 bug 应优先补最小行为回归，不重新堆积内部属性检查。实际运行结果以本次审查日志及交付说明为准。

## 2026-09-26 本轮验证与实施依赖

助手验证工作区的源码候选通过完整回归：364 passed（RTX 5060 Laptop、PyTorch 2.9.1+cu128、FlashInfer 0.3.1、本地 Qwen3-0.6B）。包含长 chunk 对当前 reference 的新门禁；没有用新容差替代旧阈值。这不是性能对比数据。

测试直接同步，用户生产源码仍由用户按教程修改。新增 `test_qwen3.py::test_dimension_types_are_not_silently_coerced` 的三个入口分别验证 from_dict、本地模型目录、HF config 不接受被 int() 静默截断/转换的维度。在尚未落实 Qwen3Config.from_hf 修复的源码上，这三个用例应失败（DID NOT RAISE）；没有添加 xfail。其余测试对本轮用户基线兼容。


## S3 HTTP/SSE（2026-09-26）

用户已提交上述架构修复（2b9914b），先前三个配置类型用例的实施依赖已经解除。
本轮新增 test_streaming_tokenizer.py、test_server.py、test_server_model.py；
测试直接同步，需先按教程实现 tokenizer 增量接口、AsyncEngine 元数据与 server.py，
再运行完整套件。尚未复现时出现缺少 StreamingDecoder/server 的导入错误属于源码未完成，不能 xfail 掩盖。

安装可选网络依赖：`python -m pip install -e '.[server]'`。
不安装 aiohttp 时，两个网络测试文件会跳过；启用完整网络测试必须安装此 extra。

```bash
python -m pytest -q tests/test_server.py tests/test_streaming_tokenizer.py
TINYVLLM_MODEL="$HOME/models/Qwen3/Qwen3-0.6B" python -m pytest -q tests/test_streaming_tokenizer.py
TINYVLLM_RUN_CUDA=1 TINYVLLM_RUN_FLASHINFER=1 TINYVLLM_MODEL="$HOME/models/Qwen3/Qwen3-0.6B" python -m pytest -q tests/test_server_model.py
```

HTTP 协议测试运行真实 localhost TCP 客户端，但执行器为确定性 CPU fake；
发送背压/EOF 超时在真实 HTTP 请求中用受控 await 注入，不能当成实际慢网卡压测。
真实 GPU 测试分别覆盖 reference、CUDA、FlashInfer，同配置单请求的 JSON/SSE 文本、reason、usage 一致。
UTF-8 测试含跨 token 字符、最终不完整字节、special/padded ID 和真实 Qwen3 HF decode 对照。
这些是正确性门禁，不是 HTTP 吞吐、SLO 或跨框架性能排名。
