# P1：全局分页 KV 与统一模型路径

日期：2026-09-14。用户教程基线：main `f929cda`（干净，184 passed）。
助手源码：`/home/ricardo/project/tiny-vllm-codex` 工作树；用户生产文件由窗口教程手动复现。

## 实现契约

- Pool/Manager 是唯一请求块所有权与有效长度来源；Runner 只有一份全局物理 K/V。
- Executor(model, block_size=..., num_blocks=..., max_model_len=...)。参数取自同一个
  Scheduler。PreparedBatch 带 kv_block_size/kv_num_blocks，Runner 拒绝存储几何不一致。
- Builder 检查全部事务请求、顺序、token_start/query_len 和 context_len_before 一致。
- 每步 CPU 验证完整 query、block offsets、块范围/唯一性、slots 映射、采样边界及词表范围。
- ModelInput 只构造实际消费的设备 tensor，所有 reference indices 是 int64。
- 每层一次 index_copy_ 写整个 batch；各请求的 read slots 一次生成后跨层复用。
- 历史 chunk mask: key_position <= past_len + query_local_position；首次 prefill 使用 causal，
  单 token decode 无 mask。每 batch 完整 Transformer 只调用一次。
- 存储 read/write 内部调用只做 shape/device/dtype/layer 检查，不 bool(GPU tensor)。
  CPU 验证后才能 lowering；不得让外部直接将未经验证的 slots 送入 GPU 接口。
- 请求完成/取消由 Manager 释放 block IDs，executor.release 释放 RNG，不清零或释放全局
  tensor。读取范围只到有效前缀，不读取旧请求残留尾部。失败则丢弃整个 store 并 fail-stop。
- 删除连续缓存文件、模型 cache= 兼容分支、executor 的 model/device/cache 代理属性。
  模型必须收到 positions + AttentionBackend；单请求也是同一 batch 路径。

## 验证

- 全量：185 passed / 1 skipped（真实权重测试默认需显式 TINYVLLM_MODEL）。
- 真实权重 GPU 单独：1 passed；RTX 5060 Laptop、PyTorch 2.9.1+cu128、BF16。
- FP32 小模型对 HF 完整前缀：多请求、partial prefill、混合批、纯 decode、重排、
  非连续块、复用；比较 hidden、selected logits、每层逻辑 K/V，atol/rtol=2e-5。
- 实际 Engine 生成对 HF greedy；完成/取消回收、部分写入后失败清理、模型只 forward 一次。
- CPU 错误 metadata 在任何 lowering/KV 写入前拒绝；拒绝事务内多余请求和错误 append 区间。
- CUDA profiler 确认存储 write/read 没有 aten::_local_scalar_dense。
- RMSNorm/RoPE 的 FP32/BF16/FP16 独立 HF 算子对照逐元素一致。
- 真实权重同 flat batch，测试侧 concat KV oracle：5 步、9 个采样行，hidden/logits/28 层
  K/V 全部逐元素一致（atol=0, rtol=0），含请求块回收复用。
- 编译检查、git diff --check 通过。没有 TTFT/TPOT/nano 对标实测。

## 数值边界与实验失败记录

1. 上一轮两个短 prompt 的 BF16 HF 误差：RMSNorm 和 RoPE 舍入顺序两处均修正后，
   所有位置 logits 都完全相同；单改任何一个不足以解决。
2. 扩展到多步混合批，HF full-prefix 和 HF incremental 对照均出现个别 KV 元素超出
   atol=.15/rtol=.02（最大报告差 .25）。没有放宽容差将这两次实验记为通过。
3. 钩子归因：第二步第0层 QKV 投影输入5120元素完全相同；输出3/20480元素不同，
   最大3.814697265625e-06。同输入同权重整批M=5 vs 分请求M=1/2/2复现，分Q/K/V权重
   不复现，关闭 BF16 reduced precision reduction 不改变差异。首差异发生在 attention 前。
4. 为隔离存储，dense oracle 需要同样的 SDPA mode/strides。最初始终显式 mask 的 oracle
   选择了不同计算路径，BF16 hidden 不相等；对齐 mode/strides 后，保留独立 concat 存储，
   5步全量精确通过。此检查不替代独立 HF FP32 数值对照。
5. P1 已验证已测分页语义；跨 batch/后端 BF16 误差门槛与采样稳定性须在 P2 对标前明确。

## 测试交付

更新 test_executor.py、test_model_runner.py、test_paged_kv.py、test_qwen3.py、test_layers.py。
新增 paged_helpers.py、test_qwen3_gpu.py。删除 test_batched_executor.py、test_runtime_contracts.py：
旧连续缓存测试和冗余 lowering 测试已经由上述集成测试覆盖。保留 engine/scheduler/block/
sampling/checkpoint 等其他公共语义测试。助手验证后仅同步 tests 与文档。

## 教程锚点（修改前行号）

- cache.py：在 BlockPlanTxn._ensure_metadata_readable 原198行前添加 metadata 方法。
- batch.py：原298行前追加 geometry 构造参数；236行 if kv 内添加 append 检查；
  原196行前添加完整 txn 请求检查；原119行前加 geometry 字段。按从后往前顺序修改。
- paged_kv.py、runtime/contracts.py、runtime/attention.py、runtime/runner.py：替换全文件。
- qwen3.py：从后往前替换 LM.forward 847–860、Model.forward 543–651、Decoder.forward
  459–496、Attention.forward 277–374；删239行 num_kv_groups 和14行 contiguous 导入。
- executor.py：原79行 Executor 到 EOF 替换；删原13行 contiguous 导入。
- layers.py：替换原247–264的 _rotate（保留246行 staticmethod）及153–159的 RMSNorm.forward。
- 删除 tinyvllm/contiguous_kv.py；不需修改 Engine 或 Scheduler。

## 接下来的路线

用户实施后先重新验收源码。下一阶段准备可复现基准、GPU/CPU profile 和跨计算形状数值
协议，再据证据引入 kernel 直接按块表读 KV 的 backend。当前 gather 会复制历史 K/V，
不能用“分页已接入”宣称高性能 PagedAttention 已完成。
