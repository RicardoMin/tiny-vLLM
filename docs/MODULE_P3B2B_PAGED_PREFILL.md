# P3-B2b：可选的直接分页历史 Prefill

日期：2026-09-21。用户基线：7b5e8f9（P3-B2a 已提交）。
本模块在助手副本实现并验证；测试/文档迁移不代表用户已完成生产代码复现。

## 定位与边界

新增显式 attention_backend="flashinfer"，默认仍为 reference；既有 cuda 不变。
这里的 flashinfer 是混合后端配置名称，不意味着所有 attention 都调用该库：

| 工作 | reference | cuda | flashinfer |
|---|---|---|---|
| KV 写入 | index_copy | 自研融合 store | 同 cuda |
| Query=1（含单 token Prefill） | gather + SDPA | 自研 paged decode | 同 cuda |
| 无历史、多 Query | gather + SDPA | 当前 K/V + SDPA | 同 cuda |
| 有历史、多 Query | gather + bool mask SDPA | gather + lower-right SDPA | FlashInfer FA2 直接分页 |

reference 的具体首段是否 gather 仍由原 ModelInput 契约决定；它保持原实现作数值基线。
不新增第二个 Scheduler/Executor 或整模型 forward。PreparedBatch 和 BlockManager 所有权不变。
P3-B2a 的 fused lower-right SDPA 是可用后端和比较基线，不是需要删除的兼容垃圾。
本模块没有修改既有 .cu 内核，没有合入两个 scores 实验，没有新增面试专题。

## 数据流

1. Runner 在 CPU 校验 PreparedBatch 的页表、长度、slots、采样行。
2. ModelInput 为 flashinfer 省去历史 read_slots 展开和 reference_tables GPU 拷贝；
   单 Query 的 SingleQueryMetadata 继续构造。
3. Runner-owned PagedPrefill.prepare 仅选择 before>0 且 query_len>1 的请求，
   在 CPU 构造 int32 的 qo_indptr、paged_kv_indptr、paged_kv_indices、last_page_len。
4. 每个有选中行的 step 只 plan 一次，所有层复用。构造时另有一次 dummy plan 预编译。
5. 每层先融合 store，再分别执行 fresh、单 Query、历史多 Query 的 attention。
   Q 使用一次创建、跨层复用的紧凑缓冲区；cache 的 NHD view 为零拷贝，输出 scatter 回 flat rows。
6. finally 调 finish 清理 step 引用。clear 同步绑定 stream 后释放 wrapper/workspace/KV。

causal=True 使用右下对齐：第 i 个 Query 允许看到 key_position<=before+i。
last_page_len=((after-1)%16)+1，整页必须填16而不是0。
cache.view 的形状是 [num_blocks,16,num_kv_heads,128]，它不排序或复制历史 K/V。
“直接分页”省去的是历史 KV gather，不能解释为没有 Q 打包、输出 scatter 或元数据开销。

## 生命周期与支持范围

- 显式 FA2、NHD、head_dim=128、page_size=16、FP16/BF16、可整除 GQA。
- 当前后端绑定创建时的 CUDA stream；换流在 KV 写入前失败，不声称支持并发 forward。
- FI 0.3.1 的 scheduler.cuh 会写持久 pinned host workspace 并异步 H2D。
  同一 CUDA stream 只能排列 GPU 工作，不能防 CPU 提前覆盖 pinned 内存。
  plan 后记录 plan_done；下一次 plan 之前等待这个事件。不是逐层全设备 synchronize。
- finish 不等待整步 GPU；同流分配和使用使 PyTorch allocator 能安全处理 step tensor 释放。
  close 对绑定流同步，并在 finally 清引用；KV 分配失败也关闭已创建的 planner。
- 元数据转 int32 前检查长度/页号上界；保守限制整个 batch<=32768 请求。
- CPU 元数据避免从设备回读长度；不传该版本有局部变量缺陷的 max_token_per_sequence/max_sequence_kv。
- plan 不在 CUDA Graph 中执行。本轮不实现生产 CUDA Graph、异步 overlap 或跨流生命周期。
- 32 MiB用户workspace已覆盖本轮测试和负载，不构成任意请求规模/上下文的容量保证；执行失败显式上报。

## 内存预算

固定持久 GPU workspace：32 MiB float +8 MiB int +128 KiB KV lengths =40.125 MiB。
此外 wrapper 有8 MiB pinned CPU workspace。避免 backend="auto" 的额外设备分配。
bootstrap 在 KV 池容量检查前预留固定40.125 MiB，再预留紧凑 Q/O 最坏上限：

    2 * max_num_batched_tokens * num_attention_heads * head_dim * dtype_bytes

Qwen3-0.6B BF16、4096 token预算时为32 MiB，合计主要新增设备预算72.125 MiB。
该预算不覆盖全部模型activation峰值、allocator对齐、少量页表/行号等动态metadata和其他库缓存；
gpu_memory_utilization 仍不是运行期间显存硬上限。显式 KV 字节配置不会被静默缩小。

## 依赖与环境

可选依赖固定 flashinfer-python==0.3.1、nvidia-cudnn-frontend==1.13.0；
本机已正常安装，使用用户 .venv-tiny 测试，未使用旧 .local/flashinfer031 PYTHONPATH。
PyTorch2.9.1+cu128、Transformers4.51.3；pip check 无冲突。
CUDA_HOME=/home/ricardo/.local/cuda-12.8，CXX=g++-14，CC=gcc-14，MAX_JOBS=1，
TORCH_CUDA_ARCH_LIST=12.0 与 FLASHINFER_CUDA_ARCH_LIST=12.0 分别控制两套编译路径。
首次构造可能编译；它被放在计量窗口外。

当前本机重建可选依赖（不是要求重复安装）：

```bash
python -m pip install "setuptools>=77" wheel ninja
python -m pip install --no-build-isolation \
  "torch==2.9.1+cu128" \
  "flashinfer-python==0.3.1" \
  "nvidia-cudnn-frontend==1.13.0"
python -m pip check
```

## 验证结果

- 常规：301 passed /61 skipped；未开启GPU项的skip不代表它们已运行。
- CUDA/FlashInfer/真实模型/storage/bootstrap组合：86 passed。
- 原7步11行预算不变：mean_abs=.044783、max_abs=.281250、TV=.018456、greedy11/11。
- 长chunk对P3-B1，5步4行：mean_abs=.034624、max_abs=.125000、TV=.011505、greedy4/4。
- 保留原预算 mean<=.06、max<=.6、TV<=.02 和稳定margin的greedy断言；不放宽阈值。
- BF16/FP16 分散物理页对独立 FP32 oracle；未来token和跨请求隔离；页尾16/1。
- 每选中step只plan一次，Q/O跨层复用；连续不采样step、跳过无historystep、不回读中间输出后重规划。
- 换流拒绝、plan/run失败后引用释放、同步自身抛错仍清理；原KV实际对象释放保持通过。
- 原real model helper只调整第三后端的预期gather为0并复用数值断言，测试finally显式clear。
- CUPTI仍出现本机初始化警告：本轮时间来自CUDA Event/host clock，不声称获得NCU硬件计数器。

测试交付：新增 test_flashinfer_contracts.py、test_flashinfer_prefill.py，
更新 test_cuda_model.py。教程不要求手抄测试。

## 同次局部与整引擎测量

RTX5060 Laptop，未锁频，有明显轮间抖动；以下仅为本地诊断，不是正式框架对标。
微基准：Hq16/Hkv8/D128 BF16/page16，50调用warm Graph、9轮交替，CUDA Event中位。
包含KV读取/attention/output与Q打包/scatter，排除KV store、lowering、plan和host提交。
Graph仅是测量手段；prepare列另测整个prepare+finish主机耗时，包含事件保护和scratch分配。

| B | 历史 | Query | cuda μs | flashinfer μs | prepare host μs |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 27.441 | 16.834 | 1083.5 |
| 1 | 512 | 64 | 29.328 | 32.632 | 292.6 |
| 1 | 4096 | 64 | 187.000 | 108.643 | 373.8 |
| 4 | 1024 | 64 | 193.417 | 156.141 | 346.4 |
| 1 | 4096 | 256 | 757.033 | 643.297 | 322.2 |

同模型、每次仅一个executor/KV池存活，每个后端各预热负载一次，再计量一次，5轮交替。
两个负载各4请求/64输出；short为已有16/128/256/512长度profile，token预算128/chunk64；
long为16/512/1024/4096、token预算256/chunk128。KV池统一384块=672MiB。
TTFT/TPOT为每轮请求p50再跨轮取中位，不是合并所有请求的总体p50。
整引擎包含CPU规划、同步保护、输入转换和采样等正常成本；没有profiler。

| 负载 | 后端 | 吞吐中位 tok/s | 五轮吞吐范围 | TTFT p50中位 ms | TPOT p50中位 ms | 进程peak allocated中位 MiB |
|---|---|---:|---|---:|---:|---:|
| short-mixed | cuda | 103.440 | 77.172–107.276 | 55.072 | 24.252 | 1842.229 |
| short-mixed | flashinfer | 97.567 | 88.053–108.589 | 55.763 | 25.945 | 1882.647 |
| long-mixed | cuda | 41.216 | 38.960–41.533 | 210.395 | 33.343 | 1860.204 |
| long-mixed | flashinfer | 41.845 | 41.343–42.344 | 213.567 | 33.621 | 1888.519 |

结论：长历史attention路径有局部收益，但短整引擎更慢、长整引擎只小幅改善，
且TTFT/TPOT没有全面改善。保留显式选项，不更换默认，不宣称显存更省或全面加速。
固定workspace也可能超过小负载节省的gather临时内存；分页直接读不自动等于总显存下降。
下一阶段根据profile推进CPU metadata/采样/小算子融合与decode Graph，优先减少整步固定成本。

## 证据与后续

Windows .module_p3b2b_20260921 保存 baseline/generated、user_source_before.json、
suite.log、cuda-gates.log、real-gates.log、microbench.json、enginebench.json（含全部token/时序）、
measurement-environment.json（含源码hash/依赖）、measure.py 和验证/迁移记录。
旧B2a候选数值保留在上一模块，不把不同时刻的性能数据直接拼接成提升比例。
该可选后端落实后，单Query和历史多Query都有直接分页实现；P4/P5、v1接口与严谨多框架对标尚未完成。

参考固定版本源码：FlashInfer v0.3.1 的 flashinfer/prefill.py、
include/flashinfer/attention/scheduler.cuh；依赖核对见同证据目录 dependency_notes.md。
