# P3-B1：融合 KV Store 与首段 Prefill 去 Gather

日期：2026-09-19。用户基线 main / 8653e1f，P3-A 已提交。
2026-09-20核对用户已按教程实施并提交bd1c34a；以下为本模块历史设计与验证记录。

## 交付与边界

- CUDA backend 每层一次 store_kv launch 同时写 K/V；reference 保留双 index_copy_。
- 无历史且 query_len>1 的 Prefill 直接使用当前层 k/v 切片进入 SDPA，不再 gather。
- 单 query（包括单 token prompt）继续使用已接受的 paged_decode；本轮未改该 .cu。
- 带历史的多 query chunk 仍 gather+带绝对位置偏移的 mask+SDPA。
- 统一 flat model forward 和 Engine/Scheduler/BlockManager 所有权保持不变。
- CudaDecodeAttention 更名为 CudaAttention；显式配置仍为 reference/cuda，默认 reference。
- 修复 pyproject 的 ["*,cu"] 拼写为 ["*.cu"]，wheel 检查包含两个 CUDA 源文件。
- 不新增依赖，不改变 KV pool 布局、不做跨层融合、不引入隐式 synchronize。

P3-B1 是完整可独立验证的数据搬运模块，不等于 P3-B 全部完成。
特别是有历史的 multi-query attention 还不是直接分页 kernel。

## 数据流与契约

CPU Runner 先验证 slot 范围/唯一性及 block ownership。ModelInput 每 step lowering 一次。
CUDA 路径 read_slots=None 的含义由 query_len 区分：

| 请求 | read_slots | Attention 数据来源 |
|---|---|---|
| query_len=1 | None | decode metadata + paged kernel |
| query_len>1, before=0 | None | 当前 k[start:end], v[start:end] |
| query_len>1, before>0 | 逻辑历史 slots | cache.read 后 SDPA |

reference 为所有请求保留 read_slots；它仍作为未更改的对照。
每条路径都必须先写持久 KV，首段不用读取缓存不代表后续步骤不用缓存。

## CUDA Store 机制

一个 CTA 负责一个输入 token，128 线程共同覆盖 Hkv*D；同线程复制对应 K/V。
目标 offset=slot[token]*(Hkv*D)+head*D+dim，使用 int64 地址。
输入各有独立 token/head/dim stride，尤其 V 来自 fused-QKV projection 的视图，
多 token Qwen3 Hq16/Hkv8/D128 的 V token stride=4096，有效 KV row 只有1024元素。

对齐路径每线程复制8个16-bit元素，用 uint4 搬运原始比特，不做浮点计算。
要求行内连续、token stride为8的倍数、行宽为8的倍数和16字节指针对齐；
不满足时走16-bit标量复制，支持非连续源和未对齐的连续目标。
输入与 cache 以及两个 cache 之间禁止共享 storage；普通 overlap 推断可能
对复杂 strided 视图返回 TooHard，因此按内部接口用 is_alias_of 保守拒绝。

不使用共享内存、原子操作或CTA同步；不同线程写不同元素、不同CTA依赖唯一slot。
保持 PyTorch current stream 与 device guard。检查 dtype/device/shape/stride 不读取GPU值。
raw op 的 slot 范围与唯一性仍是内部前置条件，不承诺任意外部输入安全。
通过 TORCH_LIBRARY_FRAGMENT 加入原 namespace；loader 在构造时一次编译两源文件。

## 验证

最终助手常规套件：282 passed，46 skipped（CUDA/实模 opt-in 未启用）。
最终 CUDA/实模/原始 storage/bootstrap 组合：52 passed。
KV Store 对 index_copy_ 原始位精确一致，覆盖NaN payload/符号零、FP16/BF16、
分散slots、其他层/未写区域、独立strides、奇数维度、未对齐和空输入。
非默认 producer stream → event → consumer，验证slot和输入在同流被更新后再消费。
Fresh Prefill 对独立CPU FP32 QK/softmax/PV oracle；未来token及跨请求不可见。
整模型保留原预算：max row mean_abs=.045703，max_abs=.281250，TV=.018456，greedy11/11。
reference paged/dense hidden、logits与各层KV仍exact；真实KV对象释放门禁通过。
CPU profiler检查未发现GPU标量读回；该环境报告CUPTI初始化警告，本模块没有GPU timeline/NCU证据。

## 本机测量

RTX5060 Laptop / torch2.9.1+cu128 / BF16 / Hq16/Hkv8/D128。
相同输入、独立cache，50次调用的warm CUDA Graph，9轮交替顺序，CUDA Event取中位数。
实际V采用packed-QKV stride。计时不含lowering、Python提交、其他模型层或网络。
Graph只用于测试，不代表生产引擎已经支持Graph。桌面GPU未锁频。

KV写入（μs，每层一组K/V）：

| 本步token数 | 两次index_copy | 融合store |
|---:|---:|---:|
| 1 | 3.523 | 1.406 |
| 4 | 3.916 | 1.892 |
| 16 | 4.404 | 1.807 |
| 64 | 4.050 | 1.583 |
| 128 | 5.059 | 1.745 |
| 512 | 13.197 | 3.583 |

单请求首段Attention（含KV写入、SDPA、output copy；μs）：

| Query数 | P3-A | P3-B1 |
|---:|---:|---:|
| 16 | 21.300 | 7.756 |
| 64 | 13.443 | 8.687 |
| 128 | 19.056 | 13.375 |
| 512 | 89.420 | 74.514 |

整引擎使用现有4请求/64输出token短混合负载，各后端预热，交替5轮，不开profiler。
初轮与最终轮原始数据都保留；结果波动较大，本轮不宣称已建立稳定的整引擎加速结论。
以下仅列各组5轮吞吐中位数，不作为泛化收益：

| 采集 | P3-A tok/s | P3-B1 tok/s |
|---|---:|---:|
| 初轮 | 77.222 | 70.764 |
| 最终防护版本 | 84.042 | 86.436 |

原始实验、完整日志和本轮源码快照位于Windows工作区 .module_p3b_20260919。
microbench.json、engine_comparison.json、*_first.json、suite.log、cuda-gates.log、wheel.log。
没有对本轮新增简历专题；ai-infra-interview专题是否新增/扩写必须用户明确同意。

## 下一模块 P3-B2

带历史multi-query attention：评估成熟varlen/paged/Tensor Core路径及16-token页适配，
先通过因果offset、GQA、混合batch、原实模数值预算，再同条件测性能。
当前torch causal_lower_right API可以表达右下对齐并尝试Flash后端，但也改变运算后端，
且fallback可能每层materialize mask；本轮仅阅读API，不作为已采用优化。
这部分完成前，不把prefill/decode都标为直接分页高性能backend。
之后进入P4 metadata/融合/Graph与采样，P5公平性/重计算及v1 API和严谨多框架对标。
