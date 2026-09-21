# P3-B2a：带历史 Chunk 的右下因果融合 Attention

日期：2026-09-20。用户源码基线 main / bd1c34a；P3-B1已提交。
本轮助手实现/验证完成，用户框架源码待按窗口教程复现。生产只改 runtime/attention.py。

## 范围与选择

历史多Query路径保持分页KV gather，改用torch2.9的causal_lower_right表达因果结构，
让支持的实际Q/K/V形状通过PyTorch内置FlashAttention计算。首段、单Query decode、KV store
及reference计算语义不变；不新增依赖、不修改已接受的CUDA源码。
P3-B2拆分为B2a（本轮融合Attention）与B2b（后续直接分页Prefill）；B2a不等于完全消除gather。

## 因果机制

历史before、本轮query_len、总长度after=before+query_len。第i个Query的绝对位置before+i，
只允许key_position<=before+i。即矩形Query×KV mask的右下对齐三角形。
普通公共SDPA的is_causal=True在非方阵上是左上对齐，不能直接代替。
causal_lower_right(query_len,after)是符号结构描述，不立即分配query_len×after GPU mask。
带历史路径传入该对象时is_causal=False；首段仍保持is_causal=True。

基类_make_mask保留原布尔mask构造；CudaAttention重写为符号bias。
公共_attend在gather及转置后把真实Q/K/V交给_attention_mask。
CudaAttention检查can_use_flash_attention(SDPAParams(Q,K,V,None,0.0,False,GQA))，
匹配当前torch内部bias分发逻辑，只查元数据、不读取GPU值或执行CUDA同步。
若不支持Flash，使用基类_make_mask创建一次布尔mask并在当前step的attention实例里复用。
这避免torch bias的通用fallback每层_materialize()；不是捕获执行异常后静默吞错。
仍为符号bias时逐层检查，覆盖stride或全局SDPA开关变化；一旦回到布尔mask，本step保持它。
不强制指定硬件shape白名单、不更改用户全局SDPA flags。
Torch已有再次能力检测，当前支持路径每层有两次CPU判断；整引擎测量包括该成本。

## 后端与显存证据

在本机BF16/Hq16/Hkv8/D128、before4096/query64代表场景：

| 路径 | CPU侧PyTorch算子记录 | Attention额外peak allocated |
|---|---|---:|
| P3-B1布尔mask | aten::_scaled_dot_product_attention_math + softmax | 171974656 B（164.008 MiB） |
| 本轮右下bias | aten::_scaled_dot_product_flash_attention | 18891776 B（18.017 MiB） |

测量基线包含预热后的cache、输入、输出和持久mask；只测_attend，排除KV写入、constructor和lowering。
原持久mask另占266240 B，本轮支持路径不分配该mask。KV池大小完全不变。
主要内存差异不能只归因于约0.254MiB mask本身：实际attention backend及其临时张量策略发生了变化。
以上是PyTorch allocator新增峰值，不是整引擎总显存、reserved显存或可持续并发结论。
CPU profiler可记录ATen分发；本机CUPTI初始化报告警告，没有GPU timeline或NCU硬件计数器证据。

## 数值与回归

- 常规：283 passed / 51 skipped。
- 新Chunk、Store、Prefill、Decode、真实模型及原storage/bootstrap组合：58 passed。
- 原7步/11采样行预算不变：max row mean_abs=.044943，max_abs=.281250，TV=.018456，greedy11/11。
- 新长Chunk对P3-B1 CUDA路径：相同模型与teacher-forced序列；前缀覆盖17/82/256/512/594/1024，
  本轮长度65/128/256/487/512及后续单token decode，batch重排；沿用同一固定预算。
  5步/4采样行：mean_abs=.029335，max_abs=.156250，TV=.015422，greedy4/4。
- 独立FP32 QK/softmax/PV oracle，分页15/16/17边界，BF16/FP16、GQA和小D6。
- future token和跨请求隔离；显式[2.5,3,3.5]用例区分右下/左上对齐。
- 禁止CausalBias._materialize，验证支持路径无dense mask；强制MATH后缓存bool mask跨层复用。
- 原reference paged/dense仍exact，真实KV对象释放通过；无放宽误差阈值。

## 同条件候选测量

RTX5060 Laptop / torch2.9.1+cu128 / BF16 / Hq16 Hkv8 D128 / page16，未锁频。
50次调用warm Graph，9轮交替顺序，CUDA Event取每调用中位数。测量含gather/Attention/output copy；
FI路径含Q index_select/output index_copy；不含KV store、lowering、plan或Python提交开销。
Graph只作为测量手段，不表示生产引擎已实现Graph。

| B | 历史 | Query | P3-B1 μs | 本轮bias μs | FI候选 μs |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 36.506 | 8.852 | 11.333 |
| 1 | 64 | 64 | 59.616 | 14.084 | 14.451 |
| 1 | 512 | 64 | 177.581 | 30.867 | 34.157 |
| 1 | 1024 | 64 | 342.570 | 43.491 | 43.793 |
| 1 | 4096 | 64 | 1728.047 | 167.974 | 105.635 |
| 4 | 1024 | 64 | 1397.901 | 176.408 | 148.499 |
| 1 | 1024 | 256 | 1248.362 | 162.189 | 158.419 |
| 1 | 4096 | 256 | 4783.613 | 578.071 | 475.311 |

同一引擎profile-mixed-qwen3-v1短负载，4请求/64输出token，各后端预热后5轮交替，不开profiler。
以下TTFT/TPOT是每轮请求p50再跨5轮取中位数；不是所有请求合并后的总体p50。

| 路径 | 吞吐中位 tok/s | 每轮TTFT p50中位 ms | 每轮TPOT p50中位 ms |
|---|---:|---:|---:|
| p3b1 | 120.170 | 49.038 | 19.940 |
| bias | 133.565 | 44.307 | 18.579 |
| flashinfer | 136.361 | 41.769 | 18.750 |

短负载不能推出生产尾延迟、所有上下文收益或与nano/vLLM/SGLang的排名；不与前一天数值直接拼接比较。

## FlashInfer候选：通过了什么，为什么没有本轮接入

隔离FlashInfer0.3.1 FA2 BatchPrefillWithPagedKVCacheWrapper，32MiB用户workspace，
NHD view直接使用现有page16 KV；只替换有历史多Query路径，保留自研decode和fresh路径。
原7步门禁通过：mean_abs=.044783、max_abs=.281250、TV=.018456、greedy11/11。
测试仅将该候选的预期gather次数改为0；其余数值/回收/flat-forward断言保留。
这修正了“此前FI decode失败意味着FI不能接入prefill”的错误推断；两种路径分别评估。

FI长历史局部更快；当前微基准各shape的plan host中位约55–189μs，另有32MiB用户workspace、
约8MiB GPU int workspace、8MiB pinned CPU、128KiB设备长度buffer。生产需Runner持有并跨层复用，
多Query行packing、int32元数据和显式关闭资源语义要单独接线。
实验FI仍构造现有read metadata（实际不gather），因此它不是充分优化的最终集成方案。
包还引入cuDNN frontend及其版本相关传递依赖。为当前Qwen3简洁目标，先交付无需新依赖且
短中历史收益明显的bias路径；FI不进入pyproject，不写成已采用成果。长历史分页收益留给B2b。

## 下一步

P3-B2b：直接分页prefill backend的生命周期、元数据复用、紧凑Q输入与长历史性能验收，
明确依赖与额外显存代价。当前不宣布P3多Query直接分页已完成。
后续P4 metadata/融合/Graph/采样、P5调度复用与公平性，再v1收口及完整多框架对比。

## 证据与工作流

Windows .module_p3b2_20260920 保存baseline/generated、suite.log、cuda-gates.log、
candidate_microbench.json、candidate_engine.json、dispatch_memory.json、summary.json及实验脚本。
用户教程只修改attention.py；测试与文档按约定迁移。没有新增简历专题。
