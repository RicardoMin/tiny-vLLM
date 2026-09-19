# P3-A：CUDA C++ 直接分页单 query attention

日期：2026-09-16。用户实施基线 main/fa202b6，目前仅开始修改 contracts.py。
助手在 `/home/ricardo/project/tiny-vllm-codex` 实现并验证；用户推理源码不自动同步。
用户在窗口按教程实施；测试和本文档直接同步，不能把助手验证写成用户已完成。
本轮完成CUDA性能优化，替代上一版较慢的CUDA候选；助手已重新进行数值和局部性能验收。

## 选择 CUDA 的原因与边界

用户希望学习 CUDA C++ 的线程组织、warp 归约、共享内存、同步和访存优化，因此本轮
自定义 attention 算子采用手写 CUDA C++。Triton 同样生成 GPU 机器代码；CUDA 提供更多
底层控制，不代表直接改用 CUDA 就更快，也不保证首次实现超过成熟 Triton 算子。
GEMM 等成熟算子继续使用 PyTorch/cuBLAS 等库，也允许评估 FlashInfer 等成熟 attention 实现。
性能仍用相同模型、dtype、负载、停止条件和预热后的普通 benchmark 判断。

## 模块范围

统一 Engine -> Scheduler -> Batch -> Executor -> Runner -> 一次 Qwen3 forward 不变。
每层先统一写入新 K/V；query_len==1 的请求直接查 block table 读 KV，其他请求继续 reference SDPA。
这包含 decode、单 token prompt 和 chunked prefill 的最后一个单 token chunk。
唯一 query 的位置为 context_len_after-1，因此能看全部有效历史，无需额外 causal mask。

显式 attention_backend=reference/cuda，默认 reference；CLI 同名选项，报告 config 保存实际值。
选择 cuda 后不支持设备/精度/配置则构造报错，不捕获运行错误后偷偷回退。
当前支持 NVIDIA CUDA FP16/BF16、head_dim 1..256、整数 GQA。CPU/reference 不加载扩展。
不增加 triton 后端兼容别名，也不保留仅为旧测试服务的 Triton 分支。

## 文件与构建边界

- `tinyvllm/kernels/paged_decode.py`：Python 包装器、按需加载和缓存构建结果。
- `tinyvllm/kernels/paged_decode.cu`：CUDA kernel、host 张量契约检查、torch.ops 注册。
- `runtime/attention.py`：CUDA 单 query 路由，与 reference 共用写 KV/分配输出流程。
- `runtime/contracts.py`：SingleQueryMetadata 和选择性 lowering，契约与上一版保持一致。
- Runner/Executor/bootstrap/config/benchmark CLI：接通 reference/cuda 选择。
- `pyproject.toml`：扩展构建辅助依赖与 .cu package data；分发包包含源码的检查已通过。

Python 通过 torch.utils.cpp_extension.load 构建共享库，再调用注册的 torch.ops。
显式选择 cuda 时在构造阶段按需编译，跨层和后续调用复用加载结果。编译产物进入
PyTorch 扩展缓存，不进入 Git；首次编译时间不属于稳定运行的推理耗时。
这个方案需要运行环境有匹配的 CUDA Toolkit，不是预编译 wheel 交付方案。

本机独立工具链位于 `/home/ricardo/.local/cuda-12.8`，nvcc 12.8.93 与 torch2.9.1+cu128 匹配。
系统原有 CUDA13.2 未修改。当前构建配置为：

```bash
export CUDA_HOME="$HOME/.local/cuda-12.8"
export CXX=g++-14
export CC=gcc-14
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST=12.0
```

12.0 是当前 RTX5060 的构建目标；更换 GPU 时按设备计算能力调整，不视为所有设备的固定配置。
MAX_JOBS=1 控制编译并行度，适配本机 WSL 内存；不会把推理限制为一个 GPU block 或线程。

## 数据与 kernel

SingleQueryMetadata 包含 query_rows、压紧 block IDs、table offsets、context_lens。
每 step lowering 一次，所有层复用。被选请求 read_slots=None，不展开历史 token 索引，
不 gather 历史 K/V。多 query 请求保留既有 read_slots 和 causal mask。
两类输出写入不重叠 query 行，选择由同一个 lowering 参数产生。
用户当前 contracts.py 的 read_slots 标注需为 `tuple[torch.Tensor | None, ...]`；
已完成的 metadata 设计可保留，不需要因语言变化重新设计其结构。

物理 slot = table[logical_position // block_size] * block_size + logical_position % block_size。
一个 CTA 处理一个请求的一个 query head；GQA 映射为 q_head // group_size。
QK 点积由线程子组协作归约，softmax 统计经 warp 归约得到，再通过共享内存提供给 V 累加。
block table 的尾部读取和 K/V 读取都必须受有效长度保护；不可只在最终输出时 mask。
传入 Q/output 的真实 stride，cache 每层布局为 [slot, kv_head, head_dim]。

online softmax 保留 FP32 最大分数 m、指数和 l、加权 V 累计 acc；新 tile 的最大值变化时，
旧累计量按 exp(old_m-new_m) 重标定。P 乘 V 前量化至输入 dtype，分母及输出累加仍使用 FP32。
分块遍历、归约顺序和精度转换属于数值契约；它不承诺逐元素复现 FlashAttention。
context_len 为运行时值，不为每个生成 token 编译一个版本。

本轮性能改动仅位于 paged_decode.cu，Python API、SingleQueryMetadata 和后续接线保持不变。
最终版仍按倒序64-token tile执行online softmax及重标定，保持原P量化策略；优化线程分工、
加载时机和数据搬运。没有采用实验中未通过整模型门禁的1024-token segment方案。
二次幂页用shift/mask代替通用除法；非二次幂页保留正确寻址。P/V分为8组并行累计，
对齐时使用16B向量读取及共享内存float4搬运，减少细碎加载；实际head_dim128有编译特化，
配合K/V 8-token预取提高加载与计算重叠。其他维度、尾部和未对齐地址走通用安全路径。
这里使用CUDA Core和FP32运算，不是自写Tensor Core算子，也未引入split-KV或GQA共享加载。
最终共享库的cuobjdump资源记录显示BF16/D128/二次幂页/向量预取特化使用116个寄存器、
LOCAL0、STACK0：未观察到local spill，但预取增加寄存器占用；不能仅凭无spill就推断occupancy最优。

host 入口检查 shape、dtype、device、stride 和 metadata 长度；索引值范围由现有 CPU batch
验证负责，不逐层回读 GPU 值。使用 CUDAGuard 选择设备，并在 PyTorch 当前 CUDA stream 发射，
通过 launch check 检查发射错误；不在算子中加入 cudaDeviceSynchronize。
当前模型仍以顺序 step 完成后回收 blocks，不能由支持 current stream 推导出已支持跨 step 异步回收。

## 本轮数值门禁与验证状态

最终CUDA常规回归：281 passed,22 skipped。GPU专项：27 passed，包含新kernel/模型、
host张量契约、stream依赖和原reference模型/启动门禁，保持exact paged/dense和实际KV释放。
真实Qwen3重新测得7步11采样行：max row mean_abs=.045703、max_abs=.281250、
max TV=.018456，greedy11/11，稳定margin9行。不是复用旧Triton的结果。
优化前后都没有放宽阈值，最终误差摘要与上一版通过的CUDA候选一致。

预设预算保持不变：真实每行 logits mean_abs<=.06、max_abs<=.6、分布 TV<=.02，
稳定 margin 行 greedy 一致。TV = .5 * sum(abs(softmax(actual)-softmax(reference)))。
这是本模块回归预算，不是普遍质量标准；HF 跨 shape、真实语料 log-likelihood 和长生成仍待补齐。
两 Runner 共享权重、各自维护 KV，使用相同 flat batch 和 teacher forcing，隔离 attention 后端变化。

测试文件使用 test_cuda_decode.py、test_cuda_model.py，替换原 test_triton_*；
test_bootstrap.py、test_benchmark_cli.py 验证新名称从配置传入并记录到报告。
独立 kernel oracle 使用已量化输入及 CPU FP32 数学：BF16 atol/rtol=.02，FP16=.003。
覆盖 MHA/GQA/MQA、Batch=1、物理块乱序、长度1/15/16/17/65/129/257/1025、
非连续 Q/output、head_dim6/32/64/80/128/129/256、NaN 尾部、非零层索引和大分数稳定性。
新增3个参数case：block_size3/head_dim80覆盖非二次幂分页与尾维，head_dim129覆盖非向量宽度维度，
Q/K/V/output均storage_offset1且仍contiguous覆盖16B未对齐地址；复用原FP32 oracle和预算。
小模型对照 hidden/logits/各层有效 KV；真实模型包含 mixed/reorder 和块复用。
每 step 仍断言一次完整模型 forward，CUDA 路径只让 multi-query 项执行 cache.read。
新增 host 张量契约用例及非默认 stream/event 依赖测试；这些 GPU 用例均通过环境变量显式启用。
通过打包检查仅证明 .cu 被纳入分发包，不代替编译、数值、stream 或性能验收。

## 本轮性能结果

RTX5060 Laptop、torch2.9.1+cu128，BF16/Hq16/Hkv8/D128；单位为微秒，越小越好：

| Batch | Context | 最终CUDA | 旧Triton候选 |
|---:|---:|---:|---:|
|1|16|2.676|3.133|
|1|128|4.900|6.691|
|4|128|7.229|8.985|
|1|1024|31.052|44.911|
|4|1024|43.702|59.184|
|1|4096|127.133|198.957|
|4|4096|340.931|467.891|
|16|128|21.426|24.161|
|16|1024|280.636|364.269|

方法：同一批tensor，预热后每个CUDA Graph含50次算子调用，交替先后顺序测9轮，
CUDA event耗时取中位数并折算每次。排除host dispatch、metadata lowering、KV写入和完整模型。
9个shape的中位数均优于旧Triton候选；长上下文样本波动明显，完整样本保留在报告中。
这只对上述设备、dtype、shape和比较对象成立，不表示CUDA语言普遍快于Triton。

同一引擎、现有短混合负载，各后端预热后交替测5轮、无profiler，每轮4请求/64输出token：

| 后端 | 五轮吞吐 tok/s | 中位吞吐 tok/s |
|---|---|---:|
|reference|88.89 / 87.54 / 85.43 / 86.06 / 86.77|86.77|
|旧Triton候选|96.72 / 96.69 / 92.47 / 95.27 / 94.91|95.27|
|最终CUDA|98.66 / 101.69 / 96.41 / 98.27 / 97.78|98.27|

同一组报告中，每轮请求级p50再跨5轮取中位数：reference TTFT66.08ms/TPOT29.71ms，
旧Triton64.51ms/26.02ms，最终CUDA62.49ms/25.15ms。这是“各轮p50的中位数”，
不是合并全部请求后的总体p50，更不是可靠的生产尾延迟统计。

实验通过运行时替换attention实现复用同一引擎；没有向生产配置重新加入triton选项。
这些是本轮局部性能验收，不能代替nano-vLLM/vLLM/SGLang完整对标，也不声称TTFT/TPOT和并发全面领先。
算子Graph用于稳定计时，不代表生产引擎已实现CUDA Graph；引擎端到端计时包含现有完整执行开销。
报告：Windows工作区 `.module_p3a_perf_20260916/final_microbench.json` 与 `engine_comparison.json`。
引擎结论只采用最终完整成功的engine_comparison.json，不使用此前cleanup报错的首轮文件。

## 成熟库候选的本轮评估

隔离环境试用FlashInfer0.3.1的FA2 Tensor Core与CUDA Core路径。局部算子表现有优势，
但同flat-batch真实模型门禁分别出现mean_abs=.066726>.06、TV=.027327>.02。
因此本轮没有引入项目依赖，没有通过放宽门禁接受候选，也不将实验包混入项目安装要求。
这不代表FlashInfer库存在错误；不同算法和精度顺序可能产生超出本项目当前预算的漂移。
后续可在明确数值标准与归因之后重新评估成熟库。

本轮读取的FlashAttention官方main `flash_attn_interface.py` 对paged KV要求page_block_size为256的倍数；
当前16-token页不能直接按该接口接入。这是被检查接口的限制，不是所有FlashAttention实现或未来版本的普遍结论。

## 历史记录：上一版较慢CUDA基线

上一轮CUDA候选通过数值门禁，但局部性能不足，用户要求本轮继续优化后交付。
其B1/L128、B4/L128、B1/L1024、B4/L1024耗时分别为15.48、16.62、136.70、155.09微秒。
这是旧版5轮局部计时，不能误当成当前实现，也不与本轮9轮结果拼成同一次实验。
原报告为 `.module_p3a_cuda_20260916/microbench.json`。
旧候选曾有TV=.020834超预算，修正归约/指数/除法策略后通过；本轮继续保留既定数值门禁。

## 历史记录：此前 Triton 候选

以下只描述上一轮 Triton 实现，**不描述本轮 CUDA 的结果**。
最初 32-token 顺序扫描、P/V 全 FP32 乘加通过独立 kernel 数学对照，但真实 BF16
首采样行 TV=.034029，超出原门禁 .02。greedy 相同不算通过。
同输入逐层对照显示当前自动 SDPA 使用 FLASH；初版若干层与手写 FP32 输出转 BF16 相同，
却与 FLASH 有差异。128-token/P 转低精度候选仍失败，TV=.030890。
最终 Triton 的 64-token 倒序/P 转低精度候选通过；没有放宽原预算。

历史 Triton 7 步、11 采样行结果为 max row mean_abs=.045703、max_abs=.281250、
max TV=.018456、greedy11/11，稳定 margin9行；历史回归281 passed/10 skipped，GPU专项15 passed。
这说明舍入和归约顺序对整模型有实际影响，不能归纳为“FP32一定更差”，也不能把旧通过推导为新实现通过。

历史无 nsys 短 A/B 的 reference 输出 tok/s 为78.23/76.85/83.80，Triton 为94.31/90.11/94.15。
它只是一组顺序运行的局部冒烟数据，不承诺稳定提速比例，更不能用于比较本轮 CUDA。
历史报告位于助手 `.local/p3a-20260916/{reference,triton}.json`。

## 用户完成源码后的验证

先配置上述编译环境，再运行：

```bash
cd ~/project/tiny-vllm
source .venv-tiny/bin/activate
python -m pytest -q

TINYVLLM_RUN_CUDA=1 TINYVLLM_MODEL="$HOME/models/Qwen3/Qwen3-0.6B" \
  python -m pytest -q tests/test_cuda_decode.py tests/test_cuda_model.py \
  tests/test_qwen3_gpu.py tests/test_bootstrap_gpu.py -s
```

通过 EngineConfig(attention_backend="cuda") 或 --attention-backend cuda 启用新后端。
用户需要在源码复现后自行运行门禁，不能在仅同步测试时预期全部通过。
默认 reference 保留作数值和 A/B 对照；不重复运行无关 GPU 工作负载。

## 后续路线

下一模块仍为 P3-B：多 query prefill/chunked prefill attention 与 KV store；之后 metadata
复用、局部融合和 Graph，再做调度抢占/复用及 v1 接口收口。自动 KV 定容与 HF 跨 shape 门禁仍未关闭。
单 CTA 在长 context 下可能并行度不足，GQA 还可能重复读 KV；后续按证据引入 split-KV 或共享加载。
完整 nano/vLLM/SGLang 对标放在 v1 收口，不凭短负载宣称全面领先。
