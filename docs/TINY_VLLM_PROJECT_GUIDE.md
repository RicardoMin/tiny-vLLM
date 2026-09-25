# tiny-vLLM 项目入口与协作约定

更新：2026-09-23。以 V0_1_DELIVERY_PLAN.md 为近期交付路线，TINY_VLLM_V1_REFACTOR_PLAN.md 保留长期技术路线；审计证据见 ARCHITECTURE_AUDIT_20260915.md。

## 目标与状态

单GPU、当前dense Qwen3，学习简洁高性能推理框架。目标是在相同条件下改善相对nano-vLLM的TTFT、TPOT、吞吐、显存及可持续并发，并与vLLM/SGLang比较；代码量不显著增加。
不擅自改成单项快10%或接受数倍代码量。目前没有完成框架间性能比较，不能声称领先。

- 用户源码：WSL /home/ricardo/project/tiny-vllm，main，S1已提交5ae8a9b，生产执行器名Executor。
- 助手工作区：WSL /home/ricardo/project/tiny-vllm-codex，main；P3-A改用手写CUDA C++直接分页单query attention，用户源码通过窗口教程复现。
- 模型：/home/ricardo/models/Qwen3/Qwen3-0.6B；权重不进Git。
- 环境：PyTorch2.9.1+cu128、Transformers4.51.3、RTX5060Laptop；本轮pyproject对齐torch2.9系列；旧L4/cu124锁仅为历史环境，不能用于本机重建；runner保存实际packages与模型文件hash。
- P1全局分页KV reference已完成。P3-A改用显式cuda后端，单query直接读分页KV，fresh多query直用当前KV，带历史多query仍gather；默认reference。2026-09-21核对用户已提交P3-B2b；S1文本API/CLI已由用户完成并提交5ae8a9b；本轮S2在助手副本验证，用户待复现。
- 算子学习路线采用CUDA C++，便于理解线程、warp、共享内存、同步和访存；GEMM等继续使用成熟库，也允许评估FlashInfer等成熟attention实现。CUDA与Triton都生成GPU代码，语言选择本身不保证更快，收益须按同条件A/B验收。
- 编译工具链：本机隔离安装CUDA Toolkit 12.8.93于/home/ricardo/.local/cuda-12.8，匹配PyTorch cu128；原系统13.2保持不变。构建使用CXX=g++-14、CC=gcc-14、MAX_JOBS=1、TORCH_CUDA_ARCH_LIST=12.0；这些是当前RTX5060开发配置，不写死为所有设备支持范围。
- P3-A通过Python loader编译paged_decode.cu并注册torch.ops，显式选择cuda时在构造阶段按需加载；kernel使用PyTorch当前CUDA stream。CPU/reference无需编译CUDA扩展。
- P2-A：逻辑容量准入、入口限制、活跃ID/远期堆、独占generate、失败资源释放、显式KV字节预算与Engine.from_config。

- P2-B1：逐token观测、open-loop回放、完整结果验收、CUDA/KV峰值和源码内容身份；见MODULE_P2B_MEASUREMENT.md。

- P2-B2：用户已成功采集20260916-183722 trace；见MODULE_P2C_NSYS.md。
- 当前推进顺序：S1文本API/CLI → S2有界异步桥与GPU owner线程 → S3 HTTP/SSE → S4 v0.1发布与完整基准。P4/P5依profile穿插，全部优化不再是HTTP前置。详见V0_1_DELIVERY_PLAN.md。

## 协作规则

助手先在自己的WSL工作区实现并验证；推理源码仅由用户按当前窗口教程修改。
教程必须给用户当前基线行号、函数锚点、准确代码、目的与难点，不能只给文件链接或补丁。
代码用正常开发格式：合理空行、缩进、空格和分段，不挤成连续短行；一个模块有完整可验证成果。
同文件按旧行号从下往上操作。测试验证后由助手直接同步到用户tests/，窗口只给简报，不贴长篇完整测试。
架构/路线文档可同步；两边在main开发，用户自行add/commit，不再每模块建分支。
不得覆盖用户独立源码；测试同步不代表用户实施完成。

## 文档

- V0_1_DELIVERY_PLAN.md：v0.1范围、服务模块顺序和发布验收。
- MODULE_S1_TEXT_API.md：已完成的文本接口、输出及关闭生命周期。
- MODULE_S2_ASYNC_ENGINE.md：当前GPU owner线程、有界异步token流、取消和关闭契约。
- TINY_VLLM_V1_REFACTOR_PLAN.md：长期技术路线与职责。
- MODULE_P3B2B_PAGED_PREFILL.md：当前可选历史Prefill接入、成本、数值和限制。
- ARCHITECTURE_AUDIT_20260915.md：175975c全仓审计、复现证据和对标。
- MODULE_P3A_PAGED_DECODE.md：当前模块设计、数值预算、验证与实施基线。
- MODULE_P2C_NSYS.md：已完成Nsight教程、实际采集状态和用户结果解读。
- MODULE_P2B_MEASUREMENT.md：已完成的测量口径、验证与后续边界。
- MODULE_P2A_ADMISSION.md：已完成的容量/生命周期/预算契约。
- MODULE_P1_PAGED_REFERENCE.md、MODULE_P0_BLOCK_POOL.md、MODULE_REPO_CLEANUP.md：历史模块记录。
- P1_NUMERICAL_ATTRIBUTION.md：BF16归因及尚未关闭的跨计算形状数值问题。

## 验收

每模块做相关回归、必要真实模型门禁和git diff --check。
P2-A的保守容量准入只保证有限工作负载推进，不宣称无饥饿或性能收益。
显式KV预算不是运行时显存硬上限，本轮真实负载allocator峰值不等价于activation/Graph硬上限，自动定容仍待后续验证。
nano候选固定SHA bb823b3e06983d71485a8e1f23715ebd87d98ef8；正式测量记录所有框架SHA/依赖/模型/负载/停止条件/原始指标。
用户仓库实施后需要重新验证，不能直接复用助手通过结论。
P3-A CUDA性能优化已在助手工作区完成：常规281 passed/22 skipped，GPU专项27 passed。
真实7步11采样行保持原预算，max row mean_abs=.045703、max_abs=.281250、max TV=.018456，greedy11/11。
最终kernel在本机9个测量shape的中位耗时均低于旧Triton候选；同引擎五轮交替短混合负载中位吞吐
reference86.77、旧Triton95.27、最终CUDA98.27 tok/s。只说明当前测量条件，不宣称已超过nano-vLLM/vLLM/SGLang。
本轮性能代码相对上一版CUDA仅改paged_decode.cu；接口/metadata/接线不变。此条为P3-A历史记录；用户已提交8653e1f，本轮P3-B1源码按窗口教程实施。
P3-A历史FlashInfer decode候选未通过当时预算；不能据此否定prefill。P3-B2b的FI0.3.1 FA2历史prefill通过相同原预算，现作为可选依赖接入。
FlashAttention本轮检查的官方main接口要求paged KV页大小为256的倍数，当前16不能直接接入。详见MODULE_P3A_PAGED_DECODE.md。
对应测试为test_cuda_decode.py、test_cuda_model.py，替换旧test_triton_*；保留原数值预算，补充host张量契约和非默认stream/event依赖门禁。CUDA源文件已验证能进入分发包，打包通过不等于数值通过。

## P3-B1历史交付（用户已提交bd1c34a）

见 MODULE_P3B1_KV_STORE_PREFILL.md。助手已验证融合KV store与首段prefill直接读当前K/V，
保留reference和带历史chunk的原路径；CudaDecodeAttention更名CudaAttention，配置仍为cuda。
本轮未改变paged_decode.cu，未合入两个scores实验。修复用户package-data拼写。
常规282 passed/46 skipped，GPU组合52 passed；本地微基准改善，整引擎噪声大，不宣称稳定吞吐收益。
测试和文档迁移到用户仓库不表示用户框架源码已实施。本轮源码基线为用户8653e1f及实际文件。

## 面试资料沉淀约定

Windows工作区 ai-infra-interview/README.md 是入口，Paged Decode专题已完成，不重复。
发现值得沉淀的亮点时先简述理由；只有用户明确同意才创建或扩写专题。普通模块不必写成简历。
区分实现、验证、实验和计划；算子/整引擎结果分开，不凭语言或指令宽度推导性能。

## P3-B2a历史交付（用户已提交7b5e8f9）

见MODULE_P3B2_CHUNK_ATTENTION.md。生产仅改runtime/attention.py；符号causal_lower_right
启用支持形状的torch FlashAttention，实际QKV能力检测失败时缓存原bool mask跨层复用。
无新依赖，仍gather历史，不修改CUDA kernels。助手283 passed/51 skipped，GPU组合58 passed。
原实模门禁与更长chunk对上一模块门禁均通过；短混合5轮同机同次测量有收益，不能当完整框架对标。
FI0.3.1 paged prefill候选通过本轮原门禁，长历史有性能价值；未因依赖/workspace/计划生命周期
成本而在本轮接入，属于B2b候选，不与之前FI decode失败混为一谈。没有新增面试专题。

## P3-B2b历史交付（用户已提交5fe1a09）

详见MODULE_P3B2B_PAGED_PREFILL.md。新增显式flashinfer混合后端，仅历史多Query直读分页KV；
fresh与自研decode/store不变，reference仍为默认。固定FI0.3.1/frontend1.13.0、FA2、D128/page16。
Runner持有planner，每步CPU int32计划跨层复用，Q/O scratch跨层复用，plan_done事件保护pinned CPU复用。
后端绑定创建stream，clear同步后释放，bootstrap预留40.125MiB GPU workspace及Q/O上限。
常规301 passed/61 skipped，GPU86 passed；原/长chunk实模门禁未放宽，greedy11/11和4/4。
长历史局部更快；短整引擎97.567vs103.440tok/s，长41.845vs41.216tok/s，轮间抖动明显。
不宣称全面提速/省显存，不替换默认。原P4/P5技术路线保留；按2026-09-21调整后的S1-S4顺序先完成v0.1可用闭环，不重复面试专题。

## S1历史交付（用户已提交5ae8a9b）

Tokenizer从本地Qwen3目录加载；raw文本批量生成、显式chat template/thinking开关，
TextOutput保存文本、完整token IDs、usage与EOS/length/cancelled分类。
LLM复用原Engine.generate，python -m tinyvllm输出JSON，不新增另一套推理循环。
Engine.close/上下文管理清理模型/KV/planner/RNG；关闭永久终止、丢弃未完成工作，不等于请求取消。
下一模块S2才增加GPU owner线程和有界异步桥，S3再增加HTTP与增量文本；当前接口仍串行、非线程安全。
S1常规327 passed/65 skipped，显存空闲后GPU组合18 passed；真实中文CLI生成成功。
最初显存不足失败保留记录，最终沿用原2GiB bootstrap及64MiB CLI预算补验通过。
新模块测试和完整限制见MODULE_S1_TEXT_API.md。源码完成仍以用户实际工作区为准。

## S2当前交付（助手验证，用户待复现）

仅新增tinyvllm/async_engine.py与包导出，保持Engine/Scheduler/模型/CUDA核心不变。
一个事件循环连接一个GPU owner线程；factory、submit/cancel/step/close均在owner线程执行。
请求数与保守token预算覆盖pending/active/completed-unread；输出deque有界，terminal独立。
慢消费者仅取消自身；dirty IDs合并通知防止事件循环回调排队无界。无轮询sleep、无每请求GPU线程。
request.aclose放弃消费并异步取消；service.aclose才等待线程退出。退出async context处理break/任务取消。
具体测试口径见MODULE_S2_ASYNC_ENGINE.md；没有发布性能提升数字，没有新建面试专题。
紧接S3：HTTP JSON/SSE、增量文本、断连与有界tokenization；继续沿S1-S4路线完成v0.1。
