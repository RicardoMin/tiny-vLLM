# tiny-vLLM 项目入口与协作约定

更新：2026-09-16。以 TINY_VLLM_V1_REFACTOR_PLAN.md 为架构路线；审计证据见 ARCHITECTURE_AUDIT_20260915.md。

## 目标与状态

单GPU、当前dense Qwen3，学习简洁高性能推理框架。目标是在相同条件下改善相对nano-vLLM的TTFT、TPOT、吞吐、显存及可持续并发，并与vLLM/SGLang比较；代码量不显著增加。
不擅自改成单项快10%或接受数倍代码量。目前没有完成框架间性能比较，不能声称领先。

- 用户源码：WSL /home/ricardo/project/tiny-vllm，main，P2-B2已提交fa202b6，生产执行器名Executor。
- 助手工作区：WSL /home/ricardo/project/tiny-vllm-codex，main；P3-A改用手写CUDA C++直接分页单query attention，用户源码通过窗口教程复现。
- 模型：/home/ricardo/models/Qwen3/Qwen3-0.6B；权重不进Git。
- 环境：PyTorch2.9.1+cu128、Transformers4.51.3、RTX5060Laptop；本轮pyproject对齐torch2.9系列；旧L4/cu124锁仅为历史环境，不能用于本机重建；runner保存实际packages与模型文件hash。
- P1全局分页KV reference已完成。P3-A改用显式cuda后端，单query直接读分页KV，多query仍gather+SDPA；默认reference。用户仅开始修改contracts.py，后续源码尚未实施，read_slots类型需包含None。
- 算子学习路线采用CUDA C++，便于理解线程、warp、共享内存、同步和访存；GEMM等继续使用成熟库，也允许评估FlashInfer等成熟attention实现。CUDA与Triton都生成GPU代码，语言选择本身不保证更快，收益须按同条件A/B验收。
- 编译工具链：本机隔离安装CUDA Toolkit 12.8.93于/home/ricardo/.local/cuda-12.8，匹配PyTorch cu128；原系统13.2保持不变。构建使用CXX=g++-14、CC=gcc-14、MAX_JOBS=1、TORCH_CUDA_ARCH_LIST=12.0；这些是当前RTX5060开发配置，不写死为所有设备支持范围。
- P3-A通过Python loader编译paged_decode.cu并注册torch.ops，显式选择cuda时在构造阶段按需加载；kernel使用PyTorch当前CUDA stream。CPU/reference无需编译CUDA扩展。
- P2-A：逻辑容量准入、入口限制、活跃ID/远期堆、独占generate、失败资源释放、显式KV字节预算与Engine.from_config。

- P2-B1：逐token观测、open-loop回放、完整结果验收、CUDA/KV峰值和源码内容身份；见MODULE_P2B_MEASUREMENT.md。

- P2-B2：用户已成功采集20260916-183722 trace；见MODULE_P2C_NSYS.md。
- 推进顺序：P3-A直接分页单query → P3-B多query attention/KV store → P4融合/metadata/Graph → P5调度与复用 → v1接口收口和完整多框架对标。数值门禁随实现推进。

## 协作规则

助手先在自己的WSL工作区实现并验证；推理源码仅由用户按当前窗口教程修改。
教程必须给用户当前基线行号、函数锚点、准确代码、目的与难点，不能只给文件链接或补丁。
代码用正常开发格式：合理空行、缩进、空格和分段，不挤成连续短行；一个模块有完整可验证成果。
同文件按旧行号从下往上操作。测试验证后由助手直接同步到用户tests/，窗口只给简报，不贴长篇完整测试。
架构/路线文档可同步；两边在main开发，用户自行add/commit，不再每模块建分支。
不得覆盖用户独立源码；测试同步不代表用户实施完成。

## 文档

- TINY_VLLM_V1_REFACTOR_PLAN.md：路线与职责。
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
本轮性能代码相对上一版CUDA仅改paged_decode.cu；接口/metadata/接线不变。用户仍仅开始contracts修改，完整源码按窗口教程实施。
FlashInfer0.3.1的FA2 Tensor Core和CUDA Core候选局部快但未通过原整模型预算；隔离试验安装未成为项目依赖。
FlashAttention本轮检查的官方main接口要求paged KV页大小为256的倍数，当前16不能直接接入。详见MODULE_P3A_PAGED_DECODE.md。
对应测试为test_cuda_decode.py、test_cuda_model.py，替换旧test_triton_*；保留原数值预算，补充host张量契约和非默认stream/event依赖门禁。CUDA源文件已验证能进入分发包，打包通过不等于数值通过。
