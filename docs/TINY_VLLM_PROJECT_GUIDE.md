# tiny-vLLM 项目入口与协作约定

更新：2026-09-16。以 TINY_VLLM_V1_REFACTOR_PLAN.md 为架构路线；审计证据见 ARCHITECTURE_AUDIT_20260915.md。

## 目标与状态

单GPU、当前dense Qwen3，学习简洁高性能推理框架。目标是在相同条件下改善相对nano-vLLM的TTFT、TPOT、吞吐、显存及可持续并发，并与vLLM/SGLang比较；代码量不显著增加。
不擅自改成单项快10%或接受数倍代码量。目前没有完成框架间性能比较，不能声称领先。

- 用户源码：WSL /home/ricardo/project/tiny-vllm，main，P2-A已提交ad7c81c，EagerExecutor已改名Executor。
- 助手工作区：WSL /home/ricardo/project/tiny-vllm-codex，main；P2-B1测量模块已实现并验证，待用户按教程复现。
- 模型：/home/ricardo/models/Qwen3/Qwen3-0.6B；权重不进Git。
- 环境：PyTorch2.9.1+cu128、Transformers4.51.3、RTX5060Laptop；本轮pyproject对齐torch2.9系列；旧L4/cu124锁仅为历史环境，不能用于本机重建；runner保存实际packages与模型文件hash。
- P1全局分页KV reference已完成。模型每batch一次forward，各层仍逐请求gather+SDPA，直接分页kernel未完成。
- P2-A：逻辑容量准入、入口限制、活跃ID/远期堆、独占generate、失败资源释放、显式KV字节预算与Engine.from_config。

- P2-B1：逐token观测、open-loop回放、完整结果验收、CUDA/KV峰值和源码内容身份；见MODULE_P2B_MEASUREMENT.md。

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
- MODULE_P2B_MEASUREMENT.md：本模块测量口径、验证与后续边界。
- MODULE_P2A_ADMISSION.md：已完成的容量/生命周期/预算契约。
- MODULE_P1_PAGED_REFERENCE.md、MODULE_P0_BLOCK_POOL.md、MODULE_REPO_CLEANUP.md：历史模块记录。
- P1_NUMERICAL_ATTRIBUTION.md：BF16归因及尚未关闭的跨计算形状数值问题。

## 验收

每模块做相关回归、必要真实模型门禁和git diff --check。
P2-A的保守容量准入只保证有限工作负载推进，不宣称无饥饿或性能收益。
显式KV预算不是运行时显存硬上限，本轮真实负载allocator峰值不等价于activation/Graph硬上限，自动定容仍待后续验证。
nano候选固定SHA bb823b3e06983d71485a8e1f23715ebd87d98ef8；正式测量记录所有框架SHA/依赖/模型/负载/停止条件/原始指标。
用户仓库实施后需要重新验证，不能直接复用助手通过结论。
