# tiny-vLLM 项目入口与协作约定

更新：2026-09-13。以同目录 TINY_VLLM_V1_REFACTOR_PLAN.md 为唯一有效路线。

## 目标与实际状态

单 GPU、当前 dense Qwen3，学习简洁高性能推理框架。保留用户的目标：在相同条件下
改善相对 nano-vLLM 的 TTFT、TPOT、吞吐和可持续并发，代码量不显著增加。
目前没有完成同条件性能比较的证据。不得擅自将目标改成单项快 10%，也不预先接受
代码量为 nano-vLLM 的 2–2.5 倍；按统一范围统计并解释复杂度增量。

- 用户源码：WSL `/home/ricardo/project/tiny-vllm`，本轮教程起点 `7185079`。
- 助手工作区：WSL `/home/ricardo/project/tiny-vllm-codex`，main。
- 本轮实现位于助手 `cb16aaa` 后的工作树，未替用户提交。
- 模型：`/home/ricardo/models/Qwen3/Qwen3-0.6B`；权重不进 Git。
- 本轮助手 P0 验证：全量 159 passed；用户源码实施进度须单独核查。

基线已有 Engine 生命周期、连续调度、chunked prefill、混合计划、sampling、KV 事务、
Qwen3 权重加载和 flat batch ModelRunner。Transformer 每 batch 调用一次，reference
attention 内部仍逐请求调用 SDPA，使用 contiguous cache。
PagedKVCache 已有物理存储原型；cuda 拼写、反向范围和 slot 越界检查已修复。
物理存储尚未接入模型 attention，完成 P0 不意味着完成 PagedAttention。

## 文档分工

- [TINY_VLLM_V1_REFACTOR_PLAN.md](TINY_VLLM_V1_REFACTOR_PLAN.md)：有效路线、职责与验收。
- [TINY_VLLM_ARCHITECTURE_REVIEW.md](TINY_VLLM_ARCHITECTURE_REVIEW.md)：理由、已知限制和待解决接线问题。
- [MODULE_P0_BLOCK_POOL.md](MODULE_P0_BLOCK_POOL.md)：本轮模块契约与验证记录。

## 协作规则

助手先在自己的 WSL 工作区实现并验证。推理框架源码只由用户按当前窗口教程修改；
教程提供用户基线行号、函数锚点、精确替换代码、目的和难点，不能只给文件链接。
测试由助手验证后迁移到用户 tests/，回复给测试简报，不展示长篇测试代码。
架构和路线文档可直接修正并同步，Windows 文档副本保持一致。
两边直接在 main 开发，用户自行 add/commit，不再每模块建分支。
不得覆盖用户独立改动，测试已同步不代表源码已实施。

## 验证与对比

模块运行必要回归及 git diff --check。模型/设备路径改动才要求真实权重与 GPU 验证；
CPU BlockPool 不需要加载 Qwen3。
对标候选为 `GeeeekExplorer/nano-vllm` 的
`bb823b3e06983d71485a8e1f23715ebd87d98ef8`，正式测量前核对 checkout。
记录双方 SHA、模型/tokenizer、环境、dtype、停止条件、负载、warmup 和原始结果。
不以删除空行或必要测试伪装架构精简。
