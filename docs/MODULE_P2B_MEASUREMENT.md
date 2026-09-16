# P2-B1：可审计的本地测量基线

更新：2026-09-16。原教程基线main/ad7c81c；用户已在9271c8d完成。2026-09-16复核263 passed, 3 skipped及3项真实GPU测试通过。

## 交付与实现

新增benchmarks/tools/measurement.py、run_tiny.py；修复collect_env.git_identity。
引擎路径不增加统计分支。沿用Executor名称与EngineExecutor协议。
顺便删除engine.py多余bootstrap import、修复bootstrap available_bytes拼写、恢复scheduler.shutdown的_stale_arrivals清零。
pyproject的torch范围改为2.9系列，匹配本机已运行的2.9.1+cu128；旧tiny-l4-cu124.lock是历史环境，不应在RTX5060上套用。
完整已安装包版本、模型文件hash和源码hash随报告保存，尚未宣称在干净环境完成离线可重建验证。
用户自行更新benchmarks、tinyvllm、pyproject和.gitattributes；助手只同步tests及docs。

## 时间口径

- 输入是严格校验的workload.v1；预先构造sampling并检查prompt+output总长度，拒绝静默截断。
- open-loop计划到达独立于请求完成；单线程在step边界提交已到达请求，不能模拟独立HTTP接入线程。submit lag单列。
- t0在回放开始时取monotonic_ns，首次arrival之前空闲也计入duration与吞吐分母。
- 每次step返回立刻读一次host时钟，同批token共享时刻。计时包括引擎调度、lowering、GPU、采样和事件返回，不包括模型加载、JSON读写及模型hash。
- TTFT=首token观测时刻-计划arrival；ITL=相邻token观测差；TPOT=(末token-首token)/(n-1)，n<2时null。
- TTFT/TPOT/E2E按请求统计；ITL按所有token间隔汇总，长输出自然权重更大。nearest-rank，无插值；每轮单独报告，不平均p95。
- 输出token含EOS；完整事件序列必须与completion逐项相等。拒绝未知请求、完成后token、同step重复token、重复completion、越界token、EOS后生成和非EOS提前结束。
- ignore_eos=True必须足量输出；失败/中断报告保留已完成轮次，status必须failed/interrupted，不能当成功结果参与比较。
- 当前没有HTTP、tokenizer耗时、SLO goodput或可持续并发结论。

## 内存口径

KV池是固定分配的K+V tensor bytes；请求占用块数是逻辑高水位，二者不同。
执行包装器在已分配块、尚未释放完成请求时记录占用，避免step后取值漏峰。
同步->baseline allocated/reserved->reset peak->真实回放->同步->peak。
不逐step额外同步，不每轮empty_cache。warmup完整回放后再测，每轮独立重置峰值。
extra_peak_allocated=peak-baseline，只表示额外PyTorch allocator分配，不是精确activation、设备总占用或运行显存硬上限。
SDPA报告的是启用开关，并非实际每次选择的kernel；kernel归因后续通过profiler完成。

## 可复现身份与换行问题

git identity包含commit、status和index+tracked/untracked实际内容hash；同status不同代码不再共用身份。
symlink记录目标字串、不跟随外部文件；特殊文件/submodule明确拒绝。ignored文件不纳入源码hash；模型/workload单独hash。
运行时不要并发修改源码、模型或workload。结果放.local/，避免报告改变下次源码身份。
原smoke jsonl已经提交CRLF，但manifest期望LF，实际SHA不一致。用原make_smoke_workload生成器重建，并以.gitattributes固定workloads/*.jsonl的LF。

## 验证

默认回归263 passed, 3 skipped；真实混合benchmark测试已通过。
混合GPU负载prompt长度1/80/130，各输出8，共24 token；预热1轮、测量2轮。
两轮均包含PREFILL_ONLY=1、MIXED=3、DECODE_ONLY=7；请求KV峰值16块、29,360,128 bytes。
该次allocator baseline=1,758,476,288，peak=1,766,655,488，extra=8,179,200 bytes。
数字仅描述该负载/该环境，短smoke不能代表性能优劣；尚未运行框架间公平比较。

测试简报：test_benchmark_measurement.py覆盖计时/异常事件/停止/回收；test_benchmark_identity.py覆盖内容身份；
test_benchmark_cli.py覆盖仓库workload原文件与失败/中断落盘；test_benchmark_gpu.py覆盖真实CLI混合回放与memory报告。

## 接下来

P2-B2补BF16跨计算形状数值验收、固定nano版本token观察适配、负载矩阵与operator profile。
明确数值契约后进入P3直接分页decode/varlen prefill；reference attention保留作oracle。
自动KV定容不在本轮：需要覆盖真实峰值形状并保留安全余量，不能用本轮短负载差值直接反推。
窗口教程是用户复现入口；.module_p2b_20260916保存baseline/expected及交付记录供助手恢复上下文。
