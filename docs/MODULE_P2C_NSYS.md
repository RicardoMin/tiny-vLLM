# P2-B2：Nsight Systems、NVTX与结果解读

日期2026-09-16，用户基线main/9271c8d。助手实现已通过CPU测试，用户源码未修改。
用户最新要求：先完成教程，GPU完整采集稍后进行。不要擅自再次启动GPU采集。

## 上一模块用户实测

报告：用户.local/benchmarks/20260916-162009/result.json。
三轮每轮8请求×8输出=64token，所有arrival=0，prompt16，max_num_seqs4。
吞吐160.47/169.41/170.69 tok/s；TTFT p50=21.50/21.51/21.46ms，p95=207.06/218.38/212.19ms。
TPOT p95=27.40/24.98/24.13ms；三轮完整token序列一致，raw指标独立重算144项通过。
每轮2个PREFILL_ONLY和14个DECODE_ONLY，没有MIXED。TTFT高尾来自第二波排队，不是200ms prefill。
8样本nearest-rank p95就是最大值；每轮不足0.4秒，不足以得出可靠尾延迟或稳定吞吐结论。
KV池292块、511MiB，峰值仅8块/14MiB=2.74%；allocator基线1677.01MiB、峰值1680.01MiB、extra2.99MiB，三轮不增长。
这不是无泄漏证明，也没覆盖长上下文、真实EOS、持续到达、高并发、KV压力或框架比较。
报告identity=ad7c81c+dirty+hash（提交前运行），不能直接当9271c8d正式干净版本基准。
已重新验证用户仓库263 passed, 3 skipped；三项真实GPU测试通过（在游戏资源压力出现前）。

## 本轮实现与契约

benchmarks/tools/profiling.py提供默认关闭的NVTX范围与首轮CUDA capture。
measurement.py标记tinyvllm/step及tinyvllm/execute/prefill_only|mixed|decode_only。
run_tiny.py新增--profile；purpose=diagnostic/benchmark；cuda_capture_repeat与每轮requested标志。
只有第一轮正式回放捕获，权重加载、hash、warmup排除；起止同步，不逐step同步。
NVTX是host范围，不等于GPU execution时间；嵌套范围会重叠，nvtx_sum各行时间不得相加当总时间。
capture发生错误仍尝试同步/stop并保留原始异常；NVTX自身实现异常不是全面故障恢复契约。
Engine可能把execute内KeyboardInterrupt封装RuntimeError，因此此时报告为failed，仍不会误报成功。
报告标记是请求profiling，实际有无外部collector由nsys-rep/CUDA/NVTX数据确认。

生成器新增--profile-mixed：4请求，prompt16/128/256/512，各16输出，arrival全0。
max_model_len必须>=528，教程使用1024；tokenbudget128，chunk64，seq4。
默认旧smoke字节完全不变。新负载records SHA256：316c6fc3dc27bc70d1b908257ead6d6be7ccf21bdc0488abd2e0cf1e9d3d16b4。
它用于观察chunked prefill/mixed/decode，不是完整benchmark矩阵或性能优化。

## 工具配置

nsys版本2026.1.3.425，已在/home/ricardo/.local/bin/nsys建立链接。
实际安装目录/home/ricardo/.local/opt/nvidia/nsight-systems-2026.1.3/opt/nvidia/nsight-systems/2026.1.3。
官方deb SHA256 c7309f1c9850f66a9eb95e7215883b8e8e439df6f65ea0cecd81e6b0181a4e83。
使用官方NVIDIA包解压到用户目录，无root、无驱动变更。apt链接带/./通过本机代理404，Windows curl正规URL成功并核对hash。
未安装Windows GUI、未安装ncu。ncu在完整nsys归因后针对具体kernel学习，遇到实际计数器权限错误再处理。
本机nsys status -e显示process-tree CPU环境可用，system-wide不可用；本轮关闭CPU sampling/context switch，仅cuda,nvtx，不需修改系统权限。

## 本轮验证与采集边界

最终默认回归275 passed, 3 skipped；compileall及本模块差异检查通过。

新增tests/test_profiling.py覆盖禁用无CUDA调用、范围嵌套、异常stop、时间/输出语义及仅首测量轮capture。
tests/test_profile_workload.py覆盖旧smoke字节兼容、新负载validator、可复现生成和CPU混合调度。
GPU第一次512MiB池运行因游戏等占用在bootstrap预算检查失败，无trace。
第二次128MiB池尝试进入capture，随后按用户选择中止；报告status=failed、runs=[]。
产生partial mixed-128.nsys-rep，nsys stats能解析CUDA数据与三种mode的NVTX范围，确认工具与接线可用。
该trace处于游戏竞争、且执行不完整，禁止使用其时长/百分比作瓶颈、优化或性能结论。
后续应在用户明确继续且GPU空闲时，以普通benchmark和diagnostic两次独立运行完成验收。

## 后续命令

```bash
python -m benchmarks.tools.make_smoke_workload --profile-mixed
python -m benchmarks.tools.validate_workload benchmarks/workloads/profile-mixed-qwen3-v1.json

run_dir=".local/profiles/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none \
  --output="$run_dir/tiny" \
  python -m benchmarks.tools.run_tiny \
    --model "$HOME/models/Qwen3/Qwen3-0.6B" \
    --workload benchmarks/workloads/profile-mixed-qwen3-v1.json \
    --output "$run_dir/profile-result.json" \
    --max-model-len 1024 --kv-cache-mib 512 --warmup 1 --repeats 1 --profile

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum "$run_dir/tiny.nsys-rep"
```

正式性能报告应不带--profile且不通过nsys启动。Trace时长只用于定位，同条件重跑benchmark验证优化收益。
系统资料：https://docs.nvidia.com/nsight-systems/UserGuide/index.html
Compute资料：https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html
GUI下载：https://developer.nvidia.com/nsight-systems/get-started
