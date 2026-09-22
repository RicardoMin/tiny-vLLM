# S1：文本API、聊天模板、CLI与显式关闭

日期：2026-09-21。用户基线5fe1a09；已提交P3-B2b。
本模块在助手副本实现并验证；教程供用户复现源码，测试/文档按约定迁移。

## 实现与职责

新增tinyvllm/tokenizer.py：只做本地编码、官方chat template和完整解码。
新增tinyvllm/llm.py：LLM、TextOutput，同步文本门面，复用Engine.generate一次提交整个batch。
新增tinyvllm/__main__.py：raw --prompt可重复、--messages UTF-8 JSON，输出结构化JSON。
修改Engine.close/_dispose，共享正常关闭和fail-stop释放；ModelRunner.clear释放model引用；
Executor.clear保证runner清理抛错时仍清理sampler；包根导出LLM/TextOutput。
没有HTTP、后台线程或流式假实现；这些紧接S2/S3，详见V0_1_DELIVERY_PLAN.md。

## 输入与输出契约

raw prompt不自动套chat模板，add_special_tokens=False；空编码拒绝，空白字符本身不做strip。
chat支持system/user/assistant和字符串content，末条必须user；不支持assistant续写、tool或多模态。
采用模型目录自带apply_chat_template(tokenize=True,add_generation_prompt=True)，没有模板后再次加special tokens。
默认enable_thinking=False可复现普通助手输出；True时保留模型生成的thinking内容，不擅自解析/删标签。
Qwen3本地实际类是Qwen2TokenizerFast，这是tokenizer实现复用，不代表加载了错误的模型。
模型实际config EOS为151645；以Engine/Scheduler配置为终止依据，不用tokenizer文本猜EOS。
text只解码新生成token；skip_special_tokens=True、clean_up_tokenization_spaces=False，保存原始IDs。
usage包含真实输入模板tokens与全部输出tokens，包括被文本隐藏的EOS；不按字符串长度统计。
finish_reason：ABORTED为cancelled；不ignore_eos且末ID为Engine EOS时stop；其余长度/上下文终止为length。
EOS恰好在长度边界时对外归类stop；这只是当前EOS/长度语义，不代表支持任意stop字符串。
原Engine保留全批入队前校验；文本批全部encode后再发布，因此后项非法不会留下前项已入队。
generate([])返回空列表，generate('')因无prompt token拒绝；当前返回完整结果，未实现增量detokenize。
多轮chat由调用者传完整messages，不持有跨请求的会话KV或自动复用前缀。

## 生命周期

LLM是串行独占门面，不允许用户多线程并发调用同一个对象。with上下文退出调用Engine.close。
close幂等且永久；清队列/请求、scheduler ownership、KV、FI workspace、模型引用与采样RNG。
close过程中generate正在运行时拒绝；不把close当请求cancel或承诺向请求发送CompletionEvent。
S2必须先终结所有桥接层等待者再调用底层close。FI创建/使用/关闭须位于同一owner stream。
_dispose沿用fail-stop的逐项清理/错误附注；正常close发现清理异常仍报错。
上下文原先有异常时，清理异常添加为note，不覆盖原始错误；不引入__del__或gc自动推理清理。
显存对象释放不等于PyTorch allocator reserved立刻下降，本模块用weakref核验真实对象不再持有。

## 验证

常规回归：327 passed、65 skipped；跳过项是显式模型/GPU门禁，另行运行。
本模块新增test_text_api.py和test_text_model.py；没有修改旧数值阈值或测试预算。
文本门禁先单独复测：4 passed，含本地tokenizer与reference/cuda/flashinfer三个真实模型case。
显存空闲后扩大GPU组合最终18 passed，覆盖文本、原CUDA数值、原2GiB KV bootstrap和FI生命周期。
每个后端内部对比文本API与同后端原token API的完整output IDs；不要求后端之间自由生成文本完全相同。
三后端模型/参数/KV/FI workspace weakref释放通过；原bootstrap实际1170块全部归还，KV对象释放通过。
真实CLI使用cuda后端、64MiB KV预算、48输出token上限，成功输出中文JSON并以EOS结束：
输入22 tokens、输出13 tokens（包含末尾EOS 151645）、总计35 tokens，finish_reason=stop。
第一次扩大组合16 passed/2 failed和CLI显存不足记录保留在gpu-memory-limited.log及cli-memory-limited.stderr。
那些失败均发生在模型加载后的预算检查；最终补验使用原测试与CLI预算，没有提高utilization或降低门禁要求。
本模块不是性能优化，GPU繁忙条件下没有发布新的吞吐或延迟提升数字。

## 使用

```python
from pathlib import Path
from tinyvllm import EngineConfig, LLM, SamplingParams

config = EngineConfig(
    model=Path('/home/ricardo/models/Qwen3/Qwen3-0.6B'),
    attention_backend='cuda',
    max_model_len=1024,
    max_num_seqs=4,
    max_num_batched_tokens=256,
    prefill_chunk_size=128,
    prefill_reserve_tokens=0,
    prefill_reserve_requests=0,
    kv_cache_memory_bytes=512 * 1024**2,
)

with LLM(config) as llm:
    output = llm.chat(
        [{'role': 'user', 'content': '请用一句话介绍你自己。'}],
        SamplingParams(max_new_tokens=64),
    )
    print(output.text)
    print(output.finish_reason, output.total_token_count)
```

python -m tinyvllm --help可查看CLI；--prompt为原始续写，不自动当作聊天。
文件证据：Windows .module_s1_text_20260921 的baseline/generated、tokenizer_probe.json、
suite.log、gpu.log、gpu-memory-limited.log、text-gpu.log、cli.json/cli.stderr、
cli-memory-limited.stderr、tutorial_validation.json、verification.json和delivered.json。
未修改已有算子、调度策略、数值阈值或用户独立文档修改；没有创建简历专题。
