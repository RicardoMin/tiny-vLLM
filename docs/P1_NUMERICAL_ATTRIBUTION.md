# P1 numerical attribution — 2026-09-14

Source inspected: user `/home/ricardo/project/tiny-vllm`, HEAD `f929cda`.
No user or assistant production source was changed for this audit. The experiment
monkeypatched two layer methods in memory on the same loaded tiny model.

Environment: PyTorch 2.9.1+cu128, local Qwen3-0.6B checkpoint,
BF16 CUDA, Hugging Face attention implementation `sdpa`, inference mode.
Models loaded sequentially; HF logits retained on CPU before loading tiny.

## Exact ablation results

Sequences: `(1, 4, 7)` and `(11, 12)`.

| Variant | Last-position max absolute diff, sequence 1 | sequence 2 | Last-position mean absolute diff, sequence 1 | sequence 2 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.171875 | 0.1796875 | 0.0261920821 | 0.0365855955 |
| Only RMSNorm HF operation order | 0.1337890625 | 0.15625 | 0.0222703125 | 0.0232316684 |
| Only RoPE HF operation order | 0.34375 | 0.1657714844 | 0.0663283244 | 0.0280579012 |
| Both changes | 0 | 0 | 0 | 0 |

The baseline combined last-position mean 0.03138884 reproduces the previous
reported result. Baseline all-position maximum is larger: 0.4375. That must not
be confused with the previous last-position comparison.

With both changes, every logit at every position was bitwise identical to HF
for both sequences (455808 + 303872 elements), not merely the greedy output.
All four variants retained greedy token IDs `[14582, 82, 606]` / `[14582, 16]`.
These two short prompts establish attribution of the observed mismatch, not
a universal bitwise-equality guarantee across prompts, shapes, kernels or devices.

## Root cause and smallest reference fix

1. Tiny used `F.rms_norm`; HF explicitly computes FP32 variance and normalized
   values, casts the normalized values to input dtype, then multiplies by the
   weight. Equivalent real-number formulas need not have identical BF16 rounding.
2. Tiny kept RoPE cos/sin and the rotation math in FP32 until final cast. HF
   computes trig in FP32, casts cos/sin to input dtype, then performs the rotation
   using input dtype. The tiny approach is not inherently less accurate; it
   follows a different rounding contract from the reference.

Recommended RMSNorm forward:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    values = x.float()
    variance = values.square().mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + self.eps)
    return self.weight * normalized.to(dtype=x.dtype)
```

Recommended RotaryEmbedding._rotate:

```python
@staticmethod
def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    first, second = x.chunk(2, dim=-1)
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin),
        dim=-1,
    )
```

This is an eager reference contract. Do not claim faster kernels from these
changes; future fused kernels may use a documented precision policy and require
fresh numerical and performance evidence. Fix both rounding differences together;
the RoPE-only ablation increases error on sequence 1.

## Isolated CPU checks and test recommendations

Random nonuniform RMSNorm weights, seed 872, 10 x 1024 values:

- FP32 original vs HF: identical.
- BF16 original vs HF: max 0.0625, 2710/10240 elements differ.
- FP16 original vs HF: max 0.00390625, 2786/10240 differ.
- Explicit HF formula: identical for all three dtypes.

RoPE head_dim 128, positions `[0, 1, 2, 7, 257, 1024, 40000]`:

- FP32 original vs HF: identical.
- BF16 original vs HF: max 0.015625, 2688/14336 query elements differ.
- FP16 original vs HF: max 0.001953125, 2959/14336 differ.
- Input-dtype cos/sin and rotation: identical for all three dtypes.

Update `tests/test_layers.py` with two compact parametrized HF comparisons:
RMSNorm with nonuniform weights and BF16/FP32 inputs; RoPE with BF16/FP32,
separate Q/K head counts, nonzero distant positions. Use independent HF layer
implementations as oracle. Existing FP32-only tests with weight=1 cannot catch
this issue. P1 model tests should separately verify mixed-request/chunked paged
execution against full-prefix HF; that isolates storage and causal-mask errors.

Reproducer: `C:/Users/17879/Documents/ChatGPT/tiny-vllm/.p1_numerical_audit.py`.

```bash
/home/ricardo/project/tiny-vllm/.venv-tiny/bin/python \
  /mnt/c/Users/17879/Documents/ChatGPT/tiny-vllm/.p1_numerical_audit.py
/home/ricardo/project/tiny-vllm/.venv-tiny/bin/python \
  /mnt/c/Users/17879/Documents/ChatGPT/tiny-vllm/.p1_numerical_audit.py --real
```

## Paged mixed-batch BF16 follow-up

After both normalization/rotation fixes, compare the assistant P1 model against
HF using independent per-request incremental caches, with the same query chunks.
`mixed_gemm_audit.py` captures projection and norm inputs/outputs at every layer.

- Step 0 (query lengths 3, 2): every captured layer output is identical.
- Step 1 (query lengths 1, 2, 2): the first divergent module is
  `model.layers.0.self_attn.qkv_proj`.
- Its input is identical (5120 elements). Only 3 of 20480 output elements differ,
  with maximum absolute difference 0.000003814697265625.

Recompute the projection repeatedly with exactly the same captured input and
weights, varying only the GEMM invocation shapes:

| Comparison | Max absolute difference | Differing elements |
| --- | ---: | ---: |
| One packed QKV GEMM versus separate Q/K/V GEMMs over the same 5 rows | 0 | 0 |
| One 5-row GEMM versus request-sized GEMMs (1, 2, 2 rows) | 0.000003814697265625 | 3 |
| One packed GEMM versus HF-style separate shards and request-sized GEMMs | 0.000003814697265625 | 3 |
| Default versus `allow_bf16_reduced_precision_reduction=False` | 0 | 0 |
| Whole/request-sized difference with reduced precision reduction disabled | 0.000003814697265625 | 3 |

Thus the earliest divergence in mixed incremental execution is a batch-size
dependent GEMM rounding difference, occurring before paged attention. This is
distinct from the old norm/RoPE rounding differences fixed above. Disabling the
reduced-precision-reduction flag does not remove it in this experiment.

This observation alone does not prove that every later difference is benign.
Use a test-only dense attention oracle with the *same flat model batch* to
isolate paging exactly, and retain tight FP32 HF integration checks. Document
HF BF16 full-logit/KV drift separately; do not describe a failed strict BF16 HF
KV tolerance as having passed just because greedy token IDs agree.
