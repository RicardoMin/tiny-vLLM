#include <ATen/ATen.h>
#include <torch/library.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <type_traits>


namespace {

constexpr int kThreads = 128;
constexpr int kTile = 64;
constexpr int kWarp = 32;  

template <int Width = kWarp>
__device__ float warp_sum(float value) {
    for (int offset = Width / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }

    return value;
}


__device__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(
            value,
            __shfl_xor_sync(0xffffffff, value, offset)
        );
    }

    return value;
}


template <typename scalar_t>
__device__ __forceinline__ void unpack_eight(
    uint4 packed,
    float (&values)[8]
) {
    // Reconstruct FP16/BF16 values from their original bits.
    const unsigned int words[4] = {
        packed.x,
        packed.y,
        packed.z,
        packed.w
    };

    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        values[pair * 2] = float(scalar_t(
            static_cast<unsigned short>(words[pair]),
            scalar_t::from_bits()
        ));

        values[pair * 2 + 1] = float(scalar_t(
            static_cast<unsigned short>(words[pair] >> 16),
            scalar_t::from_bits()
        ));
    }
}


template <typename scalar_t>
__device__ __forceinline__ void load_eight(
    const scalar_t* source,
    float (&values)[8]
) {
    // Caller guarantees eight valid values and 16-bytes olignment.
    unpack_eight<scalar_t>(
        *reinterpret_cast<const uint4*>(source),
        values
    );
}


template <
    typename scalar_t,
    int Dim,
    bool PowerOfTwoBlock,
    bool VectorizedKV,
    bool PrefetchValues
>
__global__ void paged_decode_kernel(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    scalar_t* output,
    const int64_t* rows,
    const int64_t* blocks,
    const int64_t* offsets,
    const int64_t* lengths,
    int heads,
    int kv_heads,
    int dynamic_head_dim,
    int block_size,
    int block_shift,
    float scale,
    int64_t qs0,
    int64_t qs1,
    int64_t qs2,
    int64_t os0,
    int64_t os1,
    int64_t os2
) {
    static_assert(!PrefetchValues || (Dim == 128 && VectorizedKV));

    const int head_dim = PrefetchValues ? 128 : dynamic_head_dim;

    const int request = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    const int kv_head = head / (heads / kv_heads);

    const int tid = threadIdx.x;
    const int lane = tid % kWarp;
    const int warp = tid / kWarp;

    const int64_t row = rows[request];
    const int64_t length = lengths[request];

    // Padded rows have no context. Do not read Q, block tables or KV.
    if (length == 0) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            output[row * os0 + head * os1 + d * os2] = scalar_t(0.f);
        }

        return;
    }

    const int64_t table_start = offsets[request];

    __shared__ int64_t slots[kTile];
    __shared__ float scores[kTile];
    __shared__ float weights[kTile];
    __shared__ float maximum, denominator, correction;

    __shared__ __align__(16) float partial_output[8][Dim];

    constexpr int kQueryLanes = Dim / 8;

    const int query_lane = tid % kQueryLanes;
    const int query_group = tid / kQueryLanes;
    const int dimension_start = query_lane * 8;

    float query[8];

    const scalar_t* query_start =
        q + row * qs0 + head * qs1 + dimension_start * qs2;

    if (
        qs2 == 1
        && dimension_start + 8 <= head_dim
        && (reinterpret_cast<uintptr_t>(query_start) & 15) == 0
    ) {
        load_eight(query_start, query);
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = dimension_start + j;

            query[j] = d < head_dim
                ? float(q[row * qs0 + head * qs1 + d * qs2])
                : 0.f;
        }
    }

    float acc[2] = {0.f, 0.f};

    if (tid == 0) {
        maximum = -INFINITY;
        denominator = 0.f;
    }

    __syncthreads();

    for (int64_t tile = (length - 1) / kTile; tile >= 0; --tile) {
        const int valid = int(
            min(int64_t(kTile), length - tile * kTile)
        );

        uint4 prefetched_values[8];
        uint4 prefetched_keys[8];

        // D128: issue independent K/V loads before QK arithmetic.
        // Keep packed 16-bit values instead of expanding everything to FP32.
        if constexpr (PrefetchValues) {
            #pragma unroll
            for (int item = 0; item < 8; ++item) {
                const int token = query_group + item * 8;

                if (token < valid) {
                    const int64_t position = tile * kTile + token;
                    
                    int64_t logical_block;
                    int64_t block_offset;

                    if constexpr (PowerOfTwoBlock) {
                        logical_block = position >> block_shift;
                        block_offset = position & (block_size - 1);
                    } else {
                        logical_block = position / block_size;
                        block_offset = position % block_size;
                    }

                    const int64_t block = 
                        blocks[table_start + logical_block];

                    const int64_t slot = 
                        block * block_size + block_offset;

                    const int64_t base = 
                        (slot * kv_heads + kv_head) * head_dim;

                    prefetched_keys[item] =
                        *reinterpret_cast<const uint4*>(
                            k + base + dimension_start
                        );

                    prefetched_values[item] = 
                        *reinterpret_cast<const uint4*>(
                            v + base + dimension_start
                        );

                    if (query_lane == 0) {
                        slots[token] = slot;
                    }
                }
            }
        }

        // Each subgroup computes one QK dot product.
        constexpr int kQueryGroups = kThreads / kQueryLanes;

        #pragma unroll (PrefetchValues ? 8 : 1)
        for (int item = 0; item < kTile / kQueryGroups; ++item) {
            const int token = query_group + item * kQueryGroups;

            int64_t slot = 0;

            if constexpr (!PrefetchValues) {
                if (token < valid) {
                    const int64_t position = tile * kTile + token;

                    int64_t logical_block;
                    int64_t block_offset;

                    if constexpr (PowerOfTwoBlock) {
                        logical_block = position >> block_shift;
                        block_offset = position & (block_size - 1);
                    } else {
                        logical_block = position / block_size;
                        block_offset = position % block_size;
                    }

                    const int64_t block =
                        blocks[table_start + logical_block];
                    
                    slot = block * block_size + block_offset;
                }
            }

            float keys[8] = {};

            if (token < valid) {
                if constexpr (PrefetchValues) {
                    unpack_eight<scalar_t>(
                        prefetched_keys[item],
                        keys
                    );
                } else {
                    const int64_t base = 
                        (slot * kv_heads + kv_head) * head_dim;

                    if constexpr (VectorizedKV) {
                        if (dimension_start < head_dim) {
                            load_eight(
                                k + base + dimension_start,
                                keys
                            );
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int d = dimension_start + j;

                            keys[j] = d < head_dim
                                ? float(k[base + d])
                                : 0.f;
                        }
                    }
                }
            }

            float dot = 0.f;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                dot = fmaf(query[j], keys[j], dot);
            }

            const float score = warp_sum<kQueryLanes>(dot) * scale;

            if (query_lane == 0) {
                if constexpr (!PrefetchValues) {
                    slots[token] = slot;
                }

                scores[token] = token < valid ? score : -INFINITY;
            }
        }
        __syncthreads();

        // Update online-softmax statistics for this tile.
        if (warp == 0) {
            const float next_max = fmaxf(
                maximum,
                warp_max(fmaxf(scores[lane], scores[lane + kWarp]))
            );

            const float rescale = __expf(maximum - next_max);
            const float p0 = __expf(scores[lane] - next_max);
            const float p1 = __expf(scores[lane + kWarp] - next_max);

            weights[lane] = float(scalar_t(p0));
            weights[lane + kWarp] = float(scalar_t(p1));

            scores[lane] = p0;
            scores[lane + kWarp] = p1;

            __syncwarp();

            float sum = scores[lane % 8];

            #pragma unroll
            for (int token = 8; token < kTile; token += 8) {
                sum += scores[lane % 8 + token];
            }

            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 4);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);

            if (lane == 0) {
                maximum = next_max;
                denominator = denominator * rescale + sum;
                correction = rescale;
            }
        }

        __syncthreads();

        // Eight groups accumulate disjoint token stripes.
        for (
            int group = query_group;
            group < 8;
            group += kThreads / kQueryLanes
        ) {
            float partial[8] = {};

            #pragma unroll
            for (int item = 0; item < kTile / 8; ++item) {
                const int token = group + item * 8;

                if (token < valid) {
                    const int64_t base = 
                        (slots[token] * kv_heads + kv_head) * head_dim;

                    const float weight = weights[token];

                    float values[8] = {};

                    if constexpr (PrefetchValues) {
                        unpack_eight<scalar_t>(
                            prefetched_values[item],
                            values
                        );
                    } else if constexpr (VectorizedKV) {
                        if (dimension_start < head_dim) {
                            load_eight(
                                v + base + dimension_start,
                                values
                            );
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int d = dimension_start + j;

                            values[j] = d < head_dim
                                ? float(v[base + d])
                                : 0.f;
                        }
                    }

                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int d = dimension_start + j;

                        if (d < head_dim) {
                            partial[j] = fmaf(
                                weight,
                                values[j],
                                partial[j]
                            );
                        }
                    }
                }
            }

            float4* destination = reinterpret_cast<float4*>(
                &partial_output[group][dimension_start]
            );

            destination[0] = make_float4(
                partial[0], partial[1], partial[2], partial[3]
            );

            destination[1] = make_float4(
                partial[4], partial[5], partial[6], partial[7]
            );
        }

        __syncthreads();

        // Preserve the original reduction tree and online update.
        #pragma unroll 
        for (int part = 0; part < 2; ++part) {
            const int d = tid + part * kThreads;

            if (d < head_dim) {
                const float sum =
                    (
                        (partial_output[0][d] + partial_output[1][d])
                        + (partial_output[4][d] + partial_output[5][d])
                    )
                    + (
                        (partial_output[2][d] + partial_output[3][d])
                        + (partial_output[6][d] + partial_output[7][d])
                    );

                acc[part] = fmaf(acc[part], correction, sum);
            }
        }

        // The next tile's QK barrier protects the reads above before
        // partial_output and correction can be overwritten.
    }

    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int d = tid + part * kThreads;

        if (d < head_dim) {
            output[row * os0 + head * os1 + d * os2] =
                scalar_t(__fdividef(acc[part], denominator));
        }
    }
}


void paged_decode(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& rows,
    const at::Tensor& blocks,
    const at::Tensor& offsets,
    const at::Tensor& lengths,
    at::Tensor& output,
    double scale,
    int64_t block_size
) {
    // Real indices are CPU-validated; internal graph padding uses length=0.
    TORCH_CHECK(
        q.is_cuda() && q.dim() == 3,
        "Q must be a CUDA [tokens, heads, dim] tensor"
    );

    TORCH_CHECK(
        q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
        "Q must use float16 or bfloat16"
    );

    for (const auto& tensor : {k, v, output}) {
        TORCH_CHECK(
            tensor.device() == q.device()
                && tensor.scalar_type() == q.scalar_type(),
            "Q/K/V/output must share device and dtype"
        );
    }

    TORCH_CHECK(
        output.sizes() == q.sizes()
            && output.is_non_overlapping_and_dense(),
        "output must match Q shape and have non-overlapping dense storage"
    );

    TORCH_CHECK(
        k.dim() == 3
            && k.sizes() == v.sizes()
            && k.is_contiguous()
            && v.is_contiguous(),
        "K/V must be contiguous [slots, kv_heads, dim] tensors"
    );

    const auto dim = q.size(2);
    const auto heads = q.size(1);

    TORCH_CHECK(
        dim > 0
            && dim <= 256
            && k.size(2) == dim
            && heads > 0
            && k.size(1) > 0,
        "invalid attention head geometry"
    );

    TORCH_CHECK(
        heads % k.size(1) == 0
            && block_size > 0
            && block_size <= INT_MAX
            && k.size(0) % block_size == 0,
        "invalid GQA or block geometry"
    );

    for (const auto& tensor : {rows, blocks, offsets, lengths}) {
        TORCH_CHECK(
            tensor.device() == q.device()
                && tensor.scalar_type() == at::kLong
                && tensor.dim() == 1
                && tensor.is_contiguous(),
            "metadata must be contiguous CUDA int64 vectors on Q's device"
        );
    }

    TORCH_CHECK(
        rows.numel() == lengths.numel()
            && offsets.numel() == rows.numel() + 1,
        "metadata lengths must align"
    );

    TORCH_CHECK(
        rows.numel() <= INT_MAX / heads,
        "decode grid is too large"
    );

    if (rows.numel() == 0) {
        return;
    }

    const c10::cuda::CUDAGuard guard(q.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());

    int block_shift = -1;

    if ((block_size & (block_size - 1)) == 0) {
        block_shift = 0;

        for (int64_t size = block_size; size > 1; size >>= 1) {
            ++block_shift;
        }
    }

    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        q.scalar_type(),
        "paged_decode",
        [&] {
            auto launch = [&](auto dimension) {
                auto launch_layout = [&](auto power_of_two) {
                    auto launch_access = [&](auto vectorized) {
                        auto launch_prefetch = [&](auto prefetch) {
                            paged_decode_kernel<
                                scalar_t,
                                decltype(dimension)::value,
                                decltype(power_of_two)::value,
                                decltype(vectorized)::value,
                                decltype(prefetch)::value
                            ><<<
                                rows.numel() * heads,
                                kThreads,
                                0,
                                stream.stream()
                            >>>(
                                q.data_ptr<scalar_t>(),
                                k.data_ptr<scalar_t>(),
                                v.data_ptr<scalar_t>(),
                                output.data_ptr<scalar_t>(),
                                rows.data_ptr<int64_t>(),
                                blocks.data_ptr<int64_t>(),
                                offsets.data_ptr<int64_t>(),
                                lengths.data_ptr<int64_t>(),
                                heads,
                                k.size(1),
                                dim,
                                block_size,
                                block_shift,
                                float(scale),
                                q.stride(0),
                                q.stride(1),
                                q.stride(2),
                                output.stride(0),
                                output.stride(1),
                                output.stride(2)
                            );
                        };

                        if constexpr (
                            decltype(dimension)::value == 128
                            && decltype(vectorized)::value
                        ) {
                            if (dim == 128) {
                                launch_prefetch(std::true_type{});
                            } else {
                                launch_prefetch(std::false_type{});
                            }
                        } else {
                            launch_prefetch(std::false_type{});
                        }
                    };

                    const bool aligned =
                        dim % 8 == 0
                        && (
                            reinterpret_cast<uintptr_t>(
                                k.data_ptr<scalar_t>()
                            ) & 15
                        ) == 0
                        && (
                            reinterpret_cast<uintptr_t>(
                                v.data_ptr<scalar_t>()
                            ) & 15
                        ) == 0;

                    if (aligned) {
                        launch_access(std::true_type{});
                    } else {
                        launch_access(std::false_type{});
                    }
                };

                if (block_shift >= 0) {
                    launch_layout(std::true_type{});
                } else {
                    launch_layout(std::false_type{});
                }
            };

            if (dim <= 32) {
                launch(std::integral_constant<int, 32>{});
            } else if (dim <= 64) {
                launch(std::integral_constant<int, 64>{});
            } else if (dim <= 128) {
                launch(std::integral_constant<int, 128>{});
            } else {
                launch(std::integral_constant<int, 256>{});
            }
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}   // namespace


TORCH_LIBRARY(tinyvllm_cuda, m) {
    m.def(
        "paged_decode(Tensor q, Tensor k, Tensor v, Tensor rows, Tensor blocks, "
        "Tensor offsets, Tensor lengths, Tensor(a!) output, float scale, "
        "int block_size) -> ()"
    );
}


TORCH_LIBRARY_IMPL(tinyvllm_cuda, CUDA, m) {
    m.impl("paged_decode", &paged_decode);
}