#include <ATen/ATen.h>
#include <torch/library.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <type_traits>


namespace {

constexpr int kThreads = 128;

struct Strides {
    int64_t token;
    int64_t head;
    int64_t dim;
};


template <bool Vectorized>
__global__ void store_kv_kernel(
    const uint16_t* key,
    const uint16_t* value,
    uint16_t* key_cache,
    uint16_t* value_cache,
    const int64_t* slots,
    int64_t width,
    int64_t head_dim,
    Strides ks,
    Strides vs
) {
    const int64_t token = blockIdx.x;
    const int64_t destination = slots[token] * width;

    if constexpr (Vectorized) {
        for (
            int64_t pack = threadIdx.x;
            pack < width / 8;
            pack += blockDim.x
        ) {
            const int64_t offset = pack * 8;

            *reinterpret_cast<uint4*>(key_cache + destination + offset) =
                *reinterpret_cast<const uint4*>(
                    key + token * ks.token + offset
                );

            *reinterpret_cast<uint4*>(value_cache + destination + offset) = 
                *reinterpret_cast<const uint4*>(
                    value + token * vs.token + offset
                );
        }
    } else {
        for (
            int64_t offset = threadIdx.x;
            offset < width;
            offset += blockDim.x
        ) {
            const int64_t head = offset / head_dim;
            const int64_t dim = offset % head_dim;

            key_cache[destination + offset] =
                key[
                    token * ks.token
                    + head * ks.head
                    + dim * ks.dim
                ];

            value_cache[destination + offset] =
                value[
                    token * vs.token
                    + head * vs.head
                    + dim * vs.dim
                ];
        }
    }
}


bool packed_rows(const at::Tensor& tensor) {
    return tensor.stride(2) == 1
        && tensor.stride(1) == tensor.size(2)
        && tensor.stride(0) % 8 == 0
        && (reinterpret_cast<uintptr_t>(tensor.data_ptr()) & 15) == 0;
}


void store_kv(
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& slots
) {
    TORCH_CHECK(key.is_cuda(), "KV store requires CUDA tensors");

    TORCH_CHECK(
        key.scalar_type() == at::kHalf
            || key.scalar_type() == at::kBFloat16,
        "KV store requires float16 or bfloat16"
    );

    for (const auto& tensor : {key, value, key_cache, value_cache}) {
        TORCH_CHECK(
            tensor.device() == key.device(),
            "KV device mismatch"
        );
        TORCH_CHECK(
            tensor.scalar_type() == key.scalar_type(),
            "KV dtype mismatch"
        );
        TORCH_CHECK(
            tensor.dim() == 3,
            "KV tensors must have three dimensions"
        );
    }

    TORCH_CHECK(
        value.sizes() == key.sizes(),
        "K/V shape mismatch"
    );
    TORCH_CHECK(
        key_cache.sizes() == value_cache.sizes(),
        "cache shape mismatch"
    );
    TORCH_CHECK(
        key.size(1) > 0
            && key.size(2) > 0
            && key.size(1) == key_cache.size(1)
            && key.size(2) == key_cache.size(2),
        "KV head geometry mismatch"
    );
    TORCH_CHECK(
        key_cache.is_contiguous() && value_cache.is_contiguous(),
        "KV caches must be contiguous"
    );
    TORCH_CHECK(
        slots.device() == key.device()
            && slots.scalar_type() == at::kLong
            && slots.dim() == 1
            && slots.is_contiguous()
            && slots.numel() == key.size(0),
        "slots must be contiguous CUDA int64 matching token rows"
    );
    TORCH_CHECK(
        slots.numel() <= key_cache.size(0)
            && slots.numel() <= INT_MAX,
        "too many KV rows"
    );

    TORCH_CHECK(
        !key_cache.is_alias_of(value_cache),
        "K/V caches must use separate storage"
    );

    for (const auto& source : {key, value, slots}) {
        TORCH_CHECK(
            !key_cache.is_alias_of(source)
                && !value_cache.is_alias_of(source),
            "KV inputs must not share cache storage"
        );
    }

    // Slot values and uniqueness belong to the validated CPU batch contract.
    if (slots.numel() == 0) {
        return;
    }

    const c10::cuda::CUDAGuard guard(key.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(
        key.get_device()
    );

    const int64_t width = key.size(1) * key.size(2);

    const Strides ks{
        key.stride(0),
        key.stride(1),
        key.stride(2)
    };
    const Strides vs{
        value.stride(0),
        value.stride(1),
        value.stride(2)
    };

    const bool vectorized = width % 8 == 0
        && packed_rows(key)
        && packed_rows(value)
        && packed_rows(key_cache)
        && packed_rows(value_cache);

    auto launch = [&](auto packed) {
        store_kv_kernel<decltype(packed)::value><<<
            slots.numel(),
            kThreads,
            0,
            stream.stream()
        >>>(
            reinterpret_cast<const uint16_t*>(key.data_ptr()),
            reinterpret_cast<const uint16_t*>(value.data_ptr()),
            reinterpret_cast<uint16_t*>(key_cache.data_ptr()),
            reinterpret_cast<uint16_t*>(value_cache.data_ptr()),
            slots.data_ptr<int64_t>(),
            width,
            key.size(2),
            ks,
            vs
        );
    };

    if (vectorized) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}   // namespace


TORCH_LIBRARY_FRAGMENT(tinyvllm_cuda, m) {
    m.def(
        "store_kv(Tensor key, Tensor value, Tensor(a!) key_cache, "
        "Tensor(b!) value_cache, Tensor slots) -> ()"
    );
}


TORCH_LIBRARY_IMPL(tinyvllm_cuda, CUDA, m) {
    m.impl("store_kv", &store_kv);
}