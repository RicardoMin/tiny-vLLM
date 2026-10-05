#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>


namespace {

struct Rows {
    int64_t count;
    int64_t width;
    int64_t heads;
    int64_t token_stride;
    int64_t head_stride;
};


Rows check_rows(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "pointwise input must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf
            || x.scalar_type() == at::kBFloat16
            || x.scalar_type() == at::kFloat,
        "pointwise input must be float16, bfloat16 or float32"
    );
    TORCH_CHECK(
        x.layout() == at::kStrided && (x.dim() == 2 || x.dim() ==3),
        "pointwise input must be a strided 2D or 3D tensor"
    );
    TORCH_CHECK(
        x.size(-1) > 0 && x.stride(-1) == 1,
        "pointwise rows must have positive width and last stride 1"
    );

    return {
        x.numel() / x.size(-1),
        x.size(-1),
        x.dim() == 3 ? x.size(1) : 1,
        x.stride(0),
        x.dim() == 3 ? x.stride(1) : 0,
    };
}


__device__ __forceinline__ float warp_all_sum(float value) {
    for (int delta = 1; delta < 32; delta *= 2) {
        value += __shfl_xor_sync(0xffffffff, value, delta);
    }
    return value;
}


template <typename scalar_t, int ThreadsPerRow>
__global__ void rms_norm_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    Rows rows,
    float eps
) {
    constexpr int RowsPerBlock = 128 / ThreadsPerRow;
    const int64_t row = blockIdx.x * int64_t(RowsPerBlock)
        + threadIdx.x / ThreadsPerRow;

    if (row >= rows.count) {
        return;
    }

    const int column = threadIdx.x % ThreadsPerRow;
    const int lane = threadIdx.x % 32;
    const int64_t source = row / rows.heads * rows.token_stride
        + row % rows.heads * rows.head_stride;

    float sum = 0.0f;
    for (int64_t d = column; d < rows.width; d += ThreadsPerRow) {
        const float value = static_cast<float>(input[source + d]);
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    sum = warp_all_sum(sum);

    if constexpr (ThreadsPerRow == 128) {
        __shared__ float partial[4];
        if (lane == 0) {
            partial[threadIdx.x / 32] = sum;
        }
        __syncthreads();
        sum = lane < 4 ? partial[lane] : 0.0f;
        sum = warp_all_sum(sum);
    }

    const float variance = __fmul_rn(sum, 1.0f / float(rows.width));
    const float inverse_rms = rsqrtf(__fadd_rn(variance, eps));

    for (int64_t d = column; d < rows.width; d += ThreadsPerRow) {
        const float value = static_cast<float>(input[source + d]);
        // Qwen rounds the normalized value before multiplying by the weight.
        const scalar_t normalized = static_cast<scalar_t>(value * inverse_rms);
        output[row * rows.width + d] = static_cast<scalar_t>(
            static_cast<float>(normalized) * static_cast<float>(weight[d])
        );
    }
}


template <typename scalar_t>
__global__ void silu_and_mul_kernel(
    const scalar_t* input,
    scalar_t* output,
    Rows rows,
    int64_t tiles_per_row,
    int64_t tiles
) {
    const int64_t width = rows.width / 2;
    for (
        int64_t tile = blockIdx.x;
        tile < tiles;
        tile += gridDim.x
    ) {
        const int64_t row = tile / tiles_per_row;
        const int64_t column = tile % tiles_per_row * blockDim.x + threadIdx.x;
        if (column >= width) {
            continue;
        }
        const int64_t source = row / rows.heads * rows.token_stride
            + row % rows.heads * rows.head_stride + column;

        const float gate = static_cast<float>(input[source]);
        const float up = static_cast<float>(input[source + width]);
        // Match F.silu(gate) * up, including its Low-precision intermediate.
        const scalar_t activated = static_cast<scalar_t>(
            gate / (1.0f + expf(-gate))
        );
        output[row * width + column] = static_cast<scalar_t>(
            static_cast<float>(activated) * up
        );
    }
}


at::Tensor rms_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    double eps
) {
    const Rows rows = check_rows(input);
    TORCH_CHECK(
        weight.device() == input.device()
            && weight.scalar_type() == input.scalar_type()
            && weight.dim() == 1
            && weight.numel() == rows.width
            && weight.is_contiguous(),
        "RMSNorm weight must be contiguous [width] with input device and dtype"
    );
    TORCH_CHECK(rows.count <= INT_MAX, "too many RMSNorm rows");

    const c10::cuda::CUDAGuard guard(input.device());
    auto output = at::empty(input.sizes(), input.options());
    if (rows.count == 0) {
        return output;
    }
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "rms_norm", [&] {
            if (rows.width <= 256) {
                rms_norm_kernel<scalar_t, 32><<<
                    (rows.count + 3) / 4, 128, 0, stream.stream()
                >>>(
                    input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(), rows, static_cast<float>(eps)
                );
            } else {
                rms_norm_kernel<scalar_t, 128><<<
                    rows.count, 128, 0, stream.stream()
                >>>(
                    input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(), rows, static_cast<float>(eps)
                );
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor silu_and_mul(const at::Tensor& input) {
    const Rows rows = check_rows(input);
    TORCH_CHECK(rows.width % 2 == 0, "SilU input width must be even");

    const c10::cuda::CUDAGuard guard(input.device());
    auto shape = input.sizes().vec();
    shape.back() /= 2;
    auto output = at::empty(shape, input.options());
    const int64_t count = output.numel();
    if (count == 0) {
        return output;
    }
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
    const int64_t tiles_per_row = (rows.width / 2 + 255) / 256;
    const int64_t tiles = rows.count * tiles_per_row;
    const int blocks = std::min<int64_t>(tiles, 65535);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "silu_and_mul", [&] {
            silu_and_mul_kernel<scalar_t><<<blocks, 256, 0, stream.stream()>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                rows, tiles_per_row, tiles
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}   // namespace


TORCH_LIBRARY_FRAGMENT(tinyvllm_cuda, m) {
    m.def("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
    m.def("silu_and_mul(Tensor input) -> Tensor");
}


TORCH_LIBRARY_IMPL(tinyvllm_cuda, CUDA, m) {
    m.impl("rms_norm", &rms_norm);
    m.impl("silu_and_mul", &silu_and_mul);
}