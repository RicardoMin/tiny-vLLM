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
#include <tuple>


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


template <typename scalar_t, int Width, int Threads, bool AddResidual>
__global__ void rms_norm_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* residual,
    scalar_t* output,
    scalar_t* residual_output,
    Rows rows,
    int64_t residual_stride,
    float eps
) {
    const int rows_per_block = blockDim.x / Threads;
    const int64_t row = int64_t(blockIdx.x) * rows_per_block
        + threadIdx.x / Threads; 
    if (row >= rows.count) {
        return;
    }

    const int column = threadIdx.x % Threads;
    const int width = Width == 0 ? int(rows.width) : Width;
    const int lane = threadIdx.x % 32;
    const int64_t source = row / rows.heads * rows.token_stride
        + row % rows.heads * rows.head_stride;
    const int64_t target = int64_t(row) * width;
    float values[Width == 0 ? 1 : Width / Threads];

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < (width + Threads - 1) / Threads; ++i) {
        const int d = i * Threads + column;
        if (d < width) {
            scalar_t value = input[source + d];
            if constexpr (AddResidual) {
                value = static_cast<scalar_t>(__fadd_rn(
                    static_cast<float>(value),
                    static_cast<float>(residual[int64_t(row) * residual_stride + d])
                ));
            }
            const float x = static_cast<float>(value);
            if constexpr (Width != 0) {
                values[i] = x;
            }
            sum = __fadd_rn(sum, __fmul_rn(x, x));
        }
    }
    sum = warp_all_sum(sum);

    if constexpr (Threads == 128) {
        __shared__ float partial[4];
        if (lane == 0) {
            partial[threadIdx.x / 32] = sum;
        }
        __syncthreads();
        sum = lane < 4 ? partial[lane] : 0.0f;
        sum = warp_all_sum(sum);
    }

    const float variance = __fmul_rn(sum, 1.0f / float(width));
    const float inverse_rms = rsqrtf(__fadd_rn(variance, eps));

    #pragma unroll
    for (int i = 0; i < (width + Threads - 1) / Threads; ++i) {
        const int d = i * Threads + column;
        if (d < width) {
            float value;
            if constexpr (Width != 0) {
                value = values[i];
            } else if constexpr (AddResidual) {
                const scalar_t merged = static_cast<scalar_t>(__fadd_rn(
                    static_cast<float>(input[source + d]),
                    static_cast<float>(residual[row * residual_stride + d])
                ));
                value = static_cast<float>(merged);
            } else {
                value = static_cast<float>(input[source + d]);
            }
            if constexpr (AddResidual) {
                residual_output[target + d] = static_cast<scalar_t>(value);
            }
            const scalar_t normalized = static_cast<scalar_t>(
                __fmul_rn(value, inverse_rms
            ));
            output[target + d] = static_cast<scalar_t>(__fmul_rn(
                static_cast<float>(normalized), static_cast<float>(weight[d])
            ));
        }
    }
}


template <typename scalar_t, bool AddResidual>
void launch_rms_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& residual,
    at::Tensor& output,
    at::Tensor& residual_output,
    Rows rows,
    float eps
) {
    const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
    const auto* x = input.data_ptr<scalar_t>();
    const auto* w = weight.data_ptr<scalar_t>();
    const auto* r = AddResidual ? residual.data_ptr<scalar_t>() : nullptr;
    auto* y = output.data_ptr<scalar_t>();
    auto* merged = AddResidual ? residual_output.data_ptr<scalar_t>() : nullptr;
    const int64_t stride = AddResidual ? residual.stride(0) : 0;

    const int small_threads = rows.count < 1024 ? 32 : 128;
    const int rows_per_block = small_threads / 32;
    const int small_blocks = (rows.count + rows_per_block - 1) / rows_per_block;

    if (rows.width == 128) {
        rms_norm_kernel<scalar_t, 128, 32, AddResidual><<<
            small_blocks, small_threads, 0, stream.stream()
        >>>(x, w, r, y, merged, rows, stride, eps);
    } else if (rows.width == 1024) {
        rms_norm_kernel<scalar_t, 1024, 128, AddResidual><<<
            rows.count, 128, 0, stream.stream()
        >>>(x, w, r, y, merged, rows, stride, eps);
    } else if (rows.width <= 256) {
        rms_norm_kernel<scalar_t, 0, 32, AddResidual><<<
            small_blocks, small_threads, 0, stream.stream()
        >>>(x, w, r, y, merged, rows, stride, eps);
    } else {
        rms_norm_kernel<scalar_t, 0, 128, AddResidual><<<
            rows.count, 128, 0, stream.stream()
        >>>(x, w, r, y, merged, rows, stride, eps);
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


template <typename scalar_t>
__global__ void apply_rope_kernel(
    const scalar_t* input,
    const scalar_t* cos,
    const scalar_t* sin,
    scalar_t* output,
    Rows rows
) {
    const int64_t half = rows.width / 2;
    const int64_t pairs = rows.count * half;

    for (
        int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        index < pairs;
        index += int64_t(blockDim.x) * gridDim.x
    ) {
        const int64_t row = index / half;
        const int64_t column = index % half;
        const int64_t token = row / rows.heads;

        const int64_t source = token * rows.token_stride
            + row % rows.heads * rows.head_stride + column;

        const int64_t target = row * rows.width + column;
        const int64_t angle = token * half + column;

        const float first = static_cast<float>(input[source]);
        const float second = static_cast<float>(input[source + half]);
        const float c = static_cast<float>(cos[angle]);
        const float s = static_cast<float>(sin[angle]);

        // Preserve the four materialized products of the PyTorch expression.
        const scalar_t first_cos = static_cast<scalar_t>(
            __fmul_rn(first, c)
        );
        const scalar_t second_sin = static_cast<scalar_t>(
            __fmul_rn(second, s)
        );
        const scalar_t second_cos = static_cast<scalar_t>(
            __fmul_rn(second, c)
        );
        const scalar_t first_sin = static_cast<scalar_t>(
            __fmul_rn(first, s)
        );

        output[target] = static_cast<scalar_t>(__fsub_rn(
            static_cast<float>(first_cos),
            static_cast<float>(second_sin)
        ));
        
        output[target + half] = static_cast<scalar_t>(__fadd_rn(
            static_cast<float>(second_cos),
            static_cast<float>(first_sin)
        ));
    }
}


std::tuple<at::Tensor, at::Tensor> rms_norm_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    double eps,
    const at::Tensor& residual
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
    TORCH_CHECK(
        rows.count <= INT_MAX && rows.width <= INT_MAX - 127,
        "RMSNorm dimensions exceed CUDA index range"
    );
    const bool add_residual = residual.defined();
    if (add_residual) {
        TORCH_CHECK(
            input.dim() == 2
                && residual.device() == input.device()
                && residual.scalar_type() == input.scalar_type()
                && residual.layout() == at::kStrided
                && residual.sizes() == input.sizes()
                && residual.stride(-1) == 1,
            "residual must match the 2D input shape, device and dtype, with last stride 1"
        );
    }

    const c10::cuda::CUDAGuard guard(input.device());
    auto output = at::empty(input.sizes(), input.options());
    auto merged = add_residual
        ? at::empty(input.sizes(), input.options()) : at::Tensor();
    if (rows.count == 0) {
        return {output, merged};
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "rms_norm", [&] {
            if (add_residual) {
                launch_rms_norm<scalar_t, true>(
                    input, weight, residual, output, merged, rows, float(eps)
                );
            } else {
                launch_rms_norm<scalar_t, false>(
                    input, weight, residual, output, merged, rows, float(eps)
                );
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, merged};
}


at::Tensor rms_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    double eps
) {
    return std::get<0>(rms_norm_impl(input, weight, eps, at::Tensor()));
}


std::tuple<at::Tensor, at::Tensor> add_rms_norm(
    const at::Tensor& input,
    const at::Tensor& residual,
    const at::Tensor& weight,
    double eps
) {
    return rms_norm_impl(input, weight, eps, residual);
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


at::Tensor apply_rope(
    const at::Tensor& input,
    const at::Tensor& cos,
    const at::Tensor& sin
) {
    const Rows rows = check_rows(input);

    TORCH_CHECK(
        input.dim() == 3 && rows.width % 2 == 0,
        "RoPE input must be [tokens, heads, even head_dim]"
    );

    for (const auto* table : {&cos, &sin}) {
        TORCH_CHECK(
            table->device() == input.device()
                && table->scalar_type() == input.scalar_type()
                && table->dim() == 3
                && table->size(0) == input.size(0)
                && table->size(1) == 1
                && table->size(2) == rows.width / 2
                && table->is_contiguous(),
            "RoPE cos/sin must be contiguous [tokens, 1, head_dim/2] "
            "with input device and dtype"
        );
    }

    const c10::cuda::CUDAGuard guard(input.device());
    auto output = at::empty(input.sizes(), input.options());

    const int64_t pairs = input.numel() / 2;
    if (pairs == 0) {
        return output;
    }

    const auto stream = c10::cuda::getCurrentCUDAStream(
        input.get_device()
    );
    const int blocks = std::min<int64_t>(
        (pairs + 127) / 128,
        65535
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        input.scalar_type(),
        "apply_rope",
        [&] {
            apply_rope_kernel<scalar_t><<<
                blocks, 128, 0, stream.stream()
            >>>(
                input.data_ptr<scalar_t>(),
                cos.data_ptr<scalar_t>(),
                sin.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                rows
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}   // namespace


TORCH_LIBRARY_FRAGMENT(tinyvllm_cuda, m) {
    m.def("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
    m.def("add_rms_norm(Tensor input, Tensor residual, Tensor weight, float eps) -> (Tensor, Tensor)");
    m.def("silu_and_mul(Tensor input) -> Tensor");
    m.def("apply_rope(Tensor input, Tensor cos, Tensor sin) -> Tensor");
}


TORCH_LIBRARY_IMPL(tinyvllm_cuda, CUDA, m) {
    m.impl("rms_norm", &rms_norm);
    m.impl("add_rms_norm", &add_rms_norm);
    m.impl("silu_and_mul", &silu_and_mul);
    m.impl("apply_rope", &apply_rope);
}