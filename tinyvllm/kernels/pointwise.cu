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
#include <type_traits>


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


void check_norm_weight(const at::Tensor& input, const at::Tensor& weight) {
    TORCH_CHECK(
        weight.device() == input.device()
            && weight.scalar_type() == input.scalar_type()
            && weight.dim() == 1
            && weight.numel() == input.size(-1)
            && weight.is_contiguous(),
        "RMSNorm weight must be contiguous [width] with input device and dtype"
    );
}


void check_rope_tables(
    const at::Tensor& input,
    const at::Tensor& cos,
    const at::Tensor& sin
) {
    TORCH_CHECK(
        input.dim() == 3 && input.size(-1) % 2 == 0,
        "RoPE input must be [tokens, heads, even head_dim]"
    );

    for (const auto* table : {&cos, &sin}) {
        TORCH_CHECK(
            table->device() == input.device()
                && table->scalar_type() == input.scalar_type()
                && table->dim() == 3
                && table->size(0) == input.size(0)
                && table->size(1) == 1
                && table->size(2) == input.size(-1) / 2
                && table->is_contiguous(),
            "RoPE cos/sin must be contiguous [tokens, 1, head_dim/2] "
            "with input device and dtype"
        );
    }
}


template <typename scalar_t>
__device__ __forceinline__ void rotate_pair(
    float first,
    float second,
    float cos,
    float sin,
    scalar_t* first_output,
    scalar_t* second_output
) {
    // Preserve the four materialized products of the PyTorch expression.
    const scalar_t first_cos = static_cast<scalar_t>(__fmul_rn(first, cos));
    const scalar_t second_sin = static_cast<scalar_t>(__fmul_rn(second, sin));
    const scalar_t second_cos = static_cast<scalar_t>(__fmul_rn(second, cos));
    const scalar_t first_sin = static_cast<scalar_t>(__fmul_rn(first, sin));

    *first_output = static_cast<scalar_t>(__fsub_rn(
        static_cast<float>(first_cos), static_cast<float>(second_sin)
    ));

    *second_output = static_cast<scalar_t>(__fadd_rn(
        static_cast<float>(second_cos), static_cast<float>(first_sin)
    ));
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
__global__ void silu_and_mul_single_head_kernel(
    const scalar_t* input,
    scalar_t* output,
    Rows rows
) {
    // A 2D input, or a 3D input with one head, maps each row to one token.
    const int64_t row = blockIdx.y;
    const int64_t column = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t width = rows.width / 2;

    if (column >= width) {
        return;
    }

    const int64_t source = row * rows.token_stride + column;
    const float gate = static_cast<float>(input[source]);
    const float up = static_cast<float>(input[source + width]);

    // Match F.silu(gate) * up, including its low-precision intermediate.
    const scalar_t activated = static_cast<scalar_t>(
        gate / (1.0f + expf(-gate))
    );

    output[row * width + column] = static_cast<scalar_t>(
        static_cast<float>(activated) * up
    );
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

        rotate_pair(
            first, second, c, s, output + target, output + target + half
        );
    }
}


template <typename scalar_t, int Width, int Threads>
__global__ void qk_norm_rope_kernel(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* query_weight,
    const scalar_t* key_weight,
    const scalar_t* cos,
    const scalar_t* sin,
    scalar_t* query_output,
    scalar_t* key_output,
    Rows query_rows,
    Rows key_rows,
    float query_eps,
    float key_eps
) {
    const int rows_per_block = blockDim.x / Threads;
    const int64_t global_row = int64_t(blockIdx.x) * rows_per_block
        + threadIdx.x / Threads;

    if (global_row >= query_rows.count + key_rows.count) {
        return;
    }

    // Each warp (or whole CTA for a wide head) owns exactly one Q/K row.
    const bool is_query = global_row < query_rows.count;
    const Rows rows = is_query ? query_rows : key_rows;
    const int64_t row = is_query ? global_row : global_row - query_rows.count;

    const scalar_t* input = is_query ? query : key;
    const scalar_t* weight = is_query ? query_weight : key_weight;
    scalar_t* output = is_query ? query_output : key_output;
    const float eps = is_query ? query_eps : key_eps;

    const int width = Width == 0 ? int(rows.width) : Width;
    const int half = width / 2;
    const int column = threadIdx.x % Threads;
    const int lane = threadIdx.x % 32;

    const int64_t token = row / rows.heads;
    const int64_t source = token * rows.token_stride
        + row % rows.heads * rows.head_stride;

    float values[Width == 0 ? 1 : Width / Threads];

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < (width + Threads - 1)/ Threads; ++i) {
        const int d = i * Threads + column;

        if (d < width) {
            const float value = static_cast<float>(input[source + d]);

            if constexpr (Width != 0) {
                values[i] = value;
            }

            sum = __fadd_rn(sum, __fmul_rn(value, value));
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
    for (int i = 0; i < (half + Threads - 1) / Threads; ++ i) {
        const int d = i * Threads + column;

        if (d < half) {
            float first, second;

            if constexpr (Width == 128) {
                first = values[i];
                second = values[i + 2];
            } else {
                first = static_cast<float>(input[source + d]);
                second = static_cast<float>(input[source + d + half]);
            }

            // Match RMSNorm's two low-precision casts before rotating.
            const scalar_t first_norm = static_cast<scalar_t>(
                __fmul_rn(first, inverse_rms)
            );

            const scalar_t second_norm = static_cast<scalar_t>(
                __fmul_rn(second, inverse_rms)
            );

            const scalar_t first_weighted = static_cast<scalar_t>(__fmul_rn(
                static_cast<float>(first_norm),
                static_cast<float>(weight[d])
            ));

            const scalar_t second_weighted = static_cast<scalar_t>(__fmul_rn(
                static_cast<float>(second_norm),
                static_cast<float>(weight[d + half])
            ));

            const int64_t angle = token * half + d;
            const int64_t target = row * width + d;

            rotate_pair(
                static_cast<float>(first_weighted),
                static_cast<float>(second_weighted),
                static_cast<float>(cos[angle]),
                static_cast<float>(sin[angle]),
                output + target,
                output + target + half
            );
        }
    }
}


std::tuple<at::Tensor, at::Tensor> qk_norm_rope(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& query_weight,
    const at::Tensor& key_weight,
    const at::Tensor& cos,
    const at::Tensor& sin,
    double query_eps,
    double key_eps
) {
    const Rows qr = check_rows(query);
    const Rows kr = check_rows(key);

    TORCH_CHECK(
        query.dim() == 3 && key.dim() == 3
            && query.size(0) == key.size(0)
            && qr.heads > 0 && kr.heads > 0 && qr.width == kr.width
            && key.device() == query.device()
            && key.scalar_type() == query.scalar_type(),
        "Q/K must match token count, head width, device and dtype"
    );

    check_norm_weight(query, query_weight);
    check_norm_weight(key, key_weight);
    check_rope_tables(query, cos, sin);

    TORCH_CHECK(
        qr.count <= INT_MAX && kr.count <= INT_MAX - qr.count
            && qr.width <= INT_MAX - 127,
        "Q/K RMSNorm dimensions exceed CUDA index range"
    );

    const int64_t rows = qr.count + kr.count;

    const c10::cuda::CUDAGuard guard(query.device());

    auto query_output = at::empty(query.sizes(), query.options());
    auto key_output = at::empty(key.sizes(), key.options());

    if (rows == 0) {
        return {query_output, key_output};
    }

    const auto stream = c10::cuda::getCurrentCUDAStream(query.get_device());
    const int threads = qr.width > 256 || rows >= 1024 ? 128 : 32;
    const int rows_per_block = qr.width > 256 ? 1 : threads / 32;
    const int64_t blocks = (rows + rows_per_block - 1) / rows_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, query.scalar_type(), "qk_norm_rope", [&] {
            auto launch = [&](auto width, auto lanes) {
                qk_norm_rope_kernel<
                    scalar_t,
                    decltype(width)::value,
                    decltype(lanes)::value
                ><<<blocks, threads, 0, stream.stream()>>>(
                     query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    query_weight.data_ptr<scalar_t>(),
                    key_weight.data_ptr<scalar_t>(),
                    cos.data_ptr<scalar_t>(),
                    sin.data_ptr<scalar_t>(),
                    query_output.data_ptr<scalar_t>(),
                    key_output.data_ptr<scalar_t>(),
                    qr,
                    kr,
                    float(query_eps),
                    float(key_eps)
                );
            };

            if (qr.width == 128) {
                launch(
                    std::integral_constant<int, 128>{},
                    std::integral_constant<int, 32>{}
                );
            } else if (qr.width <= 256) {
                launch(
                    std::integral_constant<int, 0>{},
                    std::integral_constant<int, 32>{}
                );
            } else {
                launch(
                    std::integral_constant<int, 0>{},
                    std::integral_constant<int, 128>{}
                );
            }
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {query_output, key_output};
}


std::tuple<at::Tensor, at::Tensor> rms_norm_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    double eps,
    const at::Tensor& residual
) {
    const Rows rows = check_rows(input);
    check_norm_weight(input, weight);
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

    const bool single_head = rows.heads == 1
        && rows.count <= 65535
        && tiles_per_row <= INT_MAX;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "silu_and_mul", [&] {
            if (single_head) {
                const dim3 grid(
                    static_cast<unsigned int>(tiles_per_row),
                    static_cast<unsigned int>(rows.count)
                );

                silu_and_mul_single_head_kernel<scalar_t><<<
                    grid, 256, 0, stream.stream()
                >>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    rows
                );
            } else {
                const int64_t tiles = rows.count * tiles_per_row;
                const int blocks = std::min<int64_t>(tiles, 65535);

                silu_and_mul_kernel<scalar_t><<<
                    blocks, 256, 0, stream.stream()
                >>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    rows,
                    tiles_per_row,
                    tiles
                );
            }
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

    check_rope_tables(input, cos, sin);

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
    m.def(
        "qk_norm_rope(Tensor query, Tensor key, Tensor query_weight, "
        "Tensor key_weight, Tensor cos, Tensor sin, float query_eps, "
        "float key_eps) -> (Tensor, Tensor)"
    );
}


TORCH_LIBRARY_IMPL(tinyvllm_cuda, CUDA, m) {
    m.impl("rms_norm", &rms_norm);
    m.impl("add_rms_norm", &add_rms_norm);
    m.impl("silu_and_mul", &silu_and_mul);
    m.impl("apply_rope", &apply_rope);
    m.impl("qk_norm_rope", &qk_norm_rope);
}