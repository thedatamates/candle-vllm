#include <cmath>

#include "cuda_compat.h"
#include "attention/attention_dtypes.h"

namespace vllm {

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}
// Activation and gating kernel template.

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

}  // namespace vllm

extern "C" void silu_and_mul(
    void *out,              // [..., d]
    void *input,            // [..., 2 * d]
    int32_t dims,          // 
    int32_t num_tokens,       //
    uint32_t dtype,         // 0 => f16; 1 => bf16; 2 => f32
    int32_t stream)
{
    // int d = input.size(-1) / 2;
    // int64_t num_tokens = input.numel() / input.size(-1);
    dim3 grid(num_tokens);
    dim3 block(std::min(dims, 1024));
    const cudaStream_t stream_ = (cudaStream_t)stream;

    if (dtype == 2) {
        vllm::act_and_mul_kernel<float, vllm::silu_kernel<float>, true>
            <<<grid, block, 0, stream_>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(input), dims);
    } else if (dtype == 0) {
        vllm::act_and_mul_kernel<uint16_t, vllm::silu_kernel<uint16_t>, true>
            <<<grid, block, 0, stream_>>>(reinterpret_cast<uint16_t *>(out), reinterpret_cast<uint16_t *>(input), dims);
    } else if (dtype == 1) {
        vllm::act_and_mul_kernel<__nv_bfloat16, vllm::silu_kernel<__nv_bfloat16>, true>
            <<<grid, block, 0, stream_>>>(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<__nv_bfloat16 *>(input), dims);
    }
}
