// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/moe/moe_fused_swiglu.h"
#include <algorithm>
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"

namespace onnxruntime::contrib::cuda {

// SwiGLU with interleaved is like the following python code using PyTorch:
//   dim = x.shape[-1]
//   x = x.view(-1, dim // 2, 2)
//   x_glu, x_linear = x[..., 0], x[..., 1]
//   y = x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + 1)
template <typename T, bool HasLimit>
__global__ void swiglu_kernel_interleaved(T* output, T const* input, int intermediate_size, int num_rows, float alpha, float limit) {
  int const row = blockIdx.x;
  if (row >= num_rows) {
    return;
  }

  T const* row_input = input + row * 2 * intermediate_size;
  T* row_output = output + row * intermediate_size;

  for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
    float glu = static_cast<float>(row_input[2 * i]);
    float linear = static_cast<float>(row_input[2 * i + 1]);

    if constexpr (HasLimit) {
      glu = fminf(glu, limit);
      linear = fminf(fmaxf(linear, -limit), limit);
    }

    float sigmoid_arg = alpha * glu;
    float sigmoid_out = 1.f / (1.f + expf(-sigmoid_arg));

    float swish_out = glu * sigmoid_out;
    row_output[i] = static_cast<T>(swish_out * (linear + 1.f));
  }
}

// Non interleaved version of SwiGLU kernel, which splits each row into two chunks of same size.
template <typename T, bool HasLimit>
__global__ void swiglu_kernel_chunked(T* output, T const* input, int intermediate_size, int num_rows, float alpha, float limit) {
  int const row = blockIdx.x;
  if (row >= num_rows) {
    return;
  }

  T const* row_input = input + row * 2 * intermediate_size;
  T* row_output = output + row * intermediate_size;

  for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
    float glu = static_cast<float>(row_input[i]);
    float linear = static_cast<float>(row_input[i + intermediate_size]);

    if constexpr (HasLimit) {
      glu = fminf(glu, limit);
      linear = fminf(fmaxf(linear, -limit), limit);
    }

    float sigmoid_arg = alpha * glu;
    float sigmoid_out = 1.f / (1.f + expf(-sigmoid_arg));

    float swish_out = glu * sigmoid_out;
    row_output[i] = static_cast<T>(swish_out * (linear + 1.f));
  }
}

template <typename T, bool IsInterLeaved, bool HasLimit>
void invokeSwiGLU(T* output, T const* input, int intermediate_size, int num_rows, float alpha, float limit, cudaStream_t stream) {
  if (num_rows == 0) {
    return;
  }
  dim3 block(std::min(intermediate_size, 1024));
  dim3 grid(num_rows);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR("swiglu input", input, num_rows, 2 * intermediate_size);

  if constexpr (IsInterLeaved) {
    swiglu_kernel_interleaved<T, HasLimit><<<grid, block, 0, stream>>>(output, input, intermediate_size, num_rows, alpha, limit);
  } else {
    swiglu_kernel_chunked<T, HasLimit><<<grid, block, 0, stream>>>(output, input, intermediate_size, num_rows, alpha, limit);
  }

  DUMP_TENSOR("swiglu output", output, num_rows, intermediate_size);
}

template void invokeSwiGLU<float, true, true>(float*, float const*, int, int, float, float, cudaStream_t);
template void invokeSwiGLU<half, true, true>(half*, half const*, int, int, float, float, cudaStream_t);
template void invokeSwiGLU<__nv_bfloat16, true, true>(__nv_bfloat16*, __nv_bfloat16 const*, int, int, float, float, cudaStream_t);

}  // namespace onnxruntime::contrib::cuda