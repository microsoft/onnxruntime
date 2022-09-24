#include "hip/hip_runtime.h"
/*
 The implementation of this file is based on skipLayerNorm plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Add SkipLayerNormKernelVec to
//                leverage vectorized load/write.
//                and templatize ComputeSkipLayerNorm for different
//                data types.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/skip_layer_norm_impl.h"

#include <hip/hip_fp16.h>

#include "contrib_ops/rocm/bert/skip_layer_norm_impl_kernel.h"
#include "contrib_ops/rocm/bert/skip_layer_norm_tunable_op.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
Status LaunchSkipLayerNormKernel(
    hipStream_t stream, T* output, const T* input, const T* skip, const T* gamma,
    const T* beta, const T* bias, float epsilon, int ld, int element_count, bool tuning) {
  // this must be true because element_count is the total size of the tensor
  assert(element_count % ld == 0);

  if (tuning) {
    static SkipLayerNormTunableOp<T> op;
    op.EnableTuning();

    SkipLayerNormParams<T> op_params(stream, output, input, skip, gamma, beta, bias, epsilon, ld, element_count);
    return op(&op_params);
  }

  bool hasBias = (bias == nullptr) ? false : true;
  if (0 == (ld % 4)) {
    const int grid_size = element_count / ld;
    if (ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall<T, block_size, 1><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output, hasBias);
    } else if (ld <= 64) {
      constexpr int block_size = 64 / 2;
      SkipLayerNormKernelSmall<T, block_size, 2><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output, hasBias);
    } else if (ld <= 128) {
      constexpr int block_size = 128 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output, hasBias);
    } else if (ld <= 384) {
      constexpr int block_size = 384 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output, hasBias);
    } else if (ld <= 768) {
      constexpr int block_size = 768 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output, hasBias);
    } else if (ld <= 1024) {
      constexpr int block_size = 1024 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output, hasBias);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<T, block_size><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output);
    }
  } else {
    const int grid_size = element_count / ld;
    if (ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall<T, block_size, 1><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output, hasBias);
    } else if (ld <= 64) {
      constexpr int block_size = 64;
      SkipLayerNormKernelSmall<T, block_size, 1><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output, hasBias);
    } else if (ld <= 128) {
      constexpr int block_size = 128;
      SkipLayerNormKernelSmall<T, block_size, 1><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output, hasBias);
    } else if (ld == 384) {
      constexpr int block_size = 384;
      SkipLayerNormKernelSmall<T, block_size, 1><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output, hasBias);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<T, block_size><<<grid_size, block_size, 0, stream>>>(
          ld, input, skip, beta, gamma, bias, maybe2half<T>(epsilon), output);
    }
  }
  return HIP_CALL(hipPeekAtLastError());
}

template Status LaunchSkipLayerNormKernel<float>(hipStream_t stream, float* output, const float* input,
                                                 const float* skip, const float* gamma, const float* beta,
                                                 const float* bias, float epsilon, int ld,
                                                 int element_count, bool tuning);

template Status LaunchSkipLayerNormKernel<half>(hipStream_t stream, half* output, const half* input,
                                                const half* skip, const half* gamma, const half* beta,
                                                const half* bias, float epsilon, int ld,
                                                int element_count, bool tuning);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
