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
#include <hip/hip_fp16.h>
#include "hip/hip_runtime.h"

#include "contrib_ops/rocm/bert/skip_layer_norm_impl.h"
#include "contrib_ops/rocm/bert/skip_layer_norm_impl_kernel.h"
#include "contrib_ops/rocm/bert/skip_layer_norm_tunable_op.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
bool LaunchSkipLayerNormKernel(
    hipStream_t stream, T* output, const T* input, const T* skip, const T* gamma,
    const T* beta, const T* bias, const float epsilon, const int ld, const int element_count, const bool tuning) {
  // this must be true because element_count is the total size of the tensor
  assert(element_count % ld == 0);

  static SkipLayerNormTunableOp<T> op;
  if (tuning) {
    op.EnableTuning();
  }

  SkipLayerNormParams<T> op_params(stream, output, input, skip, gamma, beta, bias, epsilon, ld, element_count);
  return op(&op_params).IsOK();
}

template bool LaunchSkipLayerNormKernel<float>(hipStream_t stream, float* output, const float* input,
                                               const float* skip, const float* gamma, const float* beta,
                                               const float* bias, const float epsilon, const int ld,
                                               const int element_count, const bool tuning);

template bool LaunchSkipLayerNormKernel<half>(hipStream_t stream, half* output, const half* input,
                                              const half* skip, const half* gamma, const half* beta,
                                              const half* bias, const float epsilon, const int ld,
                                              const int element_count, const bool tuning);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
