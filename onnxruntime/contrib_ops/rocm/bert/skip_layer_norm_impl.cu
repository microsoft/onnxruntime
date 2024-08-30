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

template <typename T, typename U, typename V, bool Simplified>
Status LaunchSkipLayerNormKernel(
    RocmTuningContext* tuning_ctx, Stream* stream, V* output, T* skip_input_bias_add_output, const T* input,
    const T* skip, const V* gamma, const V* beta, const T* bias, float epsilon, int ld, int element_count) {
  // this must be true because element_count is the total size of the tensor
  assert(element_count % ld == 0);

  SkipLayerNormParams<T, V> params(tuning_ctx, stream, output, skip_input_bias_add_output, input, skip,
                                   gamma, beta, bias, epsilon, ld, element_count);

  if (tuning_ctx->IsTunableOpEnabled()) {
    static SkipLayerNormTunableOp<T, U, V, Simplified> op;
    return op(&params);
  }

  return SkipLayerNormStaticSelection<T, U, V, Simplified>(&params);
}

template Status LaunchSkipLayerNormKernel<float, float, float, true>(
    RocmTuningContext* tuning_ctx, Stream* stream, float* output, float* skip_input_bias_add_output, const float* input,
    const float* skip, const float* gamma, const float* beta,
    const float* bias, float epsilon, int ld,
    int element_count);

template Status LaunchSkipLayerNormKernel<half, float, half, true>(
    RocmTuningContext* tuning_ctx, Stream* stream, half* output, half* skip_input_bias_add_output, const half* input,
    const half* skip, const half* gamma, const half* beta,
    const half* bias, float epsilon, int ld,
    int element_count);

template Status LaunchSkipLayerNormKernel<float, float, float, false>(
    RocmTuningContext* tuning_ctx, Stream* stream, float* output, float* skip_input_bias_add_output, const float* input,
    const float* skip, const float* gamma, const float* beta,
    const float* bias, float epsilon, int ld,
    int element_count);

template Status LaunchSkipLayerNormKernel<half, float, half, false>(
    RocmTuningContext* tuning_ctx, Stream* stream, half* output, half* skip_input_bias_add_output, const half* input,
    const half* skip, const half* gamma, const half* beta,
    const half* bias, float epsilon, int ld,
    int element_count);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
