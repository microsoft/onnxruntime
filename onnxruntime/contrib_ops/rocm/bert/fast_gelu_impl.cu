/*
 The implementation of this file is based on gelu plugin in TensorRT demo:
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

// Modifications: Add (bias) before Gelu is merged into this op to get better performance.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Add FastGeluKernelVec to leverage vectorized load/write
//                and modify FastGeluKernel to get better performance.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/shared_inc/rocm_call.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl_kernel.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl.h"
#include "contrib_ops/rocm/bert/fast_gelu_tunable_op.h"

using namespace onnxruntime::rocm;

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
Status LaunchFastGeluKernel(hipStream_t stream, int input_length, int bias_length,
                          const T* input, const T* bias, T* output, bool tuning) {
  static FastGeluTunableOp<T> op;
  if (tuning) {
    op.EnableTuning();
  }
  FastGeluParams<T> op_params(stream, input, bias, output, input_length, bias_length);
  return op(&op_params);
}

template Status LaunchFastGeluKernel<float>(hipStream_t stream, int input_length, int bias_length,
                                          const float* input, const float* bias, float* output, bool tuning);

template Status LaunchFastGeluKernel<BFloat16>(hipStream_t stream, int input_length, int bias_length,
                                             const BFloat16* input, const BFloat16* bias, BFloat16* output, bool tuning);

template Status LaunchFastGeluKernel<half>(hipStream_t stream, int input_length, int bias_length,
                                         const half* input, const half* bias, half* output, bool tuning);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
