/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Modifications Copyright (c) Microsoft. */

// The code below is mostly copied from Pytorch PersistentSoftmax.cuh
#include <stdio.h>
#include "hip/hip_runtime.h"

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/math/softmax.h"
#include "core/providers/rocm/math/softmax_common.h"
#include "core/providers/rocm/math/softmax_tunable_op.cuh"

#include <limits>

namespace onnxruntime {
namespace rocm {

template <typename InputT, typename OutputT, typename AccT, bool IsLogSoftmax>
Status dispatch_warpwise_softmax_forward(hipStream_t stream, OutputT* dst, const InputT* src, int softmax_elements,
                                         int softmax_elements_stride, int batch_count, RocmTuningContext* tuning_ctx) {
  SoftmaxParams<InputT, OutputT> params(tuning_ctx, stream, dst, src, softmax_elements, softmax_elements_stride,
                                        softmax_elements_stride, batch_count, IsLogSoftmax);
  if (tuning_ctx != nullptr && tuning_ctx->IsTunableOpEnabled()) {
    static SoftmaxTunableOp<InputT, OutputT, AccT> op;
    return op(&params);
  }
  return SoftmaxWarpwiseStaticSelection<InputT, OutputT, AccT>(&params);
}

#define SPECIALIZED_SOFTMAX_IMPL(InputT, OutputT, AccT)                             \
  template Status dispatch_warpwise_softmax_forward<InputT, OutputT, AccT, false>(  \
      hipStream_t stream, OutputT * dst, const InputT* src, int softmax_elements,   \
      int softmax_elements_stride, int batch_count, RocmTuningContext* tuning_ctx); \
  template Status dispatch_warpwise_softmax_forward<InputT, OutputT, AccT, true>(   \
      hipStream_t stream, OutputT * dst, const InputT* src, int softmax_elements,   \
      int softmax_elements_stride, int batch_count, RocmTuningContext* tuning_ctx);

SPECIALIZED_SOFTMAX_IMPL(float, float, float)
SPECIALIZED_SOFTMAX_IMPL(half, half, float)
SPECIALIZED_SOFTMAX_IMPL(half, float, float)
SPECIALIZED_SOFTMAX_IMPL(double, double, double)
SPECIALIZED_SOFTMAX_IMPL(BFloat16, BFloat16, float)

template <typename InputT, typename OutputT, typename AccT, bool IsLogSoftmax>
Status dispatch_blockwise_softmax_forward(hipStream_t stream, OutputT* output,
                                          const InputT* input, int softmax_elements,
                                          int input_stride, int output_stride,
                                          int batch_count, RocmTuningContext* tuning_ctx) {
  SoftmaxParams<InputT, OutputT> params(tuning_ctx, stream, output, input, softmax_elements, input_stride,
                                        output_stride, batch_count, IsLogSoftmax);
  if (tuning_ctx != nullptr && tuning_ctx->IsTunableOpEnabled()) {
    static SoftmaxTunableOp<InputT, OutputT, AccT> op;
    return op(&params);
  }
  return SoftmaxBlockwiseStaticSelection<InputT, OutputT, AccT>(&params);
}

#define SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(InputT, OutputT, AccT)                           \
  template Status dispatch_blockwise_softmax_forward<InputT, OutputT, AccT, false>(         \
      hipStream_t stream, OutputT * output, const InputT* input, int softmax_elements,      \
      int input_stride, int output_stride, int batch_count, RocmTuningContext* tuning_ctx); \
  template Status dispatch_blockwise_softmax_forward<InputT, OutputT, AccT, true>(          \
      hipStream_t stream, OutputT * output, const InputT* input, int softmax_elements,      \
      int input_stride, int output_stride, int batch_count, RocmTuningContext* tuning_ctx);

SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(float, float, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(half, half, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(half, float, float)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(double, double, double)
SPECIALIZED_BLOCKWISE_SOFTMAX_IMPL(BFloat16, BFloat16, float)
}  // namespace rocm
}  // namespace onnxruntime
