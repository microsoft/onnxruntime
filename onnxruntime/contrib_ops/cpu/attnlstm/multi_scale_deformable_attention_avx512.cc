// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/attnlstm/multi_scale_deformable_attention.h"

#include <cstring>

namespace onnxruntime {
namespace contrib {
  void MultiScaleDeformableAttention::ComputeInternal(
    const OpKernelContext* context,
    const float* value,
    const int64_t* value_spatial_shapes,
    const float* reference_points,
    const float* sampling_locations,
    const float* attention_weights,
    float* output,
    int64_t S,
    int64_t M,
    int64_t L,
    int64_t P,
    int64_t D,
    int64_t Q,
    concurrency::ThreadPool& thread_pool,
    AllocatorPtr alloc) const {
    memset(output, 0, sizeof(float) * Q * M * D);
  }
}  // namespace contrib
}  // namespace onnxruntime
