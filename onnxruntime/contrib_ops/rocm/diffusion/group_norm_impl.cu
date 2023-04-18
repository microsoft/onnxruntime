// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The ROCM kernel is hipified from CUDA kernel.
#include "contrib_ops/rocm/diffusion/group_norm_impl.h"

#include <hip/hip_fp16.h>
#include "contrib_ops/rocm/diffusion/group_norm_common.h"
#include "contrib_ops/rocm/diffusion/group_norm_tunable_op.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
Status LaunchGroupNormKernel(
    RocmTuningContext* tuning_ctx,
    hipStream_t stream,
    T* output,
    const T* input,
    const float* gamma,
    const float* beta,
    void* workspace,
    float epsilon,
    int batch_size,
    int num_channels,
    int height,
    int width,
    int num_groups,
    bool use_swish_activation) {
  if (batch_size > static_cast<int>(kMaxGroupNormBatchSize)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED,
                           "only support batch_size <= 32. Got", batch_size);
  }

  if (num_groups != static_cast<int>(kGroupNormNumberOfGroups)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED,
                           "only num_groups=32 is supported. Got", num_groups);
  }

  GroupNormNHWCParams<T> params(tuning_ctx, stream, output, reinterpret_cast<float*>(workspace), input, gamma, beta,
                                batch_size, height, width, num_channels, num_groups, epsilon, use_swish_activation);

  if (tuning_ctx->IsTunableOpEnabled()) {
    static GroupNormNHWCTunableOp<T> op;
    return op(&params);
  }

  return GroupNormNHWCStaticSelection(&params);
}

template Status LaunchGroupNormKernel<half>(RocmTuningContext* tuning_ctx, hipStream_t stream, half* output,
                                            const half* input, const float* gamma, const float* beta, void* workspace,
                                            float epsilon, int batch_size, int num_channels,
                                            int height, int width, int num_groups, bool swish);

template Status LaunchGroupNormKernel<float>(RocmTuningContext* tuning_ctx, hipStream_t stream, float* output,
                                             const float* input, const float* gamma, const float* beta, void* workspace,
                                             float epsilon, int batch_size, int num_channels,
                                             int height, int width, int num_groups, bool swish);
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
