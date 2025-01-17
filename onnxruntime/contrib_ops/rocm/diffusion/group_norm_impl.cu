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
    Stream* ort_stream,
    T* output,
    T* add_out,
    const T* input,
    const T* skip,
    const T* bias,
    const float* gamma,
    const float* beta,
    void* workspace,
    float epsilon,
    int batch_size,
    int num_channels,
    int height,
    int width,
    int num_groups,
    bool use_silu,
    bool broadcast_skip,
    int channels_per_block) {
  GroupNormNHWCTunableParams<T> params(tuning_ctx, ort_stream, output, add_out, input, skip, bias, gamma, beta,
                                       reinterpret_cast<float*>(workspace), epsilon, batch_size, num_channels,
                                       height, width, num_groups, use_silu, broadcast_skip, channels_per_block);

  if (params.channels_per_block % params.channels_per_group != 0 ||
      params.channels_per_block > kMaxSize ||
      (params.channels_per_group % CHANNELS_PER_THREAD != 0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "GroupNorm in ROCM does not support the input: n=", batch_size,
                           " h=", height,
                           " w=", width,
                           " c=", num_channels,
                           " groups=", num_groups);
  }

  HIP_RETURN_IF_ERROR(hipMemsetAsync(
      params.group_sum_buffer, 0, GetGroupNormWorkspaceSizeInBytes(batch_size, num_groups), params.StreamHandle()));

  if (tuning_ctx->IsTunableOpEnabled()) {
    static GroupNormNHWCTunableOp<T> op;
    return op(&params);
  }

  return GroupNormNHWCStaticSelection(&params);
}

template Status LaunchGroupNormKernel<half>(RocmTuningContext* tuning_ctx, Stream* stream, half* output,
                                            half* add_out, const half* input, const half* skip, const half* bias,
                                            const float* gamma, const float* beta, void* workspace, float epsilon,
                                            int batch_size, int num_channels, int height, int width, int num_groups,
                                            bool use_silu, bool broadcast_skip, int channels_per_block);

template Status LaunchGroupNormKernel<float>(RocmTuningContext* tuning_ctx, Stream* stream, float* output,
                                             float* add_out, const float* input, const float* skip, const float* bias,
                                             const float* gamma, const float* beta, void* workspace, float epsilon,
                                             int batch_size, int num_channels, int height, int width, int num_groups,
                                             bool use_silu, bool broadcast_skip, int channels_per_block);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
