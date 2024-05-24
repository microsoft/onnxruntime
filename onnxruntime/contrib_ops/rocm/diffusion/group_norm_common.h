// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "contrib_ops/rocm/diffusion/group_norm_common_base.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct GroupNormNHWCTunableParams : OpParams, GroupNormNHWCParams<T> {
  GroupNormNHWCTunableParams(RocmTuningContext* tuning_ctx,
                             onnxruntime::Stream* ort_stream,
                             T* output,
                             T* add_out,
                             const T* input,
                             const T* skip,
                             const T* bias,
                             const float* gamma,
                             const float* beta,
                             float* workspace,
                             float epsilon,
                             int batch_size,
                             int num_channels,
                             int height,
                             int width,
                             int num_groups,
                             bool use_silu,
                             bool broadcast_skip,
                             int channels_per_block)
      : OpParams(tuning_ctx, ort_stream),
        GroupNormNHWCParams<T>(output, add_out, input, skip, bias, gamma, beta, workspace, epsilon, batch_size,
                               num_channels, height, width, num_groups, use_silu, broadcast_skip, channels_per_block) {}

  std::string Signature() const override {
    std::string silu_suffix = this->use_silu ? "_silu" : "_pass";
    std::string skip_suffix = this->skip != nullptr ? "_skip" : "_noskip";
    std::string broadcast_suffix = this->broadcast_skip ? "_broadcast" : "_nobroadcast";
    std::string bias_suffix = this->bias != nullptr ? "_bias" : "_nobias";
    std::string sig = std::to_string(this->n) + "_" + std::to_string(this->h * this->w) + "_" +
                      std::to_string(this->c) + "_" + std::to_string(this->groups) + silu_suffix +
                      skip_suffix + broadcast_suffix + bias_suffix;
    return sig;
  }
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
