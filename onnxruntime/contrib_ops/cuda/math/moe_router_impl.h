// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct MoERouterParams {
  int num_tokens;
  int num_experts;
  int topk;
  int local_expert_start;
  int local_expert_count;
  float route_scale;
};

template <typename T>
Status LaunchMoERouter(cudaStream_t stream, const MoERouterParams& params,
                       const float* scores, const float* bias, const int64_t* expert_ids,
                       T* router_probs, float* weight_scale);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
