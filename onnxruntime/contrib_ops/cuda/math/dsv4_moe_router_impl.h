// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct DSV4MoERouterParams {
  int num_tokens;
  int num_experts;
  int topk;
  int local_expert_start;
  int local_expert_count;
  float route_scale;
};

template <typename T>
Status LaunchDSV4MoERouter(cudaStream_t stream, const DSV4MoERouterParams& params,
                           const float* scores, const float* bias, const int64_t* expert_ids,
                           T* router_probs, float* weight_scale);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
