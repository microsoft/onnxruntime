// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "core/framework/op_kernel.h"

namespace onnxruntime::contrib {

struct SharedQMoEAttributes {
  std::string quant_type = "int";
  int64_t weight_bits = 4;
  int64_t block_size = 32;
  int64_t top_k = 1;
  bool normalize_routing_weights = false;
  bool track_experts = false;
  // Applies only to INT3/INT4; NVFP4/MXFP4 retain their existing scale/weight rounding.
  bool deferred_scale = false;
  float activation_alpha = 1.0f;
  float activation_beta = 0.0f;
  float swiglu_limit = std::numeric_limits<float>::infinity();
};

// Adapts ordinary CPU QMoE routing to the same immutable shared-layout kernels
// used by offload workers. Source tensors remain owned by the kernel context
// without online weight prepacking or unpacking.
Status ComputeSharedQMoE(OpKernelContext* context, const SharedQMoEAttributes& attributes);

}  // namespace onnxruntime::contrib
