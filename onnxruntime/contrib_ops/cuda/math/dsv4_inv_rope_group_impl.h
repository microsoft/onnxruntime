// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct DSV4InvRopeGroupParams {
  int num_tokens;
  int num_heads;
  int head_dim;
  int rope_head_dim;
  int nope_dim;
  int num_groups;
  int group_dim;
};

template <typename T>
Status LaunchDSV4InvRopeGroup(cudaStream_t stream, const DSV4InvRopeGroupParams& params,
                              const T* input, const float* cos_table, const float* sin_table,
                              T* output);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
