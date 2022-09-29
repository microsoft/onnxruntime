// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/multi_tensor/common.cuh"

namespace onnxruntime {
namespace cuda {

#define MTA_ADAMW_GROUP_SIZE 4
#define MTA_ADAMW_CHUNK_SIZE 2048 * 32

template <typename T_WEIGHT, typename T_GRAD, typename T_MOMENTUM>
struct AdamWMTAFunctor {
  void operator()(cudaStream_t stream,
                  ChunkGroup<MTA_ADAMW_GROUP_SIZE> chunks,
                  const float alpha,
                  const float beta,
                  const float epsilon,
                  const float lr,
                  const float decay,
                  const int64_t adam_mode,
                  const int64_t correct_bias,
                  const int64_t update_count);
};

}  // namespace cuda
}  // namespace onnxruntime
