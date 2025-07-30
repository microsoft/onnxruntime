// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct GatherBlockQuantizedParam {
  cudaStream_t stream;
  int64_t after_gather_dim;
  int64_t ind_dim;
  int64_t bits;
  int64_t block_size;
  int64_t gather_axis;
  int64_t N;
};

template <typename T1, typename T2, typename Tind>
void LaunchGatherBlockQuantizedKernel(const T1* data,
                                      const Tind* indices,
                                      const T2* scales,
                                      const T1* zero_points,
                                      T2* output,
                                      GatherBlockQuantizedParam param);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
