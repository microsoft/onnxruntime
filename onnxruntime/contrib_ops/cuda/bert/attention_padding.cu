/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// TrtPaddingOffset kernel are modified from FasterTransformer

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {
__global__ void getTrtSequenceOffset(int* trt_mha_padding_offset,
                                     const int* mask_index,
                                     const int batch_size) {
  extern __shared__ int tmp_offset[];
  if (threadIdx.x == 0) {
    tmp_offset[0] = 0;
    for (int i = 0; i < batch_size; i++) {
      tmp_offset[i + 1] = tmp_offset[i] + mask_index[i];
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
    trt_mha_padding_offset[i] = tmp_offset[i];
  }
}

// Get sequence offset for TensorRT fused attention when there is no padding (or padding is removed)
// For example, when sequence length is
void LaunchTrtSequenceOffset(int* trt_mha_padding_offset,
                             const int* mask_index,
                             const int batch_size,
                             cudaStream_t stream) {
  getTrtSequenceOffset<<<1, 256, sizeof(int) * (batch_size + 1), stream>>>(
      trt_mha_padding_offset, mask_index, batch_size);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
