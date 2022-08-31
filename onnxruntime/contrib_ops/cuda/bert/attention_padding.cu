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

__global__ void getTrtPaddingOffsetKernel(int* trt_mha_padding_offset,
                                          const int* sequence_length,
                                          const int request_batch_size,
                                          const int request_sequence_length) {
  extern __shared__ int tmp_offset[];
  if (threadIdx.x == 0) {
    tmp_offset[0] = 0;
    for (int i = 0; i < request_batch_size; i++) {
      tmp_offset[i * 2 + 1] = tmp_offset[i * 2] + sequence_length[i];
      tmp_offset[i * 2 + 2] = request_sequence_length * (i + 1);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 2 * request_batch_size + 1; i += blockDim.x) {
    trt_mha_padding_offset[i] = tmp_offset[i];
  }
}

// Get TensorRT fused mha padding offset when we keep the padding
void LaunchTrtPaddingOffset(int* trt_mha_padding_offset,
                            const int* sequence_length,
                            const int request_batch_size,
                            const int request_sequence_length,
                            cudaStream_t stream) {
  getTrtPaddingOffsetKernel<<<1, 256, sizeof(int) * (2 * request_batch_size + 1), stream>>>(
      trt_mha_padding_offset, sequence_length, request_batch_size, request_sequence_length);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
