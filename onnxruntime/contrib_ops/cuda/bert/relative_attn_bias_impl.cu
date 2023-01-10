/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/relative_attn_bias_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template<typename T>
__global__ void buildRelativeAttentionBias(T* relative_attention_bias,
                                           const T* relative_attention_bias_table,
                                           const int head_num,
                                           const int seq_len,
                                           const int num_bucket,
                                           const bool is_bidirectional,
                                           const int max_distance) {
  const int head_id = blockIdx.x;
  for (int seq_id = threadIdx.x; seq_id < seq_len * seq_len; seq_id += blockDim.x) {
    int row_id = seq_id / seq_len;
    int col_id = seq_id % seq_len;

    int relative_position = col_id - row_id;

    int relative_buckets = 0;
    int tmp_num_bucket = num_bucket;

    if (is_bidirectional) {
        tmp_num_bucket /= 2;
        if (relative_position > 0) {
            relative_buckets += tmp_num_bucket;
        } else {
            relative_position *= -1;
        }
    } else {
        if (relative_position > 0) {
            relative_position = 0;
        } else {
            relative_position *= -1;
        }
    }

    int max_exact = tmp_num_bucket / 2;
    bool is_small  = relative_position < max_exact;

    int relative_position_if_large =
        max_exact
        + (int)(logf(relative_position * 1.0f / max_exact) / logf((float)max_distance / max_exact)
                * (tmp_num_bucket - max_exact));

    relative_position_if_large = min(relative_position_if_large, tmp_num_bucket - 1);

    relative_buckets += is_small ? relative_position : relative_position_if_large;

    relative_attention_bias[head_id * seq_len * seq_len + seq_id] =
        relative_attention_bias_table[head_id * num_bucket + relative_buckets];
    }
}

template <typename T>
Status LaunchRelPosAttnBiasKernel(
  cudaStream_t stream,
  T* output,
  const T* bias_table,
  const int num_heads,
  const int seq_len,
  const int num_bucket,
  const int max_distance,
  const bool is_bidirectional)
{
  dim3 grid(num_heads);
  dim3 block(256);

  buildRelativeAttentionBias<<<grid, block, 0, stream>>>(output,
                                                         bias_table,
                                                         num_heads,
                                                         seq_len,
                                                         num_bucket,
                                                         is_bidirectional,
                                                         max_distance);

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchRelPosAttnBiasKernel<float>(cudaStream_t stream,
                                                  float* output,
                                                  const float* bias_table,
                                                  const int num_heads,
                                                  const int seq_len,
                                                  const int num_bucket,
                                                  const int max_distance,
                                                  const bool is_bidirectional);

template Status LaunchRelPosAttnBiasKernel<half>(cudaStream_t stream,
                                                 half* output,
                                                 const half* bias_table,
                                                 const int num_heads,
                                                 const int seq_len,
                                                 const int num_bucket,
                                                 const int max_distance,
                                                 const bool is_bidirectional);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
