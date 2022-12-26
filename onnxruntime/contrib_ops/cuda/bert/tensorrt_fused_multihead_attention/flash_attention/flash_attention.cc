/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/flash_attention/flash_attention.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool has_flash_attention_kernel(int sm, int head_size) {
  return (sm == 75 || sm == 80 || sm == 86 || sm == 89) &&
         (head_size == 16 || head_size == 32 || head_size == 40 ||
          head_size == 64 || head_size == 80 || head_size == 128 ||
          head_size == 160 || head_size == 256);
}

// void run_flash_attention_kernel(const void* input, const void* cu_seqlens, void* output,
//                                 FusedMultiHeadFlashAttentionKernel const* kernels,
//                                 int32_t batch_size,
//                                 int32_t num_heads,
//                                 int32_t head_size,
//                                 int32_t sequence_length,
//                                 cudaStream_t stream)
// {
//     Fused_multihead_attention_params_v2 params = getMHFAParams(
//         batch_size, sequence_length, num_heads, head_size,
//         input, const_cast<void*>(cu_seqlens), output);
//     kernels->run(params, stream);
// }

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
