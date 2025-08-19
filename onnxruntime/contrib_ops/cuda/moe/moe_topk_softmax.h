/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {
/*
Launches the topk gating softmax required for the MoE layers.

Params:
input - a [num_rows x num_experts]
finished - [num_rows] vector with 1 if the sentence at this row is done translating and 0 otherwise.
output - a buffer of shape [num_rows x k] containing the top-k values of the softmax for each row.
indices - a matrix of shape [num_rows x k] containing the top-k experts each row should get routed to.
source_rows - a matrix of shape [num_rows x k] used internally for permuting. source_rows[row][k] =  k * num_rows +
row. It is constructed like this so we can track where each of the original rows end up in order to perform the
"k-way" reduction later in the routing.

num_rows - The number of rows in the matrix
num_experts - The number of expert layers present
k - k value in topk
*/
template <typename T>
void topk_gating_softmax_kernelLauncher(const T* input, const bool* finished, float* output, T* softmax_temp_out,
                                        int* indices, int* source_row, int num_rows, int num_experts, int k,
                                        bool normalize_routing_weights, bool use_sparse_mixer, cudaStream_t stream);
}  // namespace onnxruntime::contrib::cuda