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

#pragma once

#include "moe_gemm_kernels.h"
#include <cuda_runtime_api.h>

#include "core/common/common.h"

using namespace onnxruntime;

namespace ort_fastertransformer {

static inline size_t pad_to_multiple_of_16(const size_t& input) {
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

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
void topk_gating_softmax_kernelLauncher(const T* input,
                                        const bool* finished,
                                        T* output,
                                        T* softmax_temp_out,
                                        int* indices,
                                        int* source_row,
                                        const int num_rows,
                                        const int num_experts,
                                        const int k,
                                        cudaStream_t stream);

class CubKeyValueSorter {
 public:
  CubKeyValueSorter();

  CubKeyValueSorter(const int num_experts);

  void update_num_experts(const int num_experts);

  size_t getWorkspaceSize(const size_t num_key_value_pairs);

  void run(void* workspace,
           const size_t workspace_size,
           const int* keys_in,
           int* keys_out,
           const int* values_in,
           int* values_out,
           const size_t num_key_value_pairs,
           cudaStream_t stream);

 private:
  size_t num_key_value_pairs_;
  int num_experts_;
  int num_bits_;
};

template <typename T>
void initialize_moe_routing_kernelLauncher(const T* unpermuted_input,
                                           T* permuted_output,
                                           const int* expanded_dest_row_to_expanded_source_row,
                                           int* expanded_source_row_to_expanded_dest_row,
                                           const int num_rows,
                                           const int active_rows,
                                           const int cols,
                                           const int k,
                                           cudaStream_t stream);

template <typename T>
void finalize_moe_routing_kernelLauncher(const T* expanded_permuted_rows,
                                         T* reduced_unpermuted_output,
                                         const T* bias,
                                         const T* scales,
                                         const int* expanded_source_row_to_expanded_dest_row,
                                         const int* expert_for_source_row,
                                         const int num_rows,
                                         const int cols,
                                         const int k,
                                         cudaStream_t stream);

template <typename T>
void finalize_moe_routing_kernelLauncher(const T* expanded_permuted_rows,
                                         T* reduced_unpermuted_output,
                                         const T* skip,
                                         const T* bias,
                                         const T* scales,
                                         const int* expanded_source_row_to_expanded_dest_row,
                                         const int* expert_for_source_row,
                                         const int num_rows,
                                         const int cols,
                                         const int k,
                                         cudaStream_t stream);

template <typename T>
void finalize_moe_routing_kernelLauncher(const T* expanded_permuted_rows,
                                         T* reduced_unpermuted_output,
                                         const T* skip_1,
                                         const T* skip_2,
                                         const T* bias,
                                         const T* scales,
                                         const int* expanded_source_row_to_expanded_dest_row,
                                         const int* expert_for_source_row,
                                         const int num_rows,
                                         const int cols,
                                         const int k,
                                         cudaStream_t stream);

// Assumes inputs activations are row major. Weights need to be preprocessed by th_op/weight_quantize.cc .
// Nested in a class to avoid multiple calls to cudaGetDeviceProperties as this call can be expensive.
// Avoid making several duplicates of this class.
template <typename T,          /*The type used for activations/scales/compute*/
          typename WeightType, /* The type for the MoE weights */
          typename Enable = void>
class CutlassMoeFCRunner {
 public:
  CutlassMoeFCRunner(int sm_version);

  size_t getWorkspaceSize(
      const int num_rows, const int hidden_size, const int inter_size, const int num_experts, const int k);

  void run_moe_fc(const T* input_activations,
                  const T* gating_output,
                  const WeightType* fc1_expert_weights,
                  const T* fc1_scales,
                  const T* fc1_expert_biases,
                  ActivationType fc1_activation_type,
                  const WeightType* fc2_expert_weights,
                  const T* fc2_scales,
                  const int num_rows,
                  const int hidden_size,
                  const int inter_size,
                  const int num_experts,
                  const int k,
                  char* workspace_ptr,
                  T* fc2_result,
                  T* expert_scales,
                  int* expanded_source_row_to_expanded_dest_row,
                  int* expert_for_source_row,
                  cudaStream_t stream);

  void run_moe_fc(const T* input_activations,
                  const T* gating_output,
                  const WeightType* fc1_expert_weights,
                  const T* fc1_scales,
                  const T* fc1_expert_biases,
                  ActivationType fc1_activation_type,
                  const WeightType* fc2_expert_weights,
                  const T* fc2_scales,
                  const int num_rows,
                  const int hidden_size,
                  const int inter_size,
                  const int num_experts,
                  const int k,
                  char* workspace_ptr,
                  T* fc2_result,
                  const bool* finished,
                  const int active_rows,
                  T* expert_scales,
                  int* expanded_source_row_to_expanded_dest_row,
                  int* expert_for_source_row,
                  cudaStream_t stream);

  void compute_total_rows_before_expert(const int* sorted_indices,
                                        const int total_indices,
                                        const int num_experts,
                                        int64_t* total_rows_before_expert,
                                        cudaStream_t stream);

 private:
  void configure_ws_ptrs(char* ws_ptr,
                         const int num_rows,
                         const int hidden_size,
                         const int inter_size,
                         const int num_experts,
                         const int k);

 private:
  CubKeyValueSorter sorter_;
  MoeGemmRunner<T, WeightType> moe_gemm_runner_;

  // Pointers
  int* source_rows_;
  int* permuted_rows_;
  int* permuted_experts_;
  char* sorter_ws_;
  T* permuted_data_;
  T* softmax_out_;

  int64_t* total_rows_before_expert_;

  T* fc1_result_;
};

template <typename WeightType>
class CutlassMoeFCRunner<float, WeightType, typename std::enable_if_t<!std::is_same<float, WeightType>::value>> {
 public:
  CutlassMoeFCRunner(int sm_version);

  size_t getWorkspaceSize(
      const int num_rows, const int hidden_size, const int inter_size, const int num_experts, const int k) {
    return 0;
  }
};

}  // namespace ort_fastertransformer