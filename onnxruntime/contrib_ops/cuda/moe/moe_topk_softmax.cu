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

#include "contrib_ops/cuda/moe/moe_topk_softmax.h"

#include <cfloat>

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>

#include "contrib_ops/cuda/bert/transformer_cuda_common.h"
#include "core/common/common.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"

namespace onnxruntime::contrib::cuda {
static constexpr int WARP_SIZE = 32;

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template <typename T, int TPB>
__launch_bounds__(TPB) __global__
    void moe_softmax(const T* input, const bool* finished, T* output, const int num_cols) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float normalizing_factor;
  __shared__ float float_max;

  const int thread_row_offset = blockIdx.x * num_cols;

  float threadData(-FLT_MAX);

  // Don't touch finished rows.
  if ((finished != nullptr) && finished[blockIdx.x]) {
    return;
  }

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData = max(static_cast<float>(input[idx]), threadData);
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12090
  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, ::cuda::maximum());
#else
  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
#endif

  if (threadIdx.x == 0) {
    float_max = maxElem;
  }
  __syncthreads();

  threadData = 0;

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData += exp((static_cast<float>(input[idx]) - float_max));
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12090
  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, ::cuda::std::plus());
#else
  // Deprecated on CUDA 12.9
  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, cub::Sum());
#endif

  if (threadIdx.x == 0) {
    normalizing_factor = 1.f / Z;
  }
  __syncthreads();

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
    output[idx] = T(val);
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_top_k(const T*, const bool*, float*, int*, int*, int, int, bool) {
  // Does not support pre-Kepler architectures
  ;
}
#else
template <typename T, int TPB>
__launch_bounds__(TPB) __global__
    void moe_top_k(const T* inputs_after_softmax, const bool* finished, float* output, int* indices, int* source_rows,
                   int num_experts, int k, bool normalize_routing_weights) {
  using cub_kvp = cub::KeyValuePair<int, T>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  int num_rows = gridDim.x;
  const int block_row = blockIdx.x;

  const bool should_process_row = finished ? !finished[block_row] : true;
  const int thread_row_offset = blockIdx.x * num_experts;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-1.f);

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_row_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_softmax[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[k * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int idx = k * block_row + k_idx;
      output[idx] = static_cast<float>(result_kvp.value);
      indices[idx] = should_process_row ? result_kvp.key : num_experts;
      source_rows[idx] = k_idx * num_rows + block_row;

      if (normalize_routing_weights && k_idx == k - 1) {
        float current_sum = 0.f;
#pragma unroll
        for (int ki = 0; ki < k; ++ki) {
          current_sum += output[idx - ki];
        }

        if (current_sum > 0.f) {
          const float inv_sum = 1.f / current_sum;
#pragma unroll
          for (int ki = 0; ki < k; ++ki) {
            output[idx - ki] = output[idx - ki] * inv_sum;
          }
        }
      }
    }
    __syncthreads();
  }
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
template <typename T, int TPB, int NUM_EXPERTS>
__launch_bounds__(TPB) __global__ void sparse_mixer_top2(const T*, float*, int*, int*, const float) {
  // Does not support pre-Kepler architectures
  ;
}
#else

template <typename T, int TPB, int NUM_EXPERTS>
__launch_bounds__(TPB) __global__
    void sparse_mixer_top2(const T* inputs, float* output, int* indices, int* source_rows, const float jitter_eps) {
  static constexpr int K = 2;

  using cub_kvp = cub::KeyValuePair<int, T>;
  using KVBlockReduce = cub::BlockReduce<cub_kvp, TPB>;

  __shared__ float result_kvp_value[K];
  __shared__ typename KVBlockReduce::TempStorage kvTmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  int num_rows = gridDim.x;
  const int block_row = blockIdx.x;

  const int thread_row_offset = blockIdx.x * NUM_EXPERTS;

  float factor[K];
  bool logits_mask[K];

#pragma unroll
  for (int k_idx = 0; k_idx < K; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-1.f);

    cub_kvp inp_kvp;

#pragma unroll
    for (int expert = threadIdx.x; expert < NUM_EXPERTS; expert += TPB) {
      const int idx = thread_row_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[K * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp = KVBlockReduce(kvTmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int idx = K * block_row + k_idx;
      result_kvp_value[k_idx] = (float)result_kvp.value;
      indices[idx] = result_kvp.key;
      source_rows[idx] = k_idx * num_rows + block_row;
    }
    __syncthreads();

#pragma unroll
    for (int expert = threadIdx.x; expert < NUM_EXPERTS; expert += TPB) {
      const int idx = thread_row_offset + expert;
      factor[k_idx] = max(abs((float)inputs[idx]), result_kvp_value[k_idx]);
      logits_mask[k_idx] = (result_kvp_value[k_idx] - (float)inputs[idx]) > (2 * jitter_eps * factor[k_idx]);
      if (k_idx == 1 && expert == indices[K * block_row]) {
        logits_mask[1] = true;
      }
    }
  }

#pragma unroll
  for (int k_idx = 0; k_idx < K; ++k_idx) {
    float row_sum(0);

#pragma unroll
    for (int ii = threadIdx.x; ii < NUM_EXPERTS; ii += TPB) {
      const int idx = thread_row_offset + ii;
      row_sum += logits_mask[k_idx] ? 0 : exp((static_cast<float>(inputs[idx]) - result_kvp_value[k_idx]));
    }

#pragma unroll
    for (int mask = NUM_EXPERTS / 2; mask > 0; mask /= 2) {
      row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, NUM_EXPERTS);
    }

    const float normalizing_factor = 1.f / row_sum;

    const int idx = K * block_row + k_idx;
    if (threadIdx.x == indices[idx]) {
      const int input_idx = thread_row_offset + threadIdx.x;
      output[idx] = logits_mask[k_idx] ? 0
                                       : exp((static_cast<float>(inputs[input_idx]) - result_kvp_value[k_idx])) *
                                             normalizing_factor;
    }
  }
}
#endif

// ====================== TopK softmax things ===============================

/*
A Top-K gating softmax written to exploit when the number of experts in the MoE layers
are a small power of 2. This allows us to cleanly share the rows among the threads in
a single warp and eliminate communication between warps (so no need to use shared mem).

It fuses the softmax, max and argmax into a single kernel.

Limitations:
  1) This implementation is intended for when the number of experts is a small power of 2.
  2) This implementation assumes k is small, but will work for any k.
*/

template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk_gating_softmax(const T* input, const bool* finished, float* output, int num_rows, int* indices,
                             int* source_rows, int k, bool normalize_routing_weights) {
  // We begin by enforcing compile time assertions and setting up compile time constants.
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  // Restrictions based on previous section.
  static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

  // We have NUM_EXPERTS elements per row. We specialize for small #experts
  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  // Restrictions for previous section.
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

  // ===================== From this point, we finally start computing run-time variables. ========================

  // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
  // This, each block processes a chunk of rows. We start by computing the start row for each block.
  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

  // Now, using the base row per thread block, we compute the base row per warp.
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

  // The threads in a warp are split into sub-groups that will work on a row.
  // We compute row offset for each thread sub-group
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;

  // Threads with indices out of bounds should early exit here.
  if (thread_row >= num_rows)
    return;
  const bool should_process_row = finished ? !finished[thread_row] : true;

  // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
  // row it will read.
  const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

  // Now, we compute the group each thread belong to in order to determine the first column to start loads.
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
  // this can support all powers of 2 up to 16.
  using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;

  // Finally, we pull in the data from global mem
  cutlass::Array<T, VPT> row_chunk_input;
  AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk_input);
  const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  using ComputeType = float;
  using Converter = cutlass::NumericArrayConverter<ComputeType, T, VPT>;
  Converter compute_type_converter;
  cutlass::Array<ComputeType, VPT> row_chunk = compute_type_converter(row_chunk_input);

  // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
  // convert to float afterwards for the exp + sum reduction.
  ComputeType thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, row_chunk[ii]);
  }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
  }

  // From this point, thread max in all the threads have the max within the row.
  // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
  float row_sum = 0;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
  }

  // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
  // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
  // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
  // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
  // argmax after computing the softmax.
  const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
  }

  // Now, softmax_res contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
  // with the max index.​
  int start_col = first_elt_read_by_thread;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  float output_row_sum = 0.f;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    // First, each thread does the local argmax
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        float val = row_chunk[ldg * ELTS_PER_LDG + ii];

        // No check on the experts here since columns with the smallest index are processed first and only
        // updated if > (not >=)
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

      // We want lower indices to "win" in every thread so we break ties this way
      if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    // Write the max for this k iteration to global memory.
    if (thread_group_idx == 0) {
      // The lead thread from each sub-group will write out the final results to global memory. (This will be a
      // single) thread per row of the input/output matrices.
      const int idx = k * thread_row + k_idx;
      output[idx] = max_val;
      output_row_sum = output_row_sum + max_val;
      indices[idx] = should_process_row ? expert : NUM_EXPERTS;
      source_rows[idx] = k_idx * num_rows + thread_row;

      if (normalize_routing_weights && k_idx == k - 1) {
#pragma unroll
        for (int ki = 0; ki < k; ++ki) {
          output[idx - ki] = output[idx - ki] / output_row_sum;
        }
      }
    }

    // Finally, we clear the value in the thread with the current max if there is another iteration to run.
    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

      // Only the thread in the group which produced the max will reset the "winning" value to -inf.
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        // Safe to set to any negative value since row_chunk values must be between 0 and 1.
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = ComputeType(-10000.f);
      }
    }
  }
}

namespace detail {
// Constructs some constants needed to partition the work across threads at compile time.
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
  static constexpr int VECs_PER_THREAD = std::max(1, (int)EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

template <typename T, int EXPERTS, int WARPS_PER_TB>
void topk_gating_softmax_launcher_helper(const T* input, const bool* finished, float* output, int* indices, int* source_row,
                                         int num_rows, int /*num_experts*/, int k, bool normalize_routing_weights,
                                         cudaStream_t stream) {
  static constexpr unsigned long MAX_BYTES_PER_LDG = 16;

  static constexpr int BYTES_PER_LDG = std::min((int)MAX_BYTES_PER_LDG, (int)sizeof(T) * EXPERTS);
  using Constants = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
  topk_gating_softmax<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
      input, finished, output, num_rows, indices, source_row, k, normalize_routing_weights);
}

template <typename T>
void topk_gating_softmax_kernelLauncher(const T* input, const bool* finished, float* output, T* softmax_temp_output,
                                        int* indices, int* source_row, int num_rows, int num_experts, int k,
                                        bool normalize_routing_weights, bool use_sparse_mixer, cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;

  if (use_sparse_mixer) {
    static constexpr int TPB = WARP_SIZE * WARPS_PER_TB;
    static constexpr float jitter_eps = 0.01f;

    switch (num_experts) {
      case 8: {
        sparse_mixer_top2<T, TPB, 8><<<num_rows, TPB, 0, stream>>>(input, output, indices, source_row, jitter_eps);
        break;
      }
      case 16: {
        sparse_mixer_top2<T, TPB, 16><<<num_rows, TPB, 0, stream>>>(input, output, indices, source_row, jitter_eps);
        break;
      }

      default: {
        ORT_THROW("Sparse mixer only supports 8 and 16 experts");
      }
    }
    return;
  }

  switch (num_experts) {
    case 2: {
      topk_gating_softmax_launcher_helper<T, 2, WARPS_PER_TB>(input, finished, output, indices, source_row, num_rows,
                                                              num_experts, k, normalize_routing_weights, stream);
      break;
    }
    case 4: {
      topk_gating_softmax_launcher_helper<T, 4, WARPS_PER_TB>(input, finished, output, indices, source_row, num_rows,
                                                              num_experts, k, normalize_routing_weights, stream);
      break;
    }
    case 8: {
      topk_gating_softmax_launcher_helper<T, 8, WARPS_PER_TB>(input, finished, output, indices, source_row, num_rows,
                                                              num_experts, k, normalize_routing_weights, stream);
      break;
    }
    case 16: {
      topk_gating_softmax_launcher_helper<T, 16, WARPS_PER_TB>(input, finished, output, indices, source_row, num_rows,
                                                               num_experts, k, normalize_routing_weights, stream);
      break;
    }
    case 32: {
      topk_gating_softmax_launcher_helper<T, 32, WARPS_PER_TB>(input, finished, output, indices, source_row, num_rows,
                                                               num_experts, k, normalize_routing_weights, stream);
      break;
    }
    case 64: {
      topk_gating_softmax_launcher_helper<T, 64, WARPS_PER_TB>(input, finished, output, indices, source_row, num_rows,
                                                               num_experts, k, normalize_routing_weights, stream);
      break;
    }
    case 128: {
      topk_gating_softmax_launcher_helper<T, 128, WARPS_PER_TB>(
          input, finished, output, indices, source_row, num_rows, num_experts, k, normalize_routing_weights, stream);
      break;
    }
    case 256: {
      topk_gating_softmax_launcher_helper<T, 256, WARPS_PER_TB>(
          input, finished, output, indices, source_row, num_rows, num_experts, k, normalize_routing_weights, stream);
      break;
    }
    default: {
      static constexpr int TPB = 256;
      moe_softmax<T, TPB><<<num_rows, TPB, 0, stream>>>(input, finished, softmax_temp_output, num_experts);
      moe_top_k<T, TPB><<<num_rows, TPB, 0, stream>>>(softmax_temp_output, finished, output, indices, source_row,
                                                      num_experts, k, normalize_routing_weights);
    }
  }
}

// ========================= specializations ===========================
template void topk_gating_softmax_kernelLauncher(const float*, const bool*, float*, float*, int*, int*, int, int,
                                                 int, bool, bool, cudaStream_t);
template void topk_gating_softmax_kernelLauncher(const half*, const bool*, float*, half*, int*, int*, int, int,
                                                 int, bool, bool, cudaStream_t);
template void topk_gating_softmax_kernelLauncher(const __nv_bfloat16*, const bool*, float*, __nv_bfloat16*, int*, int*, int, int,
                                                 int, bool, bool, cudaStream_t);

}  // namespace onnxruntime::contrib::cuda