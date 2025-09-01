// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*
 * CPU MoE Implementation with CUDA-like Batched Processing
 *
 * This implementation adopts a batched processing approach similar to CUDA MoE while
 * maintaining compatibility with MLAS (Microsoft Linear Algebra Subprograms) which
 * requires row-major tensor layouts.
 *
 * Key Differences from CUDA MoE:
 * 1. Weight Layout: Uses row-major (MLAS) vs column-major (CUTLASS)
 * 2. Memory Management: Uses std::vector vs GPU memory allocation
 * 3. Processing: Sequential batched GEMM vs parallel GPU kernels
 * 4. Data Organization: Explicit permutation arrays vs implicit GPU indexing
 *
 * Layout Compatibility:
 * - All tensors stored in row-major format for MLAS compatibility
 * - Uses transpose operations (B^T) in GEMM to handle different logical layouts
 * - Supports both legacy and new tensor shape formats
 * - Maintains same layout detection logic as CUDA version
 */

#include "contrib_ops/cpu/moe/moe_cpu.h"
#include "contrib_ops/cpu/moe/moe_utils.h"
#include "contrib_ops/cpu/moe/moe_helper.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#include "core/framework/float16.h"

#include <algorithm>
#include <vector>
#include <cstring>

namespace onnxruntime {
namespace contrib {

template <typename T>
MoE<T>::MoE(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info), MoEBaseCPU(op_kernel_info) {
}

template <typename T>
Status MoE<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_experts_bias = context->Input<Tensor>(3);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(4);
  const Tensor* fc2_experts_bias = context->Input<Tensor>(5);
  const Tensor* fc3_experts_weights = context->Input<Tensor>(6);
  const Tensor* fc3_experts_bias = context->Input<Tensor>(7);

  // Validate inputs using the shared helper
  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias, nullptr,
      fc2_experts_weights, fc2_experts_bias, nullptr,
      fc3_experts_weights, fc3_experts_bias, nullptr,
      1,  // no quantization so pack size is 1
      activation_type_ == ActivationType::SwiGLU));

  Tensor* output = context->Output(0, input->Shape());

  return MoEImpl(context, input, router_probs,
                 fc1_experts_weights, fc1_experts_bias,
                 fc2_experts_weights, fc2_experts_bias,
                 fc3_experts_weights, fc3_experts_bias,
                 output);
}

template <typename T>
Status MoE<T>::MoEImpl(OpKernelContext* context,
                       const Tensor* input,
                       const Tensor* router_probs,
                       const Tensor* fc1_experts_weights,
                       const Tensor* fc1_experts_bias,
                       const Tensor* fc2_experts_weights,
                       const Tensor* fc2_experts_bias,
                       const Tensor* fc3_experts_weights,
                       const Tensor* fc3_experts_bias,
                       Tensor* output) const {
  // Check if FC3 is present and throw error
  if (fc3_experts_weights != nullptr || fc3_experts_bias != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "FC3 is not implemented for CPU MoE. Please use GPU implementation for FC3 support.");
  }

  const auto& input_shape = input->Shape();
  const auto& router_shape = router_probs->Shape();
  const auto& fc1_weight_shape = fc1_experts_weights->Shape();
  const auto& fc2_weight_shape = fc2_experts_weights->Shape();

  const int64_t num_rows = input_shape.Size() / input_shape[input_shape.NumDimensions() - 1];
  const int64_t hidden_size = input_shape[input_shape.NumDimensions() - 1];
  const int64_t num_experts = router_shape[1];

  // Determine layout and sizes based on MoE helper logic
  const bool is_swiglu = activation_type_ == ActivationType::SwiGLU;
  // Calculate inter_size exactly like the helper function
  int64_t inter_size = (fc2_weight_shape[1] * fc2_weight_shape[2]) / hidden_size;

  // Use the same layout detection as CUDA (via helper function)
  const bool legacy_shape = (hidden_size != inter_size && fc2_weight_shape[1] == inter_size) ||
                            (hidden_size == inter_size && is_swiglu && fc1_weight_shape[1] == hidden_size);

  const int64_t fc1_inter_size = is_swiglu ? (inter_size * 2) : inter_size;

  const T* input_data = input->Data<T>();
  const T* router_data = router_probs->Data<T>();
  const T* fc1_weights_data = fc1_experts_weights->Data<T>();
  const T* fc1_bias_data = fc1_experts_bias ? fc1_experts_bias->Data<T>() : nullptr;
  const T* fc2_weights_data = fc2_experts_weights->Data<T>();
  const T* fc2_bias_data = fc2_experts_bias ? fc2_experts_bias->Data<T>() : nullptr;
  const T* fc3_weights_data = fc3_experts_weights ? fc3_experts_weights->Data<T>() : nullptr;
  const T* fc3_bias_data = fc3_experts_bias ? fc3_experts_bias->Data<T>() : nullptr;

  T* output_data = output->MutableData<T>();

  // Initialize output to zero with proper type
  const T zero_value = static_cast<T>(0.0f);
  std::fill_n(output_data, output->Shape().Size(), zero_value);

  // Use batched processing approach similar to CUDA for better performance
  return MoEImplBatched(input, router_probs, fc1_experts_weights, fc1_experts_bias,
                        fc2_experts_weights, fc2_experts_bias, output,
                        num_rows, hidden_size, num_experts, inter_size, fc1_inter_size, legacy_shape);
}
// Batched CPU MoE implementation similar to CUDA approach but using row-major MLAS layout
//
// Layout Analysis:
// - Input tensors: Row-major [num_rows, hidden_size]
// - FC1 weights: Row-major storage, logical layout depends on legacy_shape:
//   * Legacy: [num_experts, hidden_size, fc1_inter_size] stored row-major
//   * New: [num_experts, fc1_inter_size, hidden_size] stored row-major
// - FC2 weights: Row-major storage, logical layout depends on legacy_shape:
//   * Legacy: [num_experts, inter_size, hidden_size] stored row-major
//   * New: [num_experts, hidden_size, inter_size] stored row-major
// - MLAS GEMM: Always expects row-major C = A * B^T
//
template <typename T>
Status MoE<T>::MoEImplBatched(const Tensor* input, const Tensor* router_probs,
                              const Tensor* fc1_experts_weights, const Tensor* fc1_experts_bias,
                              const Tensor* fc2_experts_weights, const Tensor* fc2_experts_bias,
                              Tensor* output, int64_t num_rows, int64_t hidden_size, int64_t num_experts,
                              int64_t inter_size, int64_t fc1_inter_size, bool legacy_shape) const {
  const T* input_data = input->Data<T>();
  const T* router_data = router_probs->Data<T>();
  const T* fc1_weights_data = fc1_experts_weights->Data<T>();
  const T* fc1_bias_data = fc1_experts_bias ? fc1_experts_bias->Data<T>() : nullptr;
  const T* fc2_weights_data = fc2_experts_weights->Data<T>();
  const T* fc2_bias_data = fc2_experts_bias ? fc2_experts_bias->Data<T>() : nullptr;
  T* output_data = output->MutableData<T>();

  // Initialize output tensor to zero for accumulation
  const int64_t output_size = num_rows * hidden_size;
  std::fill(output_data, output_data + output_size, T{});

  // Step 1: Collect and organize all active tokens (similar to CUDA's initialize_moe_routing)
  struct TokenExpertAssignment {
    int64_t source_row;
    int64_t expert_idx;
    float weight;
    int64_t dest_row;  // Position in permuted batch
  };

  std::vector<TokenExpertAssignment> assignments;
  assignments.reserve(num_rows * num_experts);  // Upper bound: all experts for all tokens

  // Collect all token-expert assignments
  for (int64_t row = 0; row < num_rows; ++row) {
    const T* current_router = router_data + row * num_experts;

    // router_probs already contains processed probabilities (top-k selected and normalized)
    // We just need to extract non-zero entries without additional processing
    for (int64_t expert = 0; expert < num_experts; ++expert) {
      const T weight = current_router[expert];
      if (static_cast<float>(weight) > 1e-8f) {  // Only include non-zero weights
        assignments.push_back({row, expert, static_cast<float>(weight), static_cast<int64_t>(assignments.size())});
      }
    }
  }

  const int64_t total_active_tokens = static_cast<int64_t>(assignments.size());
  if (total_active_tokens == 0) {
    return Status::OK();  // No active tokens, output remains zero
  }

  // Step 2: Sort assignments by expert for batched processing
  std::sort(assignments.begin(), assignments.end(),
            [](const TokenExpertAssignment& a, const TokenExpertAssignment& b) {
              return a.expert_idx < b.expert_idx;
            });

  // Step 3: Build expert boundaries for batched GEMM
  std::vector<int64_t> expert_token_counts(num_experts, 0);
  std::vector<int64_t> expert_start_idx(num_experts, 0);

  // Count tokens per expert
  for (const auto& assignment : assignments) {
    expert_token_counts[assignment.expert_idx]++;
  }

  // Calculate start indices
  int64_t cumulative = 0;
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    expert_start_idx[expert] = cumulative;
    cumulative += expert_token_counts[expert];
  }

  // Step 4: Create permuted input data (row-major layout for MLAS)
  std::vector<T> permuted_input(total_active_tokens * hidden_size);
  for (int64_t i = 0; i < total_active_tokens; ++i) {
    const int64_t source_row = assignments[i].source_row;
    const T* source_ptr = input_data + source_row * hidden_size;
    T* dest_ptr = permuted_input.data() + i * hidden_size;
    std::copy(source_ptr, source_ptr + hidden_size, dest_ptr);
  }

  // Step 5: Allocate intermediate buffers
  std::vector<T> fc1_output(total_active_tokens * fc1_inter_size);
  std::vector<T> fc2_input_buffer;
  std::vector<T> fc2_output(total_active_tokens * hidden_size);  // Step 6: Process each expert with batched GEMM (maintaining row-major layout)
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    const int64_t tokens_for_expert = expert_token_counts[expert];
    if (tokens_for_expert == 0) continue;

    const int64_t start_idx = expert_start_idx[expert];

    // Calculate expert weight pointers based on layout
    const T* fc1_expert_weights;
    const T* fc2_expert_weights;

    if (legacy_shape) {
      // Legacy layout: [num_experts, hidden_size, fc1_inter_size] for fc1
      fc1_expert_weights = fc1_weights_data + expert * hidden_size * fc1_inter_size;
      fc2_expert_weights = fc2_weights_data + expert * inter_size * hidden_size;
    } else {
      // New layout: [num_experts, fc1_inter_size, hidden_size] for fc1
      fc1_expert_weights = fc1_weights_data + expert * fc1_inter_size * hidden_size;
      fc2_expert_weights = fc2_weights_data + expert * hidden_size * inter_size;
    }

    const T* fc1_expert_bias = fc1_bias_data ? fc1_bias_data + expert * fc1_inter_size : nullptr;
    const T* fc2_expert_bias = fc2_bias_data ? fc2_bias_data + expert * hidden_size : nullptr;

    // FC1 Batched GEMM: [tokens_for_expert, hidden_size] * [fc1_inter_size, hidden_size]^T
    ORT_RETURN_IF_ERROR(ProcessExpertBatch(
        permuted_input.data() + start_idx * hidden_size,  // input batch
        fc1_expert_weights, fc1_expert_bias,
        fc2_expert_weights, fc2_expert_bias,
        fc1_output.data() + start_idx * fc1_inter_size,  // fc1 output batch
        fc2_output.data() + start_idx * hidden_size,     // final output batch
        tokens_for_expert, hidden_size, inter_size, fc1_inter_size, legacy_shape));
  }

  // Step 7: Finalize routing - accumulate results back to original positions (similar to CUDA's finalize_moe_routing)
  for (int64_t i = 0; i < total_active_tokens; ++i) {
    const TokenExpertAssignment& assignment = assignments[i];
    const float weight = assignment.weight;
    const int64_t source_row = assignment.source_row;

    const T* expert_output = fc2_output.data() + i * hidden_size;
    T* target_output = output_data + source_row * hidden_size;

    // Accumulate weighted expert output with proper type conversion
    float weight_f = weight;
    for (int64_t j = 0; j < hidden_size; ++j) {
      target_output[j] = static_cast<T>(static_cast<float>(target_output[j]) + weight_f * static_cast<float>(expert_output[j]));
    }
  }

  return Status::OK();
}

// Batched expert processing using MLAS with row-major layout
//
// Layout Verification:
// - All tensors use row-major storage (compatible with MLAS)
// - Input batch: [batch_size, hidden_size] row-major
// - FC1 weights: Stored row-major, but logical shape depends on legacy_shape
// - FC2 weights: Stored row-major, but logical shape depends on legacy_shape
// - GEMM operations: A * B^T where B is transposed to match MLAS expectations
//
template <typename T>
Status MoE<T>::ProcessExpertBatch(const T* input_batch,
                                  const T* fc1_weights, const T* fc1_bias,
                                  const T* fc2_weights, const T* fc2_bias,
                                  T* fc1_output_batch, T* final_output_batch,
                                  int64_t batch_size, int64_t hidden_size, int64_t inter_size,
                                  int64_t fc1_inter_size, bool legacy_shape) const {
  const bool is_swiglu = (activation_type_ == ActivationType::SwiGLU);

  // Validate batch dimensions
  if (batch_size <= 0 || fc1_inter_size <= 0 || hidden_size <= 0 || inter_size <= 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid batch dimensions: batch_size=", batch_size,
                           ", hidden_size=", hidden_size, ", inter_size=", inter_size);
  }

  // Dispatch to appropriate GEMM implementation based on data type
  ORT_RETURN_IF_ERROR(ProcessGemm(input_batch, fc1_weights, fc1_output_batch,
                                  batch_size, hidden_size, fc1_inter_size, legacy_shape, true));

  // Add FC1 bias if present
  if (fc1_bias) {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      T* batch_output = fc1_output_batch + batch * fc1_inter_size;
      for (int64_t i = 0; i < fc1_inter_size; ++i) {
        batch_output[i] = static_cast<T>(static_cast<float>(batch_output[i]) + static_cast<float>(fc1_bias[i]));
      }
    }
  }

  // Apply activation function
  T* fc2_input_data = fc1_output_batch;  // Default for non-SwiGLU case
  std::vector<T> fc2_input_buffer;

  if (is_swiglu) {
    // For SwiGLU, allocate separate buffer for FC2 input
    fc2_input_buffer.resize(batch_size * inter_size);
    fc2_input_data = fc2_input_buffer.data();

    // Apply SwiGLU activation: transform fc1_output[batch_size, 2*inter_size] -> fc2_input[batch_size, inter_size]
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      ApplySwiGLUActivationTyped(fc1_output_batch + batch * fc1_inter_size,
                                 fc2_input_data + batch * inter_size,
                                 inter_size);
    }
  } else {
    // Apply activation in-place for non-SwiGLU
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      ApplyActivationInPlace(fc1_output_batch + batch * fc1_inter_size, fc1_inter_size);
    }
  }

  // FC2 Batched GEMM: [batch_size, inter_size] * [hidden_size, inter_size]^T = [batch_size, hidden_size]
  ORT_RETURN_IF_ERROR(ProcessGemm(fc2_input_data, fc2_weights, final_output_batch,
                                  batch_size, inter_size, hidden_size, legacy_shape, false));

  // Add FC2 bias if present
  if (fc2_bias) {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      T* batch_output = final_output_batch + batch * hidden_size;
      for (int64_t i = 0; i < hidden_size; ++i) {
        batch_output[i] = static_cast<T>(static_cast<float>(batch_output[i]) + static_cast<float>(fc2_bias[i]));
      }
    }
  }

  return Status::OK();
}

// Helper function to handle GEMM operations for different data types
template <>
Status MoE<float>::ProcessGemm(const float* A, const float* B, float* C,
                               int64_t M, int64_t K, int64_t N, bool legacy_shape, bool is_fc1) const {
  MLAS_SGEMM_DATA_PARAMS params;
  params.A = A;
  params.lda = K;
  params.alpha = 1.0f;
  params.beta = 0.0f;
  params.C = C;
  params.ldc = N;

  if ((legacy_shape && is_fc1) || (!legacy_shape && !is_fc1)) {
    // No transpose needed: A[M, K] * B[K, N] = C[M, N]
    params.B = B;
    params.ldb = N;
    MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, params, nullptr);
  } else {
    // Transpose needed: A[M, K] * B^T[N, K] = C[M, N]
    params.B = B;
    params.ldb = K;
    MlasGemm(CblasNoTrans, CblasTrans, M, N, K, params, nullptr);
  }

  return Status::OK();
}

// Specialized implementation for MLFloat16
template <>
Status MoE<MLFloat16>::ProcessGemm(const MLFloat16* A, const MLFloat16* B, MLFloat16* C,
                                   int64_t M, int64_t K, int64_t N, bool legacy_shape, bool is_fc1) const {
  MLAS_HALF_GEMM_DATA_PARAMS params;
  params.A = A;
  params.lda = K;
  params.C = C;
  params.ldc = N;
  params.AIsfp32 = false;
  params.BIsfp32 = false;

  if ((legacy_shape && is_fc1) || (!legacy_shape && !is_fc1)) {
    // No transpose needed: A[M, K] * B[K, N] = C[M, N]
    params.B = B;
    params.ldb = N;
  } else {
    // Transpose needed: A[M, K] * B^T[N, K] = C[M, N]
    params.B = B;
    params.ldb = K;
  }

  MlasHalfGemmBatch(M, N, K, 1, &params, nullptr);

  return Status::OK();
}

template <typename T>
Status MoE<T>::ProcessExpert(const T* input_data,
                             const T* fc1_weights,
                             const T* fc1_bias,
                             const T* fc2_weights,
                             const T* fc2_bias,
                             T* output_data,
                             int64_t hidden_size,
                             int64_t inter_size,
                             int64_t fc1_inter_size,
                             bool legacy_shape) const {
  // Use a temporary buffer for FC1 output since ProcessExpertBatch expects separate buffers
  std::vector<T> fc1_temp(fc1_inter_size);

  // Delegate to batched implementation for single token
  return ProcessExpertBatch(input_data, fc1_weights, fc1_bias, fc2_weights, fc2_bias,
                            fc1_temp.data(), output_data, 1, hidden_size, inter_size, fc1_inter_size, legacy_shape);
}

template <typename T>
void MoE<T>::ApplyActivationInPlace(T* data, int64_t size, bool is_swiglu_format) const {
  if (is_swiglu_format) {
    // This is handled in SwiGLU specific function
    return;
  }

  for (int64_t i = 0; i < size; ++i) {
    data[i] = static_cast<T>(ApplyActivation(static_cast<float>(data[i]), activation_type_));
  }
}

// SwiGLU activation function for templated types (GPT-OSS specific implementation)
template <typename T>
void MoE<T>::ApplySwiGLUActivationTyped(const T* input, T* output, int64_t size) const {
  // Determine whether the input is interleaved based on kernel attribute or fusion flag.
  const bool input_is_interleaved = (swiglu_fusion_ == 1) || swiglu_interleaved_;

  if (input_is_interleaved) {
    // Input is in interleaved format: [gate_0, linear_0, gate_1, linear_1, ...]
    for (int64_t i = 0; i < size; ++i) {
      float gate_val = static_cast<float>(input[2 * i]);        // interleaved: gate at even indices
      float linear_val = static_cast<float>(input[2 * i + 1]);  // interleaved: linear at odd indices

      gate_val = std::min(gate_val, swiglu_limit_);
      linear_val = std::clamp(linear_val, -swiglu_limit_, swiglu_limit_);

      float sigmoid_arg = activation_alpha_ * gate_val;
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      float swish_out = gate_val * sigmoid_out;

      float activated = swish_out * (linear_val + activation_beta_);
      output[i] = static_cast<T>(activated);
    }
  } else {
    // Non-interleaved (concatenated) format: [gate_0, gate_1, ..., linear_0, linear_1, ...]
    // First half: gates, second half: linears
    for (int64_t i = 0; i < size; ++i) {
      float gate_val = static_cast<float>(input[i]);
      float linear_val = static_cast<float>(input[size + i]);

      gate_val = std::min(gate_val, swiglu_limit_);
      linear_val = std::clamp(linear_val, -swiglu_limit_, swiglu_limit_);

      float sigmoid_arg = activation_alpha_ * gate_val;
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      float swish_out = gate_val * sigmoid_out;

      float activated = swish_out * (linear_val + activation_beta_);
      output[i] = static_cast<T>(activated);
    }
  }
}

// Specialized float version using the GPT-OSS implementation from moe_utils.cc
template <>
void MoE<float>::ApplySwiGLUActivationTyped(const float* input, float* output, int64_t size) const {
  // Determine whether the input is interleaved based on kernel attribute or fusion flag.
  const bool input_is_interleaved = (swiglu_fusion_ == 1) || swiglu_interleaved_;

  // Use the existing GPT-OSS implementation and pass the correct format flag
  ApplySwiGLUActivation(input, output, size, input_is_interleaved,
                        activation_alpha_, activation_beta_, swiglu_limit_);
}

// Explicit template instantiations
template class MoE<float>;
template class MoE<MLFloat16>;

#define REGISTER_KERNEL_TYPED(type)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      MoE, kMSDomain, 1, type, kCpuExecutionProvider,                \
      (*KernelDefBuilder::Create())                                  \
          .MayInplace(0, 0)                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      MoE<type>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
