// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/moe/moe_cpu.h"
#include "contrib_ops/cpu/moe/moe_utils.h"
#include "contrib_ops/cpu/moe/moe_helper.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

#include <algorithm>
#include <vector>
#include <cstring>

namespace onnxruntime {
namespace contrib {

MoE::MoE(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info), MoEBaseCPU(op_kernel_info) {
}

Status MoE::Compute(OpKernelContext* context) const {
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

Status MoE::MoEImpl(OpKernelContext* context,
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

  const float* input_data = input->Data<float>();
  const float* router_data = router_probs->Data<float>();
  const float* fc1_weights_data = fc1_experts_weights->Data<float>();
  const float* fc1_bias_data = fc1_experts_bias ? fc1_experts_bias->Data<float>() : nullptr;
  const float* fc2_weights_data = fc2_experts_weights->Data<float>();
  const float* fc2_bias_data = fc2_experts_bias ? fc2_experts_bias->Data<float>() : nullptr;
  const float* fc3_weights_data = fc3_experts_weights ? fc3_experts_weights->Data<float>() : nullptr;
  const float* fc3_bias_data = fc3_experts_bias ? fc3_experts_bias->Data<float>() : nullptr;

  float* output_data = output->MutableData<float>();

  // Initialize output to zero
  std::fill_n(output_data, output->Shape().Size(), 0.0f);

  // Process each row
  for (int64_t row = 0; row < num_rows; ++row) {
    const float* current_input = input_data + row * hidden_size;
    const float* current_router = router_data + row * num_experts;
    float* current_output = output_data + row * hidden_size;

    // Find top-k experts for this row
    std::vector<std::pair<float, int64_t>> expert_scores;
    expert_scores.reserve(num_experts);  // Pre-allocate to avoid reallocations
    for (int64_t expert = 0; expert < num_experts; ++expert) {
      expert_scores.emplace_back(current_router[expert], expert);
    }

    // Sort by score in descending order and take top-k
    std::partial_sort(expert_scores.begin(), expert_scores.begin() + k_, expert_scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Normalize routing weights if required
    float total_weight = 0.0f;
    if (normalize_routing_weights_) {
      for (int64_t i = 0; i < k_; ++i) {
        total_weight += expert_scores[i].first;
      }
      // Check for numerical stability - avoid division by very small numbers
      if (total_weight > 1e-8f) {
        for (int64_t i = 0; i < k_; ++i) {
          expert_scores[i].first /= total_weight;
        }
      } else {
        // If total weight is too small, set uniform weights
        for (int64_t i = 0; i < k_; ++i) {
          expert_scores[i].first = 1.0f / static_cast<float>(k_);
        }
      }
    }

    // Accumulate outputs from top-k experts
    std::vector<float> expert_output(hidden_size);
    for (int64_t i = 0; i < k_; ++i) {
      const float weight = expert_scores[i].first;
      const int64_t expert_idx = expert_scores[i].second;

      if (weight <= 0.0f) continue;

      // Validate expert index to prevent out-of-bounds access
      if (expert_idx < 0 || expert_idx >= num_experts) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Expert index out of bounds: ", expert_idx, " (valid range: 0-", num_experts - 1, ")");
      }

      // Calculate weight offsets based on layout
      const float* fc1_expert_weights;
      const float* fc2_expert_weights;

      if (legacy_shape) {
        // Legacy layout: [num_experts, hidden_size, inter_size] for fc1
        fc1_expert_weights = fc1_weights_data + expert_idx * hidden_size * fc1_inter_size;
        fc2_expert_weights = fc2_weights_data + expert_idx * inter_size * hidden_size;
      } else {
        // New layout: [num_experts, inter_size, hidden_size] for fc1
        fc1_expert_weights = fc1_weights_data + expert_idx * fc1_inter_size * hidden_size;
        fc2_expert_weights = fc2_weights_data + expert_idx * hidden_size * inter_size;
      }

      const float* fc1_expert_bias = fc1_bias_data ? fc1_bias_data + expert_idx * fc1_inter_size : nullptr;
      const float* fc2_expert_bias = fc2_bias_data ? fc2_bias_data + expert_idx * hidden_size : nullptr;

      // Reset expert output
      std::fill(expert_output.begin(), expert_output.end(), 0.0f);

      // Process this expert
      ORT_RETURN_IF_ERROR(ProcessExpert(current_input, fc1_expert_weights, fc1_expert_bias,
                                        fc2_expert_weights, fc2_expert_bias,
                                        expert_output.data(), hidden_size, inter_size, fc1_inter_size,
                                        legacy_shape));

      // Accumulate weighted expert output
      for (int64_t j = 0; j < hidden_size; ++j) {
        current_output[j] += weight * expert_output[j];
      }
    }
  }

  return Status::OK();
}

Status MoE::ProcessExpert(const float* input_data,
                          const float* fc1_weights,
                          const float* fc1_bias,
                          const float* fc2_weights,
                          const float* fc2_bias,
                          float* output_data,
                          int64_t hidden_size,
                          int64_t inter_size,
                          int64_t fc1_inter_size,
                          bool legacy_shape) const {
  const bool is_swiglu = (activation_type_ == ActivationType::SwiGLU);

  // Allocate intermediate buffer
  std::vector<float> fc1_output(fc1_inter_size);

  // Validate buffer sizes to prevent memory corruption
  if (fc1_inter_size <= 0 || hidden_size <= 0 || inter_size <= 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid tensor dimensions: hidden_size=", hidden_size,
                           ", inter_size=", inter_size, ", fc1_inter_size=", fc1_inter_size);
  }

  // FC1: input -> intermediate using MLAS GEMM for better performance
  // For legacy layout: weights are [hidden_size, fc1_inter_size], stored row-major
  // For new layout: weights are [fc1_inter_size, hidden_size], stored row-major
  // MLAS expects: C = A * B^T + bias (if beta=0, bias is ignored in GEMM)

  MLAS_SGEMM_DATA_PARAMS fc1_params;
  fc1_params.A = input_data;     // input: [1, hidden_size]
  fc1_params.lda = hidden_size;  // leading dimension of A (single row, so stride = hidden_size)
  fc1_params.alpha = 1.0f;
  fc1_params.beta = 0.0f;
  fc1_params.C = fc1_output.data();
  fc1_params.ldc = fc1_inter_size;  // leading dimension of C (single row, so stride = fc1_inter_size)

  if (legacy_shape) {
    // Legacy: weights [hidden_size, fc1_inter_size] stored row-major
    // GEMM: A[1, hidden_size] * B^T[hidden_size, fc1_inter_size] = C[1, fc1_inter_size]
    // Before transpose, B is [hidden_size, fc1_inter_size] with ldb = fc1_inter_size
    fc1_params.B = fc1_weights;
    fc1_params.ldb = fc1_inter_size;  // leading dimension of B before transpose
    MlasGemm(CblasNoTrans, CblasTrans, 1, fc1_inter_size, hidden_size, fc1_params, nullptr);
  } else {
    // New: weights [fc1_inter_size, hidden_size] stored row-major
    // GEMM: A[1, hidden_size] * B^T[fc1_inter_size, hidden_size] = C[1, fc1_inter_size]
    // Before transpose, B is [fc1_inter_size, hidden_size] with ldb = hidden_size
    fc1_params.B = fc1_weights;
    fc1_params.ldb = hidden_size;  // leading dimension of B before transpose
    MlasGemm(CblasNoTrans, CblasTrans, 1, fc1_inter_size, hidden_size, fc1_params, nullptr);
  }

  // Add bias if present
  if (fc1_bias) {
    for (int64_t i = 0; i < fc1_inter_size; ++i) {
      fc1_output[i] += fc1_bias[i];
    }
  }

  // Apply activation function
  std::vector<float> fc2_input_buffer;
  float* fc2_input = fc1_output.data();  // Default for non-SwiGLU case

  if (is_swiglu) {
    // For SwiGLU, we need separate input and output buffers
    // FC1 output has size fc1_inter_size (2 * inter_size)
    // FC2 input has size inter_size after SwiGLU transformation
    fc2_input_buffer.resize(inter_size);
    fc2_input = fc2_input_buffer.data();

    // Apply SwiGLU activation: transform fc1_output[2*inter_size] -> fc2_input[inter_size]
    ApplySwiGLUActivation(fc1_output.data(), fc2_input, inter_size, true,
                          activation_alpha_, activation_beta_, swiglu_limit_);
  } else {
    ApplyActivationInPlace(fc1_output.data(), fc1_inter_size);
  }

  // FC2: intermediate -> output using MLAS GEMM
  // FC2 input is either the activated fc1_output (non-SwiGLU) or fc2_input_buffer (SwiGLU)
  // Both have size inter_size

  MLAS_SGEMM_DATA_PARAMS fc2_params;
  fc2_params.A = fc2_input;     // intermediate: [1, inter_size]
  fc2_params.lda = inter_size;  // leading dimension of A (single row, so stride = inter_size)
  fc2_params.alpha = 1.0f;
  fc2_params.beta = 0.0f;
  fc2_params.C = output_data;
  fc2_params.ldc = hidden_size;  // leading dimension of C (single row, so stride = hidden_size)

  if (legacy_shape) {
    // Legacy: weights [inter_size, hidden_size] stored row-major
    // GEMM: A[1, inter_size] * B^T[inter_size, hidden_size] = C[1, hidden_size]
    // Before transpose, B is [inter_size, hidden_size] with ldb = hidden_size
    fc2_params.B = fc2_weights;
    fc2_params.ldb = hidden_size;  // leading dimension of B before transpose
    MlasGemm(CblasNoTrans, CblasTrans, 1, hidden_size, inter_size, fc2_params, nullptr);
  } else {
    // New: weights [hidden_size, inter_size] stored row-major
    // GEMM: A[1, inter_size] * B^T[hidden_size, inter_size] = C[1, hidden_size]
    // Before transpose, B is [hidden_size, inter_size] with ldb = inter_size
    fc2_params.B = fc2_weights;
    fc2_params.ldb = inter_size;  // leading dimension of B before transpose
    MlasGemm(CblasNoTrans, CblasTrans, 1, hidden_size, inter_size, fc2_params, nullptr);
  }

  // Add bias if present
  if (fc2_bias) {
    for (int64_t i = 0; i < hidden_size; ++i) {
      output_data[i] += fc2_bias[i];
    }
  }

  return Status::OK();
}

void MoE::ApplyActivationInPlace(float* data, int64_t size, bool is_swiglu_format) const {
  if (is_swiglu_format) {
    // This is handled in SwiGLU specific function
    return;
  }

  for (int64_t i = 0; i < size; ++i) {
    data[i] = ApplyActivation(data[i], activation_type_);
  }
}

#define REGISTER_KERNEL()                                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      MoE, kMSDomain, 1, float, kCpuExecutionProvider,                \
      (*KernelDefBuilder::Create())                                   \
          .MayInplace(0, 0)                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      MoE);

REGISTER_KERNEL()

}  // namespace contrib
}  // namespace onnxruntime
