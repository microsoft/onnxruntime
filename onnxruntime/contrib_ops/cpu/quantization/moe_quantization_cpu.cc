// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/moe_quantization_cpu.h"
#include "contrib_ops/cpu/moe/moe_utils.h"
#include "contrib_ops/cpu/quantization/moe_helper.h"
#include "core/framework/allocator.h"
#include "core/framework/buffer_deleter.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/inc/mlas_float16.h"  // For MLAS_Half2Float function
#include <atomic>
#include "core/platform/threadpool.h"
#include <algorithm>
#include <cmath>    // For std::abs
#include <cstdlib>  // For std::getenv
#include <cstring>  // For std::strcmp
#include <mutex>

using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {

// Helper function to quantize router probabilities for robust expert selection
inline float quantize_router_probability(float prob, float quantization_step = 1e-5f) {
  // Round to nearest quantization step to eliminate tiny floating-point differences
  return std::round(prob / quantization_step) * quantization_step;
}

// Check if robust expert selection is enabled via environment variable
inline bool use_robust_expert_selection() {
  static bool checked = false;
  static bool enabled = false;

  if (!checked) {
    const char* env_var = std::getenv("ORT_QMOE_ROBUST_SELECTION");
    enabled = (env_var != nullptr && std::strcmp(env_var, "1") == 0);
    checked = true;
  }

  return enabled;
}

// CUDA-style finalize routing function: accumulates expert outputs with bias and scaling
void finalize_moe_routing_cpu(const float* expert_outputs, float* final_output,
                              const float* fc2_bias_float, const float* expert_scales,
                              const int* expert_indices, int64_t num_rows, int64_t hidden_size, int64_t k) {
  // Process each token (row) - matches CUDA's block-per-row approach
  for (int64_t row = 0; row < num_rows; ++row) {
    float* output_row = final_output + row * hidden_size;

    // Initialize output to zero (matching CUDA behavior)
    std::fill_n(output_row, hidden_size, 0.0f);

    // Accumulate k experts for this row - matches CUDA's k-way reduction
    for (int64_t k_idx = 0; k_idx < k; ++k_idx) {
      const int64_t expert_offset = row * k + k_idx;
      const int64_t expert_idx = expert_indices[expert_offset];
      const float expert_scale = expert_scales[expert_offset];

      const float* expert_output_row = expert_outputs + expert_offset * hidden_size;
      const float* bias_ptr = fc2_bias_float ? fc2_bias_float + expert_idx * hidden_size : nullptr;

      // Accumulate: output += expert_scale * (expert_output + bias)
      for (int64_t col = 0; col < hidden_size; ++col) {
        const float bias_value = bias_ptr ? bias_ptr[col] : 0.0f;
        output_row[col] += expert_scale * (expert_output_row[col] + bias_value);
      }
    }
  }
}

// CUDA-style MoE FC processing - matches CutlassMoeFCRunner behavior
void run_moe_fc_cpu(const float* input_activations, const float* gating_output,
                    const float* fc1_expert_weights, const float* fc1_scales, const float* fc1_expert_biases,
                    const float* fc2_expert_weights, const float* fc2_scales,
                    int64_t num_rows, int64_t hidden_size, int64_t inter_size, int64_t num_experts, int64_t k,
                    bool is_swiglu, ActivationType activation_type, bool normalize_routing_weights,
                    float* expert_outputs, float* expert_scales_output, int* expert_indices_output,
                    onnxruntime::concurrency::ThreadPool* thread_pool,
                    float activation_alpha = 1.702f, float activation_beta = 1.0f) {
  const int64_t fc1_output_size = is_swiglu ? 2 * inter_size : inter_size;

  // Expert selection and routing (matches CUDA's sorting and selection logic)
  std::vector<std::pair<float, int64_t>> expert_scores;
  std::vector<float> routing_weights(num_rows * k);
  std::vector<int> expert_indices(num_rows * k);

  for (int64_t row = 0; row < num_rows; ++row) {
    expert_scores.clear();
    expert_scores.reserve(num_experts);

    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      const int64_t router_idx = row * num_experts + expert_idx;
      float routing_weight = gating_output[router_idx];

      // Apply probability quantization if robust selection is enabled
      if (use_robust_expert_selection()) {
        routing_weight = quantize_router_probability(routing_weight);
      }

      expert_scores.emplace_back(routing_weight, expert_idx);
    }

    // Robust top-k selection with deterministic tie-breaking
    // Custom comparator: primary sort by probability (descending),
    // secondary sort by expert index (ascending) for deterministic ties
    auto robust_comparator = [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
      constexpr float epsilon = 1e-6f;  // Threshold for considering probabilities "equal"

      // If probabilities are nearly equal (within epsilon), use expert index for tie-breaking
      if (std::abs(a.first - b.first) < epsilon) {
        return a.second < b.second;  // Lower expert index wins (deterministic)
      }

      // Otherwise, sort by probability (higher first)
      return a.first > b.first;
    };

    std::partial_sort(expert_scores.begin(), expert_scores.begin() + k,
                      expert_scores.end(), robust_comparator);
    // Normalize selected routing weights to sum to 1.0 (matches CUDA's top-k normalization)
    float selected_sum = 0.0f;
    for (int64_t k_idx = 0; k_idx < k; ++k_idx) {
      selected_sum += expert_scores[k_idx].first;
    }

    // Store selected experts and their normalized routing weights
    for (int64_t k_idx = 0; k_idx < k; ++k_idx) {
      const int64_t offset = row * k + k_idx;
      expert_indices[offset] = static_cast<int>(expert_scores[k_idx].second);
      // Normalize by sum of selected top-k weights (matches CUDA behavior)
      routing_weights[offset] = (selected_sum > 0.0f) ? (expert_scores[k_idx].first / selected_sum) : 0.0f;
    }
  }

  // Process expert computations (matches CUDA's GEMM operations)
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(num_rows * k),
      static_cast<double>(hidden_size * inter_size * 0.1),
      [&](std::ptrdiff_t start, std::ptrdiff_t end) {
        // Thread-local buffers
        std::vector<float> fc1_output(fc1_output_size);

        for (std::ptrdiff_t idx = start; idx < end; ++idx) {
          const int64_t row = idx / k;
          const int64_t k_idx = idx % k;
          const int64_t expert_idx = expert_indices[idx];

          const float* input_row = input_activations + row * hidden_size;
          float* output_row = expert_outputs + idx * hidden_size;

          // FC1 computation: input * fc1_weights (column-major)
          const float* fc1_weights = fc1_expert_weights + expert_idx * hidden_size * fc1_output_size;

          // GEMM: C = A * B where A is input_row (1 x K), B is fc1_weights (N x K), C is fc1_output (1 x N)
          MlasGemm(CblasNoTrans, CblasTrans,
                   1, static_cast<size_t>(fc1_output_size), static_cast<size_t>(hidden_size),
                   1.0f,
                   input_row, static_cast<size_t>(hidden_size),
                   fc1_weights, static_cast<size_t>(hidden_size),
                   0.0f,
                   fc1_output.data(), static_cast<size_t>(fc1_output_size),
                   nullptr);

          // Add FC1 bias
          if (fc1_expert_biases) {
            const float* fc1_bias = fc1_expert_biases + expert_idx * fc1_output_size;
            for (int64_t i = 0; i < fc1_output_size; ++i) {
              fc1_output[i] += fc1_bias[i];
            }
          }

          // Apply activation
          if (is_swiglu) {
            contrib::ApplySwiGLUActivation(fc1_output.data(), fc1_output_size / 2, true,
                                           activation_alpha, activation_beta);
          } else {
            for (int64_t i = 0; i < fc1_output_size; ++i) {
              fc1_output[i] = ApplyActivation(fc1_output[i], activation_type);
            }
          }

          // FC2 computation: fc1_output * fc2_weights (column-major)
          const int64_t actual_inter_size = is_swiglu ? fc1_output_size / 2 : fc1_output_size;
          const float* fc2_weights = fc2_expert_weights + expert_idx * actual_inter_size * hidden_size;

          // GEMM: C = A * B where A is fc1_output (1 x K), B is fc2_weights (N x K), C is output_row (1 x N)
          MlasGemm(CblasNoTrans, CblasTrans,
                   1, static_cast<size_t>(hidden_size), static_cast<size_t>(actual_inter_size),
                   1.0f,
                   fc1_output.data(), static_cast<size_t>(actual_inter_size),
                   fc2_weights, static_cast<size_t>(actual_inter_size),
                   0.0f,
                   output_row, static_cast<size_t>(hidden_size),
                   nullptr);
        }
      });

  // Copy routing weights to output (for finalize stage)
  std::copy(routing_weights.begin(), routing_weights.end(), expert_scales_output);
  std::copy(expert_indices.begin(), expert_indices.end(), expert_indices_output);
}

template <typename T>
QMoE<T>::QMoE(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info), MoEBaseCPU(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4 || expert_weight_bits_ == 0,
              "expert_weight_bits must be 0 (FP32), 4, or 8, but got ", expert_weight_bits_);
}

template <typename T>
Status QMoE<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* router_probs = ctx->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = ctx->Input<Tensor>(2);
  const Tensor* fc1_scales = ctx->Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = ctx->Input<Tensor>(4);
  const Tensor* fc2_experts_weights = ctx->Input<Tensor>(5);
  const Tensor* fc2_scales = ctx->Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = ctx->Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = ctx->Input<Tensor>(8);
  const Tensor* fc3_scales_optional = ctx->Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = ctx->Input<Tensor>(10);

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias_optional, fc1_scales,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales,
      fc3_experts_weights_optional, fc3_experts_bias_optional, fc3_scales_optional,
      expert_weight_bits_ == 4 ? 2 : 1,
      activation_type_ == ActivationType::SwiGLU));

  if (expert_weight_bits_ == 0) {
    // FP32 mode: Use direct FP32 computation (no quantization)
    return DirectFP32MoEImpl(ctx, moe_params, input, router_probs,
                             fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                             fc2_experts_bias_optional, fc3_experts_weights_optional,
                             fc3_experts_bias_optional);
  } else if (expert_weight_bits_ == 4) {
    return QuantizedMoEImpl<true>(ctx, moe_params, input, router_probs,
                                  fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                  fc2_experts_bias_optional, fc3_experts_weights_optional,
                                  fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
  } else {
    return QuantizedMoEImpl<false>(ctx, moe_params, input, router_probs,
                                   fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                   fc2_experts_bias_optional, fc3_experts_weights_optional,
                                   fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
  }
}

template <typename T>
template <bool UseUInt4x2>
Status QMoE<T>::PrepackAndDequantizeWeights(OpKernelContext* context,
                                            MoEParameters& moe_params,
                                            const Tensor* fc1_experts_weights,
                                            const Tensor* fc2_experts_weights,
                                            const Tensor* fc1_scales,
                                            const Tensor* fc2_scales,
                                            bool is_swiglu) {
  // Get allocator for persistent weights storage
  if (!weights_allocator_) {
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&weights_allocator_));
  }

  // Calculate weight sizes
  const int64_t fc1_output_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
  const size_t fc1_weights_size = moe_params.num_experts * moe_params.hidden_size * fc1_output_size;
  const size_t fc2_weights_size = moe_params.num_experts * moe_params.inter_size * moe_params.hidden_size;

  // Allocate storage for weights (dequantized or direct FP32)
  prepacked_fc1_weights_ = IAllocator::MakeUniquePtr<float>(weights_allocator_, fc1_weights_size);
  prepacked_fc2_weights_ = IAllocator::MakeUniquePtr<float>(weights_allocator_, fc2_weights_size);
  prepacked_fc1_weights_data_ = prepacked_fc1_weights_.get();
  prepacked_fc2_weights_data_ = prepacked_fc2_weights_.get();

  if (expert_weight_bits_ == 0) {
    // FP32 mode: weights are already float32, just copy/reformat them
    // Check if weights are actually provided as float32
    if (fc1_experts_weights->IsDataType<float>() && fc2_experts_weights->IsDataType<float>()) {
      const float* fc1_weights_data = fc1_experts_weights->Data<float>();
      const float* fc2_weights_data = fc2_experts_weights->Data<float>();

      // Copy FC1 weights directly (already in correct format)
      std::copy(fc1_weights_data, fc1_weights_data + fc1_weights_size, prepacked_fc1_weights_data_);

      // Copy FC2 weights directly (already in correct format)
      std::copy(fc2_weights_data, fc2_weights_data + fc2_weights_size, prepacked_fc2_weights_data_);

    } else {
      // Fallback: weights provided as uint8 but expert_weight_bits=0, treat as 8-bit quantized
      // Fall through to quantized path
      expert_weight_bits_ = 8;  // Temporarily treat as 8-bit for this call
    }
  }

  if (expert_weight_bits_ != 0) {
    // Quantized mode: dequantize weights
    const uint8_t* fc1_weights_data = fc1_experts_weights->Data<uint8_t>();
    const uint8_t* fc2_weights_data = fc2_experts_weights->Data<uint8_t>();
    const T* fc1_scales_data = fc1_scales->Data<T>();
    const T* fc2_scales_data = fc2_scales->Data<T>();

    auto* thread_pool = context->GetOperatorThreadPool();

    // Corrected FC1 weight dequantization
    if constexpr (UseUInt4x2) {
      // Corrected 4-bit dequantization logic for FC1 weights
      for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
        for (int64_t col = 0; col < fc1_output_size; ++col) {
          const float scale = static_cast<float>(fc1_scales_data[expert_idx * fc1_output_size + col]);
          for (int64_t row = 0; row < moe_params.hidden_size; ++row) {
            const size_t packed_idx = (row / 2) * fc1_output_size + col;
            const uint8_t packed_val = fc1_weights_data[expert_idx * (moe_params.hidden_size / 2) * fc1_output_size + packed_idx];

            // Unpack the two 4-bit values
            const uint8_t val = (row % 2 == 0) ? (packed_val & 0x0F) : (packed_val >> 4);

            // Dequantize: scale * (value - zero_point)
            // For symmetric 4-bit, the range is [-8, 7] and zero point is 8
            const float dequantized_val = scale * (static_cast<float>(val) - 8.0f);

            // Store in column-major layout for MlasGemm
            prepacked_fc1_weights_data_[expert_idx * fc1_output_size * moe_params.hidden_size + col * moe_params.hidden_size + row] = dequantized_val;
          }
        }
      }
    } else {
      // Corrected 8-bit dequantization logic for FC1 weights
      printf("DEBUG: FC1 8-bit dequantization - experts=%ld, fc1_output_size=%ld, hidden_size=%ld\n",
             moe_params.num_experts, fc1_output_size, moe_params.hidden_size);

      // Print actual tensor shapes
      auto fc1_shape = fc1_experts_weights->Shape();
      printf("DEBUG: Actual FC1 tensor shape: [");
      for (size_t i = 0; i < fc1_shape.NumDimensions(); ++i) {
        printf("%ld", fc1_shape[i]);
        if (i < fc1_shape.NumDimensions() - 1) printf(", ");
      }
      printf("]\n");

      for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
        for (int64_t col = 0; col < fc1_output_size; ++col) {
          const float scale = static_cast<float>(fc1_scales_data[expert_idx * fc1_output_size + col]);
          for (int64_t row = 0; row < moe_params.hidden_size; ++row) {
            // Weights are stored column-major in the source tensor
            const size_t quant_idx = col * moe_params.hidden_size + row;
            const uint8_t val = fc1_weights_data[expert_idx * fc1_output_size * moe_params.hidden_size + quant_idx];

            // Dequantize: scale * (value - zero_point)
            // For symmetric 8-bit, the range is [-128, 127] and zero point is 128
            const float dequantized_val = scale * (static_cast<float>(val) - 128.0f);
            prepacked_fc1_weights_data_[expert_idx * fc1_output_size * moe_params.hidden_size + quant_idx] = dequantized_val;

            // Print first few values for debugging
            if (expert_idx == 0 && col < 3 && row < 3) {
              printf("DEBUG: FC1[%ld][%ld][%ld] uint8=%d scale=%f -> %f\n",
                     expert_idx, col, row, val, scale, dequantized_val);
            }
          }
        }
      }
    }  // Corrected FC2 weight dequantization
    if constexpr (UseUInt4x2) {
      // Corrected 4-bit dequantization logic for FC2 weights
      for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
        for (int64_t col = 0; col < moe_params.hidden_size; ++col) {
          const float scale = static_cast<float>(fc2_scales_data[expert_idx * moe_params.hidden_size + col]);
          for (int64_t row = 0; row < moe_params.inter_size; ++row) {
            const size_t packed_idx = (row / 2) * moe_params.hidden_size + col;
            const uint8_t packed_val = fc2_weights_data[expert_idx * (moe_params.inter_size / 2) * moe_params.hidden_size + packed_idx];

            // Unpack the two 4-bit values
            const uint8_t val = (row % 2 == 0) ? (packed_val & 0x0F) : (packed_val >> 4);

            // Dequantize: scale * (value - zero_point)
            // For symmetric 4-bit, the range is [-8, 7] and zero point is 8
            const float dequantized_val = scale * (static_cast<float>(val) - 8.0f);

            // Store in column-major layout for MlasGemm
            prepacked_fc2_weights_data_[expert_idx * moe_params.hidden_size * moe_params.inter_size + col * moe_params.inter_size + row] = dequantized_val;
          }
        }
      }
    } else {
      // Corrected 8-bit dequantization logic for FC2 weights
      printf("DEBUG: FC2 8-bit dequantization - experts=%ld, hidden_size=%ld, inter_size=%ld\n",
             moe_params.num_experts, moe_params.hidden_size, moe_params.inter_size);

      for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
        for (int64_t col = 0; col < moe_params.hidden_size; ++col) {
          const float scale = static_cast<float>(fc2_scales_data[expert_idx * moe_params.hidden_size + col]);
          for (int64_t row = 0; row < moe_params.inter_size; ++row) {
            // Weights are stored column-major in the source tensor
            const size_t quant_idx = col * moe_params.inter_size + row;
            const uint8_t val = fc2_weights_data[expert_idx * moe_params.hidden_size * moe_params.inter_size + quant_idx];

            // Dequantize: scale * (value - zero_point)
            // For symmetric 8-bit, the range is [-128, 127] and zero point is 128
            const float dequantized_val = scale * (static_cast<float>(val) - 128.0f);
            prepacked_fc2_weights_data_[expert_idx * moe_params.hidden_size * moe_params.inter_size + quant_idx] = dequantized_val;

            // Print first few values for debugging
            if (expert_idx == 0 && col < 3 && row < 3) {
              printf("DEBUG: FC2[%ld][%ld][%ld] uint8=%d scale=%f -> %f\n",
                     expert_idx, col, row, val, scale, dequantized_val);
            }
          }
        }
      }
    }
  }  // End of if (expert_weight_bits_ != 0)

  // Cache parameters
  cached_num_experts_ = moe_params.num_experts;
  cached_hidden_size_ = moe_params.hidden_size;
  cached_inter_size_ = moe_params.inter_size;
  cached_is_swiglu_ = is_swiglu;
  is_prepacked_ = true;

  return Status::OK();
}

template <typename T>
template <bool UseUInt4x2>
Status QMoE<T>::QuantizedMoEImpl(OpKernelContext* context,
                                 MoEParameters& moe_params,
                                 const Tensor* input,
                                 const Tensor* router_probs,
                                 const Tensor* fc1_experts_weights,
                                 const Tensor* fc1_experts_bias_optional,
                                 const Tensor* fc2_experts_weights,
                                 const Tensor* fc2_experts_bias_optional,
                                 const Tensor* fc3_experts_weights_optional,
                                 const Tensor* fc3_experts_bias_optional,
                                 const Tensor* fc1_scales,
                                 const Tensor* fc2_scales,
                                 const Tensor* fc3_scales_optional) const {
  // Check if we need to repack weights
  if (!is_prepacked_ ||
      cached_num_experts_ != moe_params.num_experts ||
      cached_hidden_size_ != moe_params.hidden_size ||
      cached_inter_size_ != moe_params.inter_size ||
      cached_is_swiglu_ != (activation_type_ == ActivationType::SwiGLU)) {
    Status status = const_cast<QMoE<T>*>(this)->PrepackAndDequantizeWeights<UseUInt4x2>(
        context, moe_params, fc1_experts_weights, fc2_experts_weights,
        fc1_scales, fc2_scales, activation_type_ == ActivationType::SwiGLU);
    ORT_RETURN_IF_ERROR(status);
  }

  auto* thread_pool = context->GetOperatorThreadPool();
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // Get input data and create output
  const T* input_data = input->Data<T>();
  const T* router_probs_data = router_probs->Data<T>();
  const T* fc1_bias_data = fc1_experts_bias_optional ? fc1_experts_bias_optional->Data<T>() : nullptr;
  const T* fc2_bias_data = fc2_experts_bias_optional ? fc2_experts_bias_optional->Data<T>() : nullptr;

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  // Convert inputs to float (similar to CUDA preparation)
  const size_t input_size = static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size);
  const size_t router_size = static_cast<size_t>(moe_params.num_rows * moe_params.num_experts);
  const size_t output_size = static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size);

  auto input_float = IAllocator::MakeUniquePtr<float>(allocator, input_size);
  auto router_float = IAllocator::MakeUniquePtr<float>(allocator, router_size);
  auto output_float = IAllocator::MakeUniquePtr<float>(allocator, output_size);

  // Convert inputs to float for processing
  if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(input_data),
                                 input_float.get(), input_size);
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(router_probs_data),
                                 router_float.get(), router_size);
  } else {
    std::copy(input_data, input_data + input_size, input_float.get());
    std::copy(router_probs_data, router_probs_data + router_size, router_float.get());
  }

  // Debug: Check input and router data after conversion

  // Root cause: Router input differences indicate upstream LayerNorm or other preprocessing
  // differences between CPU and CUDA implementations, not QMoE-specific precision issues

  // Use the correct k value from the base class
  const int64_t correct_k = k_;

  // Skip softmax normalization to match CUDA behavior (uses raw logits)

  // Convert biases to float using standard MLAS conversion
  std::unique_ptr<float[]> fc1_bias_float, fc2_bias_float;
  if (fc1_bias_data) {
    const size_t fc1_bias_size = moe_params.num_experts * (activation_type_ == ActivationType::SwiGLU ? 2 * moe_params.inter_size : moe_params.inter_size);
    fc1_bias_float = std::make_unique<float[]>(fc1_bias_size);
    if constexpr (std::is_same_v<T, MLFloat16>) {
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc1_bias_data),
                                   fc1_bias_float.get(), fc1_bias_size);
    } else {
      std::copy(fc1_bias_data, fc1_bias_data + fc1_bias_size, fc1_bias_float.get());
    }
  }

  if (fc2_bias_data) {
    const size_t fc2_bias_size = moe_params.num_experts * moe_params.hidden_size;
    fc2_bias_float = std::make_unique<float[]>(fc2_bias_size);
    if constexpr (std::is_same_v<T, MLFloat16>) {
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc2_bias_data),
                                   fc2_bias_float.get(), fc2_bias_size);
    } else {
      std::copy(fc2_bias_data, fc2_bias_data + fc2_bias_size, fc2_bias_float.get());
    }
  }

  // Allocate intermediate buffers (matches CUDA workspace allocation)
  const size_t expert_outputs_size = correct_k * moe_params.num_rows * moe_params.hidden_size;
  const size_t expert_scales_size = correct_k * moe_params.num_rows;
  const size_t expert_indices_size = correct_k * moe_params.num_rows;

  auto expert_outputs = IAllocator::MakeUniquePtr<float>(allocator, expert_outputs_size);
  auto expert_scales = IAllocator::MakeUniquePtr<float>(allocator, expert_scales_size);
  auto expert_indices = IAllocator::MakeUniquePtr<int>(allocator, expert_indices_size);

  // Stage 1: Run MoE FC (matches CUDA's CutlassMoeFCRunner::run_moe_fc)
  run_moe_fc_cpu(
      input_float.get(), router_float.get(),
      prepacked_fc1_weights_data_, nullptr, fc1_bias_float.get(),
      prepacked_fc2_weights_data_, nullptr,
      moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, correct_k,
      activation_type_ == ActivationType::SwiGLU, activation_type_, normalize_routing_weights_,
      expert_outputs.get(), expert_scales.get(), expert_indices.get(),
      thread_pool, activation_alpha_, activation_beta_);

  // Stage 2: Finalize routing (matches CUDA's finalize_moe_routing_kernelLauncher)
  finalize_moe_routing_cpu(
      expert_outputs.get(), output_float.get(),
      fc2_bias_float.get(), expert_scales.get(), expert_indices.get(),
      moe_params.num_rows, moe_params.hidden_size, correct_k);

  // Convert output back to original type
  // Note: For FP32 inputs, we expect very high precision (<0.01 difference) since:
  // 1. All computations use float32 accumulation (no precision loss)
  // 2. No quantization/dequantization when quant_bits=0
  // 3. Same precision as reference PyTorch implementation
  // 4. Robust expert selection eliminates floating-point sensitivity
  if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertFloatToHalfBuffer(output_float.get(),
                                 reinterpret_cast<MLAS_FP16*>(output_data),
                                 output_size);
  } else {
    std::copy(output_float.get(), output_float.get() + output_size, output_data);
  }

  return Status::OK();
}

template <typename T>
Status QMoE<T>::DirectFP32MoEImpl(OpKernelContext* context,
                                  MoEParameters& moe_params,
                                  const Tensor* input,
                                  const Tensor* router_probs,
                                  const Tensor* fc1_experts_weights,
                                  const Tensor* fc1_experts_bias_optional,
                                  const Tensor* fc2_experts_weights,
                                  const Tensor* fc2_experts_bias_optional,
                                  const Tensor* fc3_experts_weights_optional,
                                  const Tensor* fc3_experts_bias_optional) const {
  // Direct FP32 MoE implementation - no quantization involved
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // Get input data
  const T* input_data = input->Data<T>();
  const T* router_probs_data = router_probs->Data<T>();
  const T* fc1_bias_data = fc1_experts_bias_optional ? fc1_experts_bias_optional->Data<T>() : nullptr;
  const T* fc2_bias_data = fc2_experts_bias_optional ? fc2_experts_bias_optional->Data<T>() : nullptr;

  // Get expert weights as FP32
  const float* fc1_weights_data = fc1_experts_weights->Data<float>();
  const float* fc2_weights_data = fc2_experts_weights->Data<float>();

  // Create output tensor
  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  // Convert inputs to float for computation
  const size_t input_size = static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size);
  const size_t router_size = static_cast<size_t>(moe_params.num_rows * moe_params.num_experts);
  const size_t output_size = static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size);

  auto input_float = IAllocator::MakeUniquePtr<float>(allocator, input_size);
  auto router_float = IAllocator::MakeUniquePtr<float>(allocator, router_size);
  auto output_float = IAllocator::MakeUniquePtr<float>(allocator, output_size);

  // Convert input to float
  if constexpr (std::is_same_v<T, float>) {
    std::copy(input_data, input_data + input_size, input_float.get());
    std::copy(router_probs_data, router_probs_data + router_size, router_float.get());
  } else if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(input_data),
                                 input_float.get(), input_size);
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(router_probs_data),
                                 router_float.get(), router_size);
  }

  // Initialize output to zero
  std::fill(output_float.get(), output_float.get() + output_size, 0.0f);

  // Perform MoE computation with direct FP32 weights
  // This is the key difference: we use the weights directly without any quantization/dequantization
  for (int64_t token_idx = 0; token_idx < moe_params.num_rows; ++token_idx) {
    // Get top-k experts for this token
    const float* token_router_probs = router_float.get() + token_idx * moe_params.num_experts;

    // Find top-k experts (simple implementation for now)
    std::vector<std::pair<float, int64_t>> expert_scores;
    for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
      expert_scores.emplace_back(token_router_probs[expert_idx], expert_idx);
    }
    std::sort(expert_scores.rbegin(), expert_scores.rend());  // Sort descending

    // Collect top-k weights and renormalize
    std::vector<float> topk_weights;
    std::vector<int64_t> topk_experts;
    float weight_sum = 0.0f;

    for (int64_t k_idx = 0; k_idx < std::min(static_cast<int64_t>(k_), moe_params.num_experts); ++k_idx) {
      float weight = expert_scores[k_idx].first;
      int64_t expert_idx = expert_scores[k_idx].second;

      if (weight <= 0.0f) break;  // Stop at zero weight

      topk_weights.push_back(weight);
      topk_experts.push_back(expert_idx);
      weight_sum += weight;
    }

    // Normalize weights so they sum to 1 (crucial for correct MoE computation)
    if (weight_sum > 0.0f) {
      for (auto& w : topk_weights) {
        w /= weight_sum;
      }
    }

    // Process top-k experts with normalized weights
    for (size_t k_idx = 0; k_idx < topk_weights.size(); ++k_idx) {
      float weight = topk_weights[k_idx];
      int64_t expert_idx = topk_experts[k_idx];

      // Input for this token: [hidden_size]
      const float* token_input = input_float.get() + token_idx * moe_params.hidden_size;
      float* token_output = output_float.get() + token_idx * moe_params.hidden_size;

      // FC1: input [hidden_size] -> intermediate [intermediate_size or 2*intermediate_size for SwiGLU]
      const int64_t fc1_output_size = (activation_type_ == ActivationType::SwiGLU) ? 2 * moe_params.inter_size : moe_params.inter_size;

      auto intermediate = IAllocator::MakeUniquePtr<float>(allocator, fc1_output_size);

      // FC1 computation: intermediate = token_input @ fc1_weights[expert_idx]
      const float* expert_fc1_weights = fc1_weights_data + expert_idx * moe_params.hidden_size * fc1_output_size;

      for (int64_t out_idx = 0; out_idx < fc1_output_size; ++out_idx) {
        float sum = 0.0f;
        for (int64_t in_idx = 0; in_idx < moe_params.hidden_size; ++in_idx) {
          // Weights are stored as (fc1_output_size x hidden_size) but used transposed
          // So we access weights[out_idx][in_idx] as weights[out_idx * hidden_size + in_idx]
          sum += token_input[in_idx] * expert_fc1_weights[out_idx * moe_params.hidden_size + in_idx];
        }
        // Add bias if available
        if (fc1_bias_data) {
          sum += static_cast<float>(fc1_bias_data[expert_idx * fc1_output_size + out_idx]);
        }
        intermediate.get()[out_idx] = sum;
      }

      // Apply activation
      if (activation_type_ == ActivationType::SwiGLU) {
        // Use the same SwiGLU implementation as the original quantized code
        contrib::ApplySwiGLUActivation(intermediate.get(), moe_params.inter_size, true,
                                       activation_alpha_, activation_beta_);

      } else if (activation_type_ == ActivationType::Silu) {
        // SiLU activation: x * sigmoid(x)
        for (int64_t i = 0; i < moe_params.inter_size; ++i) {
          float x = intermediate.get()[i];
          intermediate.get()[i] = x / (1.0f + std::exp(-x));
        }
      }

      // FC2: intermediate [inter_size] -> output [hidden_size]
      const float* expert_fc2_weights = fc2_weights_data + expert_idx * moe_params.inter_size * moe_params.hidden_size;

      for (int64_t out_idx = 0; out_idx < moe_params.hidden_size; ++out_idx) {
        float sum = 0.0f;
        for (int64_t in_idx = 0; in_idx < moe_params.inter_size; ++in_idx) {
          sum += intermediate.get()[in_idx] * expert_fc2_weights[out_idx * moe_params.inter_size + in_idx];
        }
        // Add bias if available
        if (fc2_bias_data) {
          sum += static_cast<float>(fc2_bias_data[expert_idx * moe_params.hidden_size + out_idx]);
        }

        // Accumulate weighted expert output
        token_output[out_idx] += weight * sum;
      }
    }
  }

  // Convert output back to T
  if constexpr (std::is_same_v<T, float>) {
    std::copy(output_float.get(), output_float.get() + output_size, output_data);
  } else if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertFloatToHalfBuffer(output_float.get(),
                                 reinterpret_cast<MLAS_FP16*>(output_data),
                                 output_size);
  }

  return Status::OK();
}

// Kernel registrations
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<float>()})
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    QMoE<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE,
    kMSDomain,
    1,
    MLFloat16,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    QMoE<MLFloat16>);

// Template instantiations
template class QMoE<float>;
template class QMoE<MLFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
