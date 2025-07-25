// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/moe_quantization_cpu.h"
#include "core/framework/allocator.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/platform/threadpool.h"

using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL()                                                                  \
  ONNX_OPERATOR_KERNEL_EX(QMoE, kMSDomain, 1, kCpuExecutionProvider,                       \
                          (*KernelDefBuilder::Create())                                    \
                              .MayInplace(0, 0)                                            \
                              .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16>()) \
                              .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>())  \
                              .TypeConstraint("T2", BuildKernelDefConstraints<float>()),   \
                          QMoE);

REGISTER_KERNEL();

// QMoE CPU kernel registration is handled in cpu_contrib_kernels.cc
// Implementation matches CUDA QMoE kernel type support (MLFloat16 only)

QMoE::QMoE(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info), MoEBaseCPU(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
              "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);
}

Status QMoE::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_scales = context->Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(4);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(5);
  const Tensor* fc2_scales = context->Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = context->Input<Tensor>(8);
  const Tensor* fc3_scales_optional = context->Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = context->Input<Tensor>(10);

  MoEQuantType quant_type = expert_weight_bits_ == 4 ? MoEQuantType::UINT4 : MoEQuantType::UINT8;
  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(CheckInputs(moe_params, quant_type, input, router_probs, fc1_experts_weights,
                                  fc1_experts_bias_optional, fc2_experts_weights, fc2_experts_bias_optional,
                                  fc3_experts_weights_optional, fc3_experts_bias_optional));
  ORT_RETURN_IF_ERROR(CheckInputScales(fc1_scales, fc2_scales, fc3_scales_optional, moe_params.num_experts,
                                       moe_params.hidden_size, moe_params.inter_size));

  if (quant_type == MoEQuantType::UINT4) {
    return QuantizedMoEImpl<true>(context, moe_params, input, router_probs,
                                  fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                  fc2_experts_bias_optional, fc3_experts_weights_optional,
                                  fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
  } else {
    return QuantizedMoEImpl<false>(context, moe_params, input, router_probs,
                                   fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                   fc2_experts_bias_optional, fc3_experts_weights_optional,
                                   fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
  }
}

template <bool UseUInt4x2>
Status QMoE::QuantizedMoEImpl(OpKernelContext* context,
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
  // FC3 (gating) check - throw error if present (CPU doesn't support FC3)
  if (fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "FC3 gating is not yet implemented for CPU quantized MoE. Please use the CUDA execution provider for gated experts or disable FC3 gating.");
  }

  // Get thread pool
  auto* thread_pool = context->GetOperatorThreadPool();

  // Get input data pointers
  const MLFloat16* input_data = input->Data<MLFloat16>();
  const MLFloat16* router_probs_data = router_probs->Data<MLFloat16>();
  const uint8_t* fc1_weights_data = fc1_experts_weights->Data<uint8_t>();
  const uint8_t* fc2_weights_data = fc2_experts_weights->Data<uint8_t>();
  const float* fc1_scales_data = fc1_scales->Data<float>();
  const float* fc2_scales_data = fc2_scales->Data<float>();

  const MLFloat16* fc1_bias_data = fc1_experts_bias_optional ? fc1_experts_bias_optional->Data<MLFloat16>() : nullptr;
  const MLFloat16* fc2_bias_data = fc2_experts_bias_optional ? fc2_experts_bias_optional->Data<MLFloat16>() : nullptr;

  // Create output tensor
  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();

  // Initialize output to zero
  std::fill(output_data, output_data + moe_params.num_rows * moe_params.hidden_size, MLFloat16{});

  // Allocate temporary buffers
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // Calculate number of threads to use for parallelization
  const int64_t num_threads = std::min<int64_t>(
      static_cast<int64_t>(concurrency::ThreadPool::DegreeOfParallelism(thread_pool)),
      moe_params.num_rows);

  // Allocate thread-local buffers
  auto thread_fc1_buffers = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_threads * moe_params.inter_size));
  auto thread_fc2_buffers = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_threads * moe_params.hidden_size));
  auto thread_results = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_threads * moe_params.num_rows * moe_params.hidden_size));

  // Initialize thread results to zero
  std::fill(thread_results.get(),
            thread_results.get() + static_cast<size_t>(num_threads * moe_params.num_rows * moe_params.hidden_size), 0.0f);

  // Helper function to convert MLFloat16 to float
  auto ToFloat = [](MLFloat16 value) { return static_cast<float>(value); };
  auto FromFloat = [](float value) { return MLFloat16(value); };

  // Helper function to apply activation
  auto ApplyActivation = [](float x, ActivationType activation_type) {
    switch (activation_type) {
      case ActivationType::Relu:
        return std::max(0.0f, x);
      case ActivationType::Gelu:
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
      case ActivationType::Silu:
        // SiLU: x * sigmoid(x)
        return x * (1.0f / (1.0f + std::exp(-x)));
      case ActivationType::Identity:
        return x;
      default:
        return x;  // Default to identity
    }
  };

  if constexpr (UseUInt4x2) {
    // UInt4x2 implementation - pre-dequantize weights and use optimized GEMM-like operations

    // Pre-dequantize all expert weights once (shared across all threads)
    auto dequant_fc1_weights = IAllocator::MakeUniquePtr<float>(allocator,
                                                                static_cast<size_t>(moe_params.num_experts * moe_params.hidden_size * moe_params.inter_size));
    auto dequant_fc2_weights = IAllocator::MakeUniquePtr<float>(allocator,
                                                                static_cast<size_t>(moe_params.num_experts * moe_params.inter_size * moe_params.hidden_size));

    // Dequantize FC1 weights for all experts (Int4 unpacking)
    for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
      const uint8_t* fc1_expert_weights = fc1_weights_data + expert_idx * moe_params.hidden_size * moe_params.inter_size / 2;
      const float* fc1_expert_scales = fc1_scales_data + expert_idx * moe_params.inter_size;
      float* dequant_fc1_expert = dequant_fc1_weights.get() + expert_idx * moe_params.hidden_size * moe_params.inter_size;

      for (int64_t out_col = 0; out_col < moe_params.inter_size; ++out_col) {
        for (int64_t in_col = 0; in_col < moe_params.hidden_size; ++in_col) {
          // For Int4, two values are packed in each uint8
          size_t linear_idx = static_cast<size_t>(out_col * moe_params.hidden_size + in_col);
          size_t packed_idx = linear_idx / 2;
          uint8_t packed_value = fc1_expert_weights[packed_idx];

          uint8_t quantized_weight;
          if (linear_idx % 2 == 0) {
            quantized_weight = packed_value & 0x0F;  // Lower 4 bits
          } else {
            quantized_weight = (packed_value >> 4) & 0x0F;  // Upper 4 bits
          }

          // Dequantize from 4-bit to float (symmetric quantization, zero point = 8)
          dequant_fc1_expert[linear_idx] = (static_cast<float>(quantized_weight) - 8.0f) * fc1_expert_scales[out_col];
        }
      }
    }

    // Dequantize FC2 weights for all experts (Int4 unpacking)
    for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
      const uint8_t* fc2_expert_weights = fc2_weights_data + expert_idx * moe_params.inter_size * moe_params.hidden_size / 2;
      const float* fc2_expert_scales = fc2_scales_data + expert_idx * moe_params.hidden_size;
      float* dequant_fc2_expert = dequant_fc2_weights.get() + expert_idx * moe_params.inter_size * moe_params.hidden_size;

      for (int64_t out_col = 0; out_col < moe_params.hidden_size; ++out_col) {
        for (int64_t in_col = 0; in_col < moe_params.inter_size; ++in_col) {
          // For Int4, two values are packed in each uint8
          size_t linear_idx = static_cast<size_t>(out_col * moe_params.inter_size + in_col);
          size_t packed_idx = linear_idx / 2;
          uint8_t packed_value = fc2_expert_weights[packed_idx];

          uint8_t quantized_weight;
          if (linear_idx % 2 == 0) {
            quantized_weight = packed_value & 0x0F;  // Lower 4 bits
          } else {
            quantized_weight = (packed_value >> 4) & 0x0F;  // Upper 4 bits
          }

          // Dequantize from 4-bit to float (symmetric quantization, zero point = 8)
          dequant_fc2_expert[linear_idx] = (static_cast<float>(quantized_weight) - 8.0f) * fc2_expert_scales[out_col];
        }
      }
    }

    auto process_token_range = [&](ptrdiff_t start_token, ptrdiff_t end_token) {
      const int64_t thread_id = start_token / ((moe_params.num_rows + num_threads - 1) / num_threads);
      float* thread_fc1_output = thread_fc1_buffers.get() + thread_id * moe_params.inter_size;
      float* thread_fc2_output = thread_fc2_buffers.get() + thread_id * moe_params.hidden_size;
      float* thread_local_results = thread_results.get() + thread_id * moe_params.num_rows * moe_params.hidden_size;

      // Process each token in this thread's range
      for (int64_t token_idx = start_token; token_idx < end_token; ++token_idx) {
        const MLFloat16* token_input_typed = input_data + token_idx * moe_params.hidden_size;

        // Convert input from MLFloat16 to float for computation
        std::vector<float> token_input_float(moe_params.hidden_size);
        for (int64_t i = 0; i < moe_params.hidden_size; ++i) {
          token_input_float[static_cast<size_t>(i)] = ToFloat(token_input_typed[i]);
        }
        const float* token_input = token_input_float.data();

        float* token_result = thread_local_results + token_idx * moe_params.hidden_size;

        // Process all experts for this token
        for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
          float routing_weight = ToFloat(router_probs_data[token_idx * moe_params.num_experts + expert_idx]);
          if (routing_weight <= 1e-6f) continue;  // Skip experts with negligible routing weight

          // FC1: input -> intermediate using pre-dequantized weights + MLAS SGEMM
          const float* fc1_expert_weights = dequant_fc1_weights.get() + expert_idx * moe_params.hidden_size * moe_params.inter_size;
          const MLFloat16* fc1_expert_bias_typed = fc1_bias_data ? fc1_bias_data + expert_idx * moe_params.inter_size : nullptr;

          // Use MLAS SGEMM for FC1: input [1 x hidden_size] * weights [hidden_size x inter_size] = output [1 x inter_size]
          MLAS_SGEMM_DATA_PARAMS fc1_params;
          fc1_params.A = token_input;
          fc1_params.lda = moe_params.hidden_size;
          fc1_params.B = fc1_expert_weights;
          fc1_params.ldb = moe_params.hidden_size;
          fc1_params.C = thread_fc1_output;
          fc1_params.ldc = moe_params.inter_size;
          fc1_params.alpha = 1.0f;
          fc1_params.beta = 0.0f;

          MlasGemm(CblasNoTrans, CblasNoTrans, 1, static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(moe_params.hidden_size), fc1_params, nullptr);

          // Add bias and apply activation
          for (int64_t i = 0; i < moe_params.inter_size; ++i) {
            if (fc1_expert_bias_typed) {
              thread_fc1_output[i] += ToFloat(fc1_expert_bias_typed[i]);
            }
            thread_fc1_output[i] = ApplyActivation(thread_fc1_output[i], activation_type_);
          }

          // FC2: intermediate -> output using pre-dequantized weights + MLAS SGEMM
          const float* fc2_expert_weights = dequant_fc2_weights.get() + expert_idx * moe_params.inter_size * moe_params.hidden_size;
          const MLFloat16* fc2_expert_bias_typed = fc2_bias_data ? fc2_bias_data + expert_idx * moe_params.hidden_size : nullptr;

          // Use MLAS SGEMM for FC2: intermediate [1 x inter_size] * weights [inter_size x hidden_size] = output [1 x hidden_size]
          MLAS_SGEMM_DATA_PARAMS fc2_params;
          fc2_params.A = thread_fc1_output;
          fc2_params.lda = moe_params.inter_size;
          fc2_params.B = fc2_expert_weights;
          fc2_params.ldb = moe_params.inter_size;
          fc2_params.C = thread_fc2_output;
          fc2_params.ldc = moe_params.hidden_size;
          fc2_params.alpha = 1.0f;
          fc2_params.beta = 0.0f;

          MlasGemm(CblasNoTrans, CblasNoTrans, 1, static_cast<size_t>(moe_params.hidden_size), static_cast<size_t>(moe_params.inter_size), fc2_params, nullptr);

          // Add bias, apply routing weight, and accumulate to final result
          for (int64_t i = 0; i < moe_params.hidden_size; ++i) {
            if (fc2_expert_bias_typed) {
              thread_fc2_output[i] += ToFloat(fc2_expert_bias_typed[i]);
            }
            token_result[i] += routing_weight * thread_fc2_output[i];
          }
        }
      }
    };  // Execute token processing in parallel across threads
    concurrency::ThreadPool::TryParallelFor(thread_pool, moe_params.num_rows,
                                            static_cast<double>(std::max<int64_t>(1, moe_params.num_rows / num_threads)),
                                            process_token_range);
  } else {
    // UInt8 implementation with pre-dequantized weights and MLAS SGEMM

    // Pre-dequantize all expert weights once (shared across all threads)
    auto dequant_fc1_weights = IAllocator::MakeUniquePtr<float>(allocator,
                                                                static_cast<size_t>(moe_params.num_experts * moe_params.hidden_size * moe_params.inter_size));
    auto dequant_fc2_weights = IAllocator::MakeUniquePtr<float>(allocator,
                                                                static_cast<size_t>(moe_params.num_experts * moe_params.inter_size * moe_params.hidden_size));

    // Dequantize FC1 weights for all experts
    for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
      const uint8_t* fc1_expert_weights = fc1_weights_data + expert_idx * moe_params.hidden_size * moe_params.inter_size;
      const float* fc1_expert_scales = fc1_scales_data + expert_idx * moe_params.inter_size;
      float* dequant_fc1_expert = dequant_fc1_weights.get() + expert_idx * moe_params.hidden_size * moe_params.inter_size;

      for (int64_t out_col = 0; out_col < moe_params.inter_size; ++out_col) {
        for (int64_t in_col = 0; in_col < moe_params.hidden_size; ++in_col) {
          size_t weight_idx = static_cast<size_t>(out_col * moe_params.hidden_size + in_col);
          uint8_t quantized_weight = fc1_expert_weights[weight_idx];
          // Symmetric quantization with zero point = 128
          dequant_fc1_expert[weight_idx] = (static_cast<float>(quantized_weight) - 128.0f) * fc1_expert_scales[out_col];
        }
      }
    }

    // Dequantize FC2 weights for all experts
    for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
      const uint8_t* fc2_expert_weights = fc2_weights_data + expert_idx * moe_params.inter_size * moe_params.hidden_size;
      const float* fc2_expert_scales = fc2_scales_data + expert_idx * moe_params.hidden_size;
      float* dequant_fc2_expert = dequant_fc2_weights.get() + expert_idx * moe_params.inter_size * moe_params.hidden_size;

      for (int64_t out_col = 0; out_col < moe_params.hidden_size; ++out_col) {
        for (int64_t in_col = 0; in_col < moe_params.inter_size; ++in_col) {
          size_t weight_idx = static_cast<size_t>(out_col * moe_params.inter_size + in_col);
          uint8_t quantized_weight = fc2_expert_weights[weight_idx];
          // Symmetric quantization with zero point = 128
          dequant_fc2_expert[weight_idx] = (static_cast<float>(quantized_weight) - 128.0f) * fc2_expert_scales[out_col];
        }
      }
    }

    auto process_token_range = [&](ptrdiff_t start_token, ptrdiff_t end_token) {
      const int64_t thread_id = start_token / ((moe_params.num_rows + num_threads - 1) / num_threads);
      float* thread_fc1_output = thread_fc1_buffers.get() + thread_id * moe_params.inter_size;
      float* thread_fc2_output = thread_fc2_buffers.get() + thread_id * moe_params.hidden_size;
      float* thread_local_results = thread_results.get() + thread_id * moe_params.num_rows * moe_params.hidden_size;

      // Process each token in this thread's range
      for (int64_t token_idx = start_token; token_idx < end_token; ++token_idx) {
        const MLFloat16* token_input_typed = input_data + token_idx * moe_params.hidden_size;

        // Convert input from MLFloat16 to float for MLAS computation
        std::vector<float> token_input_float(moe_params.hidden_size);
        for (int64_t i = 0; i < moe_params.hidden_size; ++i) {
          token_input_float[static_cast<size_t>(i)] = ToFloat(token_input_typed[i]);
        }
        const float* token_input = token_input_float.data();

        float* token_result = thread_local_results + token_idx * moe_params.hidden_size;

        // Process all experts for this token
        for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
          float routing_weight = ToFloat(router_probs_data[token_idx * moe_params.num_experts + expert_idx]);
          if (routing_weight <= 1e-6f) continue;  // Skip experts with negligible routing weight

          // FC1: input -> intermediate using pre-dequantized weights + MLAS SGEMM
          const float* fc1_expert_weights = dequant_fc1_weights.get() + expert_idx * moe_params.hidden_size * moe_params.inter_size;
          const MLFloat16* fc1_expert_bias_typed = fc1_bias_data ? fc1_bias_data + expert_idx * moe_params.inter_size : nullptr;

          // Use MLAS SGEMM for FC1: input [1 x hidden_size] * weights [hidden_size x inter_size] = output [1 x inter_size]
          MLAS_SGEMM_DATA_PARAMS fc1_params;
          fc1_params.A = token_input;
          fc1_params.lda = moe_params.hidden_size;
          fc1_params.B = fc1_expert_weights;
          fc1_params.ldb = moe_params.hidden_size;
          fc1_params.C = thread_fc1_output;
          fc1_params.ldc = moe_params.inter_size;
          fc1_params.alpha = 1.0f;
          fc1_params.beta = 0.0f;

          MlasGemm(CblasNoTrans, CblasNoTrans, 1, static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(moe_params.hidden_size), fc1_params, nullptr);

          // Add bias and apply activation
          for (int64_t i = 0; i < moe_params.inter_size; ++i) {
            if (fc1_expert_bias_typed) {
              thread_fc1_output[i] += ToFloat(fc1_expert_bias_typed[i]);
            }
            thread_fc1_output[i] = ApplyActivation(thread_fc1_output[i], activation_type_);
          }

          // FC2: intermediate -> output using pre-dequantized weights + MLAS SGEMM
          const float* fc2_expert_weights = dequant_fc2_weights.get() + expert_idx * moe_params.inter_size * moe_params.hidden_size;
          const MLFloat16* fc2_expert_bias_typed = fc2_bias_data ? fc2_bias_data + expert_idx * moe_params.hidden_size : nullptr;

          // Use MLAS SGEMM for FC2: intermediate [1 x inter_size] * weights [inter_size x hidden_size] = output [1 x hidden_size]
          MLAS_SGEMM_DATA_PARAMS fc2_params;
          fc2_params.A = thread_fc1_output;
          fc2_params.lda = moe_params.inter_size;
          fc2_params.B = fc2_expert_weights;
          fc2_params.ldb = moe_params.inter_size;
          fc2_params.C = thread_fc2_output;
          fc2_params.ldc = moe_params.hidden_size;
          fc2_params.alpha = 1.0f;
          fc2_params.beta = 0.0f;

          MlasGemm(CblasNoTrans, CblasNoTrans, 1, static_cast<size_t>(moe_params.hidden_size), static_cast<size_t>(moe_params.inter_size), fc2_params, nullptr);

          // Add bias, apply routing weight, and accumulate to final result
          for (int64_t i = 0; i < moe_params.hidden_size; ++i) {
            if (fc2_expert_bias_typed) {
              thread_fc2_output[i] += ToFloat(fc2_expert_bias_typed[i]);
            }
            token_result[i] += routing_weight * thread_fc2_output[i];
          }
        }
      }
    };

    // Execute token processing in parallel across threads
    concurrency::ThreadPool::TryParallelFor(thread_pool, moe_params.num_rows,
                                            static_cast<double>(std::max<int64_t>(1, moe_params.num_rows / num_threads)),
                                            process_token_range);
  }

  // Main thread reduction: combine all thread-local results into final output
  for (int64_t thread_id = 0; thread_id < num_threads; ++thread_id) {
    const float* thread_local_results = thread_results.get() + thread_id * moe_params.num_rows * moe_params.hidden_size;
    for (int64_t token_idx = 0; token_idx < moe_params.num_rows; ++token_idx) {
      for (int64_t col = 0; col < moe_params.hidden_size; ++col) {
        size_t idx = static_cast<size_t>(token_idx * moe_params.hidden_size + col);
        output_data[idx] = FromFloat(ToFloat(output_data[idx]) + thread_local_results[idx]);
      }
    }
  }

  // Suppress unused parameter warnings for optional parameters
  ORT_UNUSED_PARAMETER(fc3_experts_bias_optional);
  ORT_UNUSED_PARAMETER(fc3_scales_optional);

  return Status::OK();
}

// Explicit template instantiations
template Status QMoE::QuantizedMoEImpl<true>(OpKernelContext* context,
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
                                             const Tensor* fc3_scales_optional) const;

template Status QMoE::QuantizedMoEImpl<false>(OpKernelContext* context,
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
                                              const Tensor* fc3_scales_optional) const;

}  // namespace contrib
}  // namespace onnxruntime
