// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/moe_quantization_cpu.h"
#include "core/framework/allocator.h"
#include "core/framework/buffer_deleter.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/platform/threadpool.h"
#include "contrib_ops/cpu/moe/moe_utils.h"
#include <algorithm>

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

  // SwiGLU validation - FC3 not supported
  bool is_swiglu = (activation_type_ == ActivationType::SwiGLU);
  if (is_swiglu && fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SwiGLU activation is not supported with fc3.");
  }
  if (!is_swiglu && fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "FC3 gating is not yet implemented on CPU.");
  }

  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  const int64_t num_threads = std::min<int64_t>(
      static_cast<int64_t>(concurrency::ThreadPool::DegreeOfParallelism(thread_pool)),
      moe_params.num_rows);

  const int64_t total_output_size = moe_params.num_rows * moe_params.hidden_size;
  std::fill_n(output_data, total_output_size, MLFloat16(0.0f));

  auto thread_fc1_buffers = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_threads * moe_params.inter_size * (is_swiglu ? 2 : 1)));
  auto thread_fc2_buffers = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_threads * moe_params.hidden_size));
  auto thread_results = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_threads * moe_params.num_rows * moe_params.hidden_size));

  const int64_t max_bias_size = std::max(moe_params.inter_size * (is_swiglu ? 2 : 1), moe_params.hidden_size);
  auto thread_bias_buffers = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_threads * max_bias_size));

  // Pre-convert all input data from MLFloat16 to float using parallel MLAS conversion
  auto input_float = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size));
  MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(input_data),
                                         input_float.get(),
                                         static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size),
                                         thread_pool);

  // Pre-convert all router probabilities to avoid repeated conversions
  auto router_probs_float = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(moe_params.num_rows * moe_params.num_experts));
  MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(router_probs_data),
                                         router_probs_float.get(),
                                         static_cast<size_t>(moe_params.num_rows * moe_params.num_experts),
                                         thread_pool);

  // Initialize thread results to zero using optimized memset
  std::memset(thread_results.get(), 0,
              static_cast<size_t>(num_threads * moe_params.num_rows * moe_params.hidden_size) * sizeof(float));

  // Determine quantization parameters based on bit width
  const bool is_4bit = UseUInt4x2;
  const float zero_point = is_4bit ? 8.0f : 128.0f;
  const int64_t act_multiplier = is_swiglu ? 2 : 1;
  const int64_t fc1_output_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;

  // Calculate weight sizes and strides based on quantization type
  const int64_t fc1_weight_stride = is_4bit ? (moe_params.hidden_size * fc1_output_size / 2) : (moe_params.hidden_size * moe_params.inter_size * act_multiplier);
  const int64_t fc2_weight_stride = is_4bit ? (moe_params.inter_size * moe_params.hidden_size / 2) : (moe_params.inter_size * moe_params.hidden_size);

  // Pre-dequantize all expert weights once (shared across all threads)
  auto dequant_fc1_weights = IAllocator::MakeUniquePtr<float>(allocator,
                                                              static_cast<size_t>(moe_params.num_experts * moe_params.hidden_size * (is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier)));
  auto dequant_fc2_weights = IAllocator::MakeUniquePtr<float>(allocator,
                                                              static_cast<size_t>(moe_params.num_experts * moe_params.inter_size * moe_params.hidden_size));

  // Helper lambda for dequantizing a single weight value
  auto DequantizeWeight = [&](const uint8_t* weights, size_t weight_idx, size_t linear_idx,
                              const float* scales, int64_t scale_idx) -> float {
    if (is_4bit) {
      // For Int4, two values are packed in each uint8
      size_t packed_idx = linear_idx / 2;
      uint8_t packed_value = weights[packed_idx];
      uint8_t quantized_weight = (linear_idx % 2 == 0) ? (packed_value & 0x0F) : ((packed_value >> 4) & 0x0F);
      return (static_cast<float>(quantized_weight) - zero_point) * scales[scale_idx];
    } else {
      // For Int8, direct access
      return (static_cast<float>(weights[weight_idx]) - zero_point) * scales[scale_idx];
    }
  };

  // Dequantize FC1 weights for all experts
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_experts),
      static_cast<double>(std::max<int64_t>(1, moe_params.num_experts / num_threads)),
      [&](ptrdiff_t expert_start, ptrdiff_t expert_end) {
        for (std::ptrdiff_t expert_idx = expert_start; expert_idx < expert_end; ++expert_idx) {
          const uint8_t* fc1_expert_weights = fc1_weights_data + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * fc1_weight_stride;
          const float* fc1_expert_scales = fc1_scales_data + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * (is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier);
          float* dequant_fc1_expert = dequant_fc1_weights.get() + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.hidden_size * (is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier);

          const int64_t output_cols = is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier;
          for (int64_t out_col = 0; out_col < output_cols; ++out_col) {
            for (int64_t in_col = 0; in_col < moe_params.hidden_size; ++in_col) {
              size_t linear_idx = static_cast<size_t>(out_col * moe_params.hidden_size + in_col);
              dequant_fc1_expert[linear_idx] = DequantizeWeight(fc1_expert_weights, linear_idx, linear_idx, fc1_expert_scales, out_col);
            }
          }
        }
      });

  // Dequantize FC2 weights for all experts
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_experts),
      static_cast<double>(std::max<int64_t>(1, moe_params.num_experts / num_threads)),
      [&](ptrdiff_t expert_start, ptrdiff_t expert_end) {
        for (std::ptrdiff_t expert_idx = expert_start; expert_idx < expert_end; ++expert_idx) {
          const uint8_t* fc2_expert_weights = fc2_weights_data + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * fc2_weight_stride;
          const float* fc2_expert_scales = fc2_scales_data + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.hidden_size;
          float* dequant_fc2_expert = dequant_fc2_weights.get() + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.inter_size * moe_params.hidden_size;

          for (int64_t out_col = 0; out_col < moe_params.hidden_size; ++out_col) {
            for (int64_t in_col = 0; in_col < moe_params.inter_size; ++in_col) {
              size_t linear_idx = static_cast<size_t>(out_col * moe_params.inter_size + in_col);
              dequant_fc2_expert[linear_idx] = DequantizeWeight(fc2_expert_weights, linear_idx, linear_idx, fc2_expert_scales, out_col);
            }
          }
        }
      });

  // Process tokens in parallel
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_rows),
      static_cast<double>(std::max<int64_t>(1, moe_params.num_rows / num_threads)),
      [&](ptrdiff_t start_token, ptrdiff_t end_token) {
        const int64_t thread_id = start_token / ((moe_params.num_rows + num_threads - 1) / num_threads);
        const int64_t thread_fc1_size = is_4bit ? (moe_params.inter_size * (is_swiglu ? 2 : 1)) : (moe_params.inter_size * act_multiplier);
        float* thread_fc1_output = thread_fc1_buffers.get() + thread_id * thread_fc1_size;
        float* thread_fc2_output = thread_fc2_buffers.get() + thread_id * moe_params.hidden_size;
        float* thread_local_results = thread_results.get() + thread_id * moe_params.num_rows * moe_params.hidden_size;
        float* thread_bias_buffer = thread_bias_buffers.get() + thread_id * max_bias_size;

        // Process each token in this thread's range
        for (std::ptrdiff_t token_idx = start_token; token_idx < end_token; ++token_idx) {
          const float* token_input = input_float.get() + static_cast<int64_t>(SafeInt<int64_t>(token_idx)) * moe_params.hidden_size;
          float* token_result = thread_local_results + static_cast<int64_t>(SafeInt<int64_t>(token_idx)) * moe_params.hidden_size;

          // Process all experts for this token
          for (std::ptrdiff_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
            float routing_weight = router_probs_float.get()[static_cast<int64_t>(SafeInt<int64_t>(token_idx)) * moe_params.num_experts + static_cast<int64_t>(SafeInt<int64_t>(expert_idx))];
            if (routing_weight <= 1e-6f) continue;  // Skip experts with negligible routing weight

            // FC1: input -> intermediate using pre-dequantized weights + MLAS SGEMM
            const int64_t fc1_weight_offset = is_4bit ? (static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.hidden_size * fc1_output_size) : (static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.hidden_size * moe_params.inter_size * act_multiplier);
            const float* fc1_expert_weights = dequant_fc1_weights.get() + fc1_weight_offset;

            const int64_t fc1_bias_size = is_4bit ? fc1_output_size : (moe_params.inter_size * act_multiplier);
            const MLFloat16* fc1_expert_bias_typed = fc1_bias_data ? fc1_bias_data + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * fc1_bias_size : nullptr;

            // Use MLAS SGEMM for FC1
            MLAS_SGEMM_DATA_PARAMS fc1_params;
            fc1_params.A = token_input;
            fc1_params.lda = static_cast<size_t>(moe_params.hidden_size);
            fc1_params.B = fc1_expert_weights;
            fc1_params.ldb = static_cast<size_t>(moe_params.hidden_size);
            fc1_params.C = thread_fc1_output;
            fc1_params.ldc = static_cast<size_t>(fc1_bias_size);
            fc1_params.alpha = 1.0f;
            fc1_params.beta = 0.0f;

            MlasGemm(CblasNoTrans, CblasNoTrans, 1, static_cast<size_t>(fc1_bias_size), static_cast<size_t>(moe_params.hidden_size), fc1_params, nullptr);

            // Handle different activation types
            if (is_swiglu) {
              // Add bias if present
              if (fc1_expert_bias_typed) {
                MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc1_expert_bias_typed),
                                             thread_bias_buffer, static_cast<size_t>(fc1_bias_size));
                for (int64_t i = 0; i < fc1_bias_size; ++i) {
                  thread_fc1_output[i] += thread_bias_buffer[i];
                }
              }

              if (is_4bit) {
                // Apply SwiGLU using the helper function
                ApplySwiGLU(thread_fc1_output, thread_fc1_output, moe_params.inter_size);
              } else {
                // For Int8, handle chunked layout manually
                float* linear_part = thread_fc1_output;
                float* gate_part = thread_fc1_output + moe_params.inter_size;

                constexpr float swiglu_alpha = 1.702f;
                for (int64_t i = 0; i < moe_params.inter_size; ++i) {
                  float sigmoid_arg = swiglu_alpha * gate_part[i];
                  float sigmoid_out = 1.0f / (1.0f + expf(-sigmoid_arg));
                  float swish_out = gate_part[i] * sigmoid_out;
                  thread_fc1_output[i] = swish_out * (linear_part[i] + 1.0f);
                }
              }
            } else {
              // Standard activation (non-SwiGLU)
              if (fc1_expert_bias_typed) {
                MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc1_expert_bias_typed),
                                             thread_bias_buffer, static_cast<size_t>(moe_params.inter_size));
                for (int64_t i = 0; i < moe_params.inter_size; ++i) {
                  thread_fc1_output[i] += thread_bias_buffer[i];
                  thread_fc1_output[i] = ApplyActivation(thread_fc1_output[i], activation_type_);
                }
              } else {
                for (int64_t i = 0; i < moe_params.inter_size; ++i) {
                  thread_fc1_output[i] = ApplyActivation(thread_fc1_output[i], activation_type_);
                }
              }
            }

            // FC2: intermediate -> output using pre-dequantized weights + MLAS SGEMM
            const float* fc2_expert_weights = dequant_fc2_weights.get() + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.inter_size * moe_params.hidden_size;
            const MLFloat16* fc2_expert_bias_typed = fc2_bias_data ? fc2_bias_data + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.hidden_size : nullptr;

            // Use MLAS SGEMM for FC2
            MLAS_SGEMM_DATA_PARAMS fc2_params;
            fc2_params.A = thread_fc1_output;
            fc2_params.lda = static_cast<size_t>(moe_params.inter_size);
            fc2_params.B = fc2_expert_weights;
            fc2_params.ldb = static_cast<size_t>(moe_params.inter_size);
            fc2_params.C = thread_fc2_output;
            fc2_params.ldc = static_cast<size_t>(moe_params.hidden_size);
            fc2_params.alpha = 1.0f;
            fc2_params.beta = 0.0f;

            MlasGemm(CblasNoTrans, CblasNoTrans, 1, static_cast<size_t>(moe_params.hidden_size), static_cast<size_t>(moe_params.inter_size), fc2_params, nullptr);

            // Add bias, apply routing weight, and accumulate to final result
            if (fc2_expert_bias_typed) {
              MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc2_expert_bias_typed),
                                           thread_bias_buffer, static_cast<size_t>(moe_params.hidden_size));
              for (int64_t i = 0; i < moe_params.hidden_size; ++i) {
                token_result[i] += routing_weight * (thread_fc2_output[i] + thread_bias_buffer[i]);
              }
            } else {
              for (int64_t i = 0; i < moe_params.hidden_size; ++i) {
                token_result[i] += routing_weight * thread_fc2_output[i];
              }
            }
          }
        }
      });

  // Allocate float buffer for final accumulation
  void* float_output_ptr = allocator->Alloc(static_cast<size_t>(total_output_size * sizeof(float)));
  BufferUniquePtr float_output_buffer(float_output_ptr, BufferDeleter(allocator));
  float* float_output = reinterpret_cast<float*>(float_output_ptr);

  // Main thread reduction: combine all thread-local results into float buffer
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_rows),
      static_cast<double>(std::max<int64_t>(1, moe_params.num_rows / num_threads)),
      [&](ptrdiff_t token_start, ptrdiff_t token_end) {
        for (std::ptrdiff_t token_idx = token_start; token_idx < token_end; ++token_idx) {
          int64_t token_idx_safe = SafeInt<int64_t>(token_idx);
          for (int64_t col = 0; col < moe_params.hidden_size; ++col) {
            size_t idx = static_cast<size_t>(token_idx_safe * moe_params.hidden_size + col);
            float accumulated = 0.0f;

            // Accumulate results from all threads for this position
            for (int64_t thread_id = 0; thread_id < num_threads; ++thread_id) {
              const float* thread_local_results = thread_results.get() + thread_id * moe_params.num_rows * moe_params.hidden_size;
              accumulated += thread_local_results[idx];
            }

            float_output[idx] = accumulated;
          }
        }
      });

  // Convert final float results to MLFloat16 using optimized MLAS conversion
  MlasConvertFloatToHalfBuffer(float_output, reinterpret_cast<MLAS_FP16*>(output_data), static_cast<size_t>(total_output_size));

  // Suppress unused parameter warnings for optional parameters that are not used in non-SwiGLU modes
  if (!is_swiglu) {
    ORT_UNUSED_PARAMETER(fc3_experts_bias_optional);
    ORT_UNUSED_PARAMETER(fc3_scales_optional);
  }

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
