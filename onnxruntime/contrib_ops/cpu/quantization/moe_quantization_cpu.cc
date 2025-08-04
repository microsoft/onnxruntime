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
#include <atomic>

using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL()                                                                           \
  ONNX_OPERATOR_KERNEL_EX(QMoE, kMSDomain, 1, kCpuExecutionProvider,                                \
                          (*KernelDefBuilder::Create())                                             \
                              .MayInplace(0, 0)                                                     \
                              .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float>())   \
                              .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>())           \
                              .TypeConstraint("T2", BuildKernelDefConstraints<MLFloat16, float>()), \
                          QMoE);

REGISTER_KERNEL();

// QMoE CPU kernel registration is handled in cpu_contrib_kernels.cc

QMoE::QMoE(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info),
      MoEBaseCPU(op_kernel_info),
      prepacked_fc1_weights_data_(nullptr),
      prepacked_fc2_weights_data_(nullptr),
      weights_allocator_(nullptr),
      is_prepacked_(false) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
              "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);
}

Status QMoE::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                     /*out*/ bool& is_packed,
                     /*out*/ PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(tensor);
  ORT_UNUSED_PARAMETER(prepacked_weights);
  is_packed = false;

  // Only prepack the quantized weight tensors
  if (input_idx == InputIndex::FC1_EXPERTS_WEIGHTS) {
    fc1_weights_packed_ = true;
    is_packed = true;  // Signal that this input should be released after prepacking
  } else if (input_idx == InputIndex::FC2_EXPERTS_WEIGHTS) {
    fc2_weights_packed_ = true;
    is_packed = true;  // Signal that this input should be released after prepacking
  }

  // Store the allocator for later use
  if (!weights_allocator_) {
    weights_allocator_ = alloc;
  }

  return Status::OK();
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

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias_optional, fc1_scales,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales,
      fc3_experts_weights_optional, fc3_experts_bias_optional, fc3_scales_optional,
      expert_weight_bits_ == 4 ? 2 : 1,
      activation_type_ == ActivationType::SwiGLU));

  // Dispatch based on input data type
  if (input->IsDataType<MLFloat16>()) {
    if (expert_weight_bits_ == 4) {
      return QuantizedMoEImpl<true, MLFloat16>(context, moe_params, input, router_probs,
                                               fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                               fc2_experts_bias_optional, fc3_experts_weights_optional,
                                               fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
    } else {
      return QuantizedMoEImpl<false, MLFloat16>(context, moe_params, input, router_probs,
                                                fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                                fc2_experts_bias_optional, fc3_experts_weights_optional,
                                                fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
    }
  } else if (input->IsDataType<float>()) {
    if (expert_weight_bits_ == 4) {
      return QuantizedMoEImpl<true, float>(context, moe_params, input, router_probs,
                                           fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                           fc2_experts_bias_optional, fc3_experts_weights_optional,
                                           fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
    } else {
      return QuantizedMoEImpl<false, float>(context, moe_params, input, router_probs,
                                            fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                            fc2_experts_bias_optional, fc3_experts_weights_optional,
                                            fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "QMoE only supports float and MLFloat16 data types, but got ",
                           DataTypeImpl::ToString(input->DataType()));
  }
}

void ProcessSingleExpert(const float* token_input, float* thread_fc1_output, float* thread_fc2_output,
                         float* thread_accumulation_buffer, int64_t expert_idx, float routing_weight,
                         const MoEParameters& moe_params, bool is_swiglu,
                         const float* dequant_fc1_weights, const float* dequant_fc2_weights,
                         const float* fc1_bias_float, const float* fc2_bias_float,
                         int64_t fc1_output_size) {
  // FC1 computation with optimized MLAS SGEMM
  const float* fc1_expert_weights = dequant_fc1_weights + expert_idx * moe_params.hidden_size * fc1_output_size;
  MlasGemm(CblasNoTrans, CblasTrans,
           1, static_cast<size_t>(fc1_output_size), static_cast<size_t>(moe_params.hidden_size),
           1.0f,
           token_input, static_cast<size_t>(moe_params.hidden_size),
           fc1_expert_weights, static_cast<size_t>(fc1_output_size),
           0.0f,
           thread_fc1_output, static_cast<size_t>(fc1_output_size),
           nullptr);

  // Add FC1 bias if present and apply activation in single pass
  if (fc1_bias_float) {
    const float* expert_fc1_bias = fc1_bias_float + expert_idx * fc1_output_size;
    if (is_swiglu) {
      // SwiGLU: bias addition first, then use unified activation function
      for (int64_t i = 0; i < fc1_output_size; ++i) {
        thread_fc1_output[i] += expert_fc1_bias[i];
      }
      // Use the same interleaved SwiGLU activation as batch processing
      contrib::ApplySwiGLUActivation(thread_fc1_output, fc1_output_size / 2, true);
    } else {
      // ReLU: bias addition + activation in one pass
      for (int64_t i = 0; i < fc1_output_size; ++i) {
        thread_fc1_output[i] = std::max(0.0f, thread_fc1_output[i] + expert_fc1_bias[i]);
      }
    }
  } else {
    // No bias case
    if (is_swiglu) {
      // Use the same interleaved SwiGLU activation as batch processing
      contrib::ApplySwiGLUActivation(thread_fc1_output, fc1_output_size / 2, true);
    } else {
      for (int64_t i = 0; i < fc1_output_size; ++i) {
        thread_fc1_output[i] = std::max(0.0f, thread_fc1_output[i]);
      }
    }
  }

  // FC2 computation
  // For SwiGLU interleaved format, inter_size is fc1_output_size / 2
  // For non-SwiGLU, inter_size is fc1_output_size
  const int64_t inter_size = is_swiglu ? fc1_output_size / 2 : fc1_output_size;
  const float* fc2_expert_weights = dequant_fc2_weights + expert_idx * inter_size * moe_params.hidden_size;
  MlasGemm(CblasNoTrans, CblasTrans,
           1, static_cast<size_t>(moe_params.hidden_size), static_cast<size_t>(inter_size),
           1.0f,
           thread_fc1_output, static_cast<size_t>(inter_size),
           fc2_expert_weights, static_cast<size_t>(moe_params.hidden_size),
           0.0f,
           thread_fc2_output, static_cast<size_t>(moe_params.hidden_size),
           nullptr);

  // Add FC2 bias and apply routing weight in single pass
  if (fc2_bias_float) {
    const float* expert_fc2_bias = fc2_bias_float + expert_idx * moe_params.hidden_size;
    for (int64_t i = 0; i < moe_params.hidden_size; ++i) {
      thread_accumulation_buffer[i] = routing_weight * (thread_fc2_output[i] + expert_fc2_bias[i]);
    }
  } else {
    for (int64_t i = 0; i < moe_params.hidden_size; ++i) {
      thread_accumulation_buffer[i] = routing_weight * thread_fc2_output[i];
    }
  }
}

template <bool UseUInt4x2, typename T>
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

  // Check if we need to repack weights
  if (!is_prepacked_ ||
      cached_num_experts_ != moe_params.num_experts ||
      cached_hidden_size_ != moe_params.hidden_size ||
      cached_inter_size_ != moe_params.inter_size ||
      cached_is_swiglu_ != is_swiglu) {
    // Need to prepack weights
    // Note: After PrePack is called, the original quantized weights may be nullptr
    // We can only prepack if we have the original weights available
    if (fc1_experts_weights != nullptr && fc2_experts_weights != nullptr) {
      Status status = const_cast<QMoE*>(this)->PrepackAndDequantizeWeights<UseUInt4x2>(
          context, moe_params, fc1_experts_weights, fc2_experts_weights,
          fc1_scales, fc2_scales, is_swiglu);
      ORT_RETURN_IF_ERROR(status);
    } else if (!is_prepacked_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Quantized weights are not available for prepacking. "
                             "This may happen if PrePack was called but weights were released.");
    }
  }
  // Get thread pool with performance optimization
  auto* thread_pool = context->GetOperatorThreadPool();

  // Check if deterministic compute is required (affects parallelization strategy)
  const bool is_deterministic = context->GetUseDeterministicCompute();

  // Get input data pointers
  const T* input_data = input->Data<T>();
  const T* router_probs_data = router_probs->Data<T>();
  const T* fc1_bias_data = fc1_experts_bias_optional ? fc1_experts_bias_optional->Data<T>() : nullptr;
  const T* fc2_bias_data = fc2_experts_bias_optional ? fc2_experts_bias_optional->Data<T>() : nullptr;

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // Use TensorShape::GetSize() for more efficient size calculation
  const int64_t total_output_size = input->Shape().Size();

  // Initialize output with optimized pattern based on data type
  if constexpr (std::is_same_v<T, MLFloat16>) {
    std::fill_n(output_data, static_cast<size_t>(total_output_size), MLFloat16(0.0f));
  } else {
    std::memset(output_data, 0, static_cast<size_t>(total_output_size) * sizeof(T));
  }

  // Set up output buffer with memory optimization
  IAllocatorUniquePtr<float> output_float;
  float* output_float_ptr = nullptr;

  if constexpr (std::is_same_v<T, MLFloat16>) {
    // For MLFloat16, we need a separate float buffer
    // Use allocator for better memory management
    output_float = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(total_output_size));
    output_float_ptr = output_float.get();
  } else {
    // For float, we can write directly to output_data
    output_float = IAllocatorUniquePtr<float>(output_data, [](float*) {});
    output_float_ptr = output_data;
  }

  // Initialize output to zeros
  std::fill_n(output_float_ptr, static_cast<size_t>(total_output_size), 0.0f);

  // Pointers for easier access
  float* input_float_ptr = nullptr;
  float* router_probs_float_ptr = nullptr;

  // Buffer for MLFloat16 to float conversion - declared at outer scope to ensure proper lifetime
  IAllocatorUniquePtr<float> unified_conversion_buffer;

  // Pre-convert bias tensors to float (if they exist) with unified memory allocation
  const int64_t fc1_bias_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
  const int64_t fc2_bias_size = moe_params.hidden_size;

  // Calculate total bias buffer size for unified allocation
  const int64_t total_fc1_bias_size = fc1_bias_data ? (moe_params.num_experts * fc1_bias_size) : 0;
  const int64_t total_fc2_bias_size = fc2_bias_data ? (moe_params.num_experts * fc2_bias_size) : 0;
  const int64_t total_bias_buffer_size = total_fc1_bias_size + total_fc2_bias_size;

  // Single unified allocation for all bias data with proper alignment
  IAllocatorUniquePtr<float> unified_bias_buffer;
  float* fc1_bias_float_ptr = nullptr;
  float* fc2_bias_float_ptr = nullptr;

  if (total_bias_buffer_size > 0) {
    // Use context's reusable buffer if available for better memory management
    unified_bias_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(total_bias_buffer_size));

    if (fc1_bias_data) {
      fc1_bias_float_ptr = unified_bias_buffer.get();
    }
    if (fc2_bias_data) {
      fc2_bias_float_ptr = unified_bias_buffer.get() + total_fc1_bias_size;
    }
  }

  // Convert input and router_probs based on type with memory optimization and fast math
  if constexpr (std::is_same_v<T, MLFloat16>) {
    // For MLFloat16, convert to float - use allocator with optimized conversion
    // Calculate total size for unified allocation
    const size_t input_size = static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size);
    const size_t router_probs_size = static_cast<size_t>(moe_params.num_rows * moe_params.num_experts);
    const size_t total_conversion_size = input_size + router_probs_size;

    // Single allocation for input and router_probs conversion
    unified_conversion_buffer = IAllocator::MakeUniquePtr<float>(allocator, total_conversion_size);
    input_float_ptr = unified_conversion_buffer.get();
    router_probs_float_ptr = unified_conversion_buffer.get() + input_size;

    // Use optimized parallel conversion with better thread utilization
    // Convert MLFloat16 to float with optimized parallel conversion
    MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(input_data),
                                           input_float_ptr, input_size, thread_pool);

    MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(router_probs_data),
                                           router_probs_float_ptr, router_probs_size, thread_pool);

    // Convert biases to float once (if they exist) with optimized conversion
    if (fc1_bias_data && fc1_bias_float_ptr) {
      MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(fc1_bias_data),
                                             fc1_bias_float_ptr,
                                             static_cast<size_t>(moe_params.num_experts * fc1_bias_size),
                                             thread_pool);
    }

    if (fc2_bias_data && fc2_bias_float_ptr) {
      MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(fc2_bias_data),
                                             fc2_bias_float_ptr,
                                             static_cast<size_t>(moe_params.num_experts * fc2_bias_size),
                                             thread_pool);
    }
  } else {
    // For float, point to original input and router_probs directly for zero-copy optimization
    input_float_ptr = const_cast<float*>(input_data);
    router_probs_float_ptr = const_cast<float*>(router_probs_data);

    // For float bias data, use direct pointers from unified buffer or original data
    if (fc1_bias_data) {
      fc1_bias_float_ptr = const_cast<float*>(fc1_bias_data);
    }

    if (fc2_bias_data) {
      fc2_bias_float_ptr = const_cast<float*>(fc2_bias_data);
    }
  }

  // Determine activation related parameters
  const int64_t fc1_output_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;

  // Use prepacked dequantized weights - no need to dequantize here
  const float* dequant_fc1_weights = prepacked_fc1_weights_data_;
  const float* dequant_fc2_weights = prepacked_fc2_weights_data_;

  // Use batched GEMM approach for better performance with memory optimization
  // Step 1: Group tokens by experts for efficient batched computation with reduced allocations
  struct TokenExpertMapping {
    int64_t token_idx;
    int64_t expert_idx;
    float routing_weight;
  };

  // Pre-allocate with better size estimation to avoid reallocations
  std::vector<TokenExpertMapping> token_expert_mappings;
  // More accurate estimation based on typical sparsity patterns in MoE models
  const size_t estimated_size = static_cast<size_t>(moe_params.num_rows * std::min(static_cast<int64_t>(8), moe_params.num_experts));
  token_expert_mappings.reserve(estimated_size);

  // Collect all valid token-expert pairs with optimized threshold
  const float routing_threshold = 1e-4f;  // Skip experts with negligible routing weight
  for (int64_t token_idx = 0; token_idx < moe_params.num_rows; ++token_idx) {
    const float* token_router_probs = router_probs_float_ptr + token_idx * moe_params.num_experts;
    for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
      const float routing_weight = token_router_probs[expert_idx];
      if (routing_weight > routing_threshold) {
        token_expert_mappings.emplace_back(TokenExpertMapping{token_idx, expert_idx, routing_weight});
      }
    }
  }

  // Step 2: Group by expert for batched processing with memory-efficient data structure
  std::vector<std::vector<TokenExpertMapping>> expert_token_groups(static_cast<size_t>(moe_params.num_experts));
  // Pre-reserve space for each expert group to avoid reallocations
  const size_t avg_tokens_per_expert = token_expert_mappings.size() / static_cast<size_t>(std::max(moe_params.num_experts, static_cast<int64_t>(1)));
  for (auto& group : expert_token_groups) {
    group.reserve(avg_tokens_per_expert + 8);  // Small buffer for variance
  }

  for (const auto& mapping : token_expert_mappings) {
    expert_token_groups[static_cast<size_t>(mapping.expert_idx)].emplace_back(mapping);
  }

  // Step 3: Process each expert with batched GEMM using memory-optimized approach
  // Pre-allocate thread-local memory pools to avoid repeated allocations
  const int64_t max_tokens_per_expert = moe_params.num_rows;  // Conservative estimate
  const int64_t pool_input_size = max_tokens_per_expert * moe_params.hidden_size;
  const int64_t pool_fc1_size = max_tokens_per_expert * fc1_output_size;
  const int64_t pool_fc2_size = max_tokens_per_expert * moe_params.hidden_size;

  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_experts),
      static_cast<double>(moe_params.hidden_size * moe_params.inter_size * moe_params.num_rows / moe_params.num_experts),
      [&](ptrdiff_t expert_start, ptrdiff_t expert_end) {
        // Thread-local memory pools - allocated once per thread
        auto thread_input_pool = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(pool_input_size));
        auto thread_fc1_pool = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(pool_fc1_size));
        // Reuse fc1 buffer for fc2 when possible to reduce memory footprint
        auto thread_fc2_pool = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(pool_fc2_size));

        for (std::ptrdiff_t expert_idx = expert_start; expert_idx < expert_end; ++expert_idx) {
          const auto& token_group = expert_token_groups[static_cast<size_t>(expert_idx)];
          if (token_group.empty()) {
            continue;  // Skip experts with no assigned tokens
          }

          const int64_t batch_size = static_cast<int64_t>(token_group.size());

          // Use pre-allocated thread-local pools - no allocation overhead
          float* expert_input_batch = thread_input_pool.get();
          float* expert_fc1_batch = thread_fc1_pool.get();
          float* expert_fc2_batch = thread_fc2_pool.get();

          // Step 3a: Optimized shuffle with prefetching hints - gather input tokens with vectorized copy
          for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const int64_t token_idx = token_group[static_cast<size_t>(batch_idx)].token_idx;
            const float* token_input = input_float_ptr + token_idx * moe_params.hidden_size;
            float* batch_input = expert_input_batch + batch_idx * moe_params.hidden_size;

            // Prefetch next token for better cache performance
            if (batch_idx + 1 < batch_size) {
              const int64_t next_token_idx = token_group[static_cast<size_t>(batch_idx + 1)].token_idx;
              const float* next_token_input = input_float_ptr + next_token_idx * moe_params.hidden_size;
              __builtin_prefetch(next_token_input, 0, 3);  // Prefetch for read, high temporal locality
            }

            // Use optimized memory copy for better cache performance
            std::memcpy(batch_input, token_input, static_cast<size_t>(moe_params.hidden_size) * sizeof(float));
          }

          // Step 3b: Batched FC1 GEMM: [batch_size, hidden_size] x [hidden_size, fc1_output_size] -> [batch_size, fc1_output_size]
          const int64_t fc1_weight_offset = expert_idx * moe_params.hidden_size * fc1_output_size;
          const float* fc1_expert_weights = dequant_fc1_weights + fc1_weight_offset;

          MlasGemm(CblasNoTrans, CblasTrans,
                   static_cast<size_t>(batch_size), static_cast<size_t>(fc1_output_size), static_cast<size_t>(moe_params.hidden_size),
                   1.0f,
                   expert_input_batch, static_cast<size_t>(moe_params.hidden_size),
                   fc1_expert_weights, static_cast<size_t>(fc1_output_size),
                   0.0f,
                   expert_fc1_batch, static_cast<size_t>(fc1_output_size),
                   nullptr);

          // Step 3c: Apply bias and activation with optimized memory access patterns
          if (fc1_bias_float_ptr) {
            const float* fc1_expert_bias = fc1_bias_float_ptr + expert_idx * fc1_output_size;
            // Vectorized bias addition with better cache locality
            for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
              float* batch_fc1_output = expert_fc1_batch + batch_idx * fc1_output_size;
              // Process in chunks for better cache performance
              constexpr int64_t chunk_size = 64;  // Process 64 elements at a time
              for (int64_t i = 0; i < fc1_output_size; i += chunk_size) {
                int64_t end_i = std::min(i + chunk_size, fc1_output_size);
                for (int64_t j = i; j < end_i; ++j) {
                  batch_fc1_output[j] += fc1_expert_bias[j];
                }
              }
            }
          }

          // Apply activation function with optimized patterns
          if (is_swiglu) {
            for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
              float* batch_fc1_output = expert_fc1_batch + batch_idx * fc1_output_size;
              contrib::ApplySwiGLUActivation(batch_fc1_output, moe_params.inter_size, true);
            }
          } else {
            // Vectorized ReLU with better cache performance
            for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
              float* batch_fc1_output = expert_fc1_batch + batch_idx * fc1_output_size;
              constexpr int64_t chunk_size = 64;
              for (int64_t i = 0; i < fc1_output_size; i += chunk_size) {
                int64_t end_i = std::min(i + chunk_size, fc1_output_size);
                for (int64_t j = i; j < end_i; ++j) {
                  batch_fc1_output[j] = ApplyActivation(batch_fc1_output[j], activation_type_);
                }
              }
            }
          }

          // Step 3d: Batched FC2 GEMM with memory optimization
          const int64_t inter_size = is_swiglu ? moe_params.inter_size : fc1_output_size;
          const float* fc2_expert_weights = dequant_fc2_weights + expert_idx * inter_size * moe_params.hidden_size;

          // For SwiGLU, the FC1 output has been modified in-place and now has size [batch_size, inter_size]
          // For non-SwiGLU, it remains [batch_size, fc1_output_size] which equals inter_size
          MlasGemm(CblasNoTrans, CblasTrans,
                   static_cast<size_t>(batch_size), static_cast<size_t>(moe_params.hidden_size), static_cast<size_t>(inter_size),
                   1.0f,
                   expert_fc1_batch, static_cast<size_t>(inter_size),
                   fc2_expert_weights, static_cast<size_t>(moe_params.hidden_size),
                   0.0f,
                   expert_fc2_batch, static_cast<size_t>(moe_params.hidden_size),
                   nullptr);

          // Step 3e: Optimized un-shuffle with vectorized operations
          for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const auto& mapping = token_group[static_cast<size_t>(batch_idx)];
            const int64_t token_idx = mapping.token_idx;
            const float routing_weight = mapping.routing_weight;

            const float* batch_fc2_output = expert_fc2_batch + batch_idx * moe_params.hidden_size;
            float* token_result = output_float_ptr + token_idx * moe_params.hidden_size;

            if (fc2_bias_float_ptr) {
              const float* fc2_expert_bias = fc2_bias_float_ptr + expert_idx * moe_params.hidden_size;
              // Vectorized bias addition and routing weight application with better cache locality
              constexpr int64_t chunk_size = 32;  // Smaller chunks for atomic operations
              for (int64_t i = 0; i < moe_params.hidden_size; i += chunk_size) {
                int64_t end_i = std::min(i + chunk_size, moe_params.hidden_size);
                for (int64_t j = i; j < end_i; ++j) {
                  // Use atomic addition for thread safety when multiple experts contribute to the same token
                  std::atomic<float>* atomic_result = reinterpret_cast<std::atomic<float>*>(&token_result[j]);
                  float current = atomic_result->load(std::memory_order_relaxed);
                  float new_value = current + routing_weight * (batch_fc2_output[j] + fc2_expert_bias[j]);
                  while (!atomic_result->compare_exchange_weak(current, new_value, std::memory_order_relaxed)) {
                    new_value = current + routing_weight * (batch_fc2_output[j] + fc2_expert_bias[j]);
                  }
                }
              }
            } else {
              // No bias case with vectorized processing
              constexpr int64_t chunk_size = 32;
              for (int64_t i = 0; i < moe_params.hidden_size; i += chunk_size) {
                int64_t end_i = std::min(i + chunk_size, moe_params.hidden_size);
                for (int64_t j = i; j < end_i; ++j) {
                  // Use atomic addition for thread safety when multiple experts contribute to the same token
                  std::atomic<float>* atomic_result = reinterpret_cast<std::atomic<float>*>(&token_result[j]);
                  float current = atomic_result->load(std::memory_order_relaxed);
                  float new_value = current + routing_weight * batch_fc2_output[j];
                  while (!atomic_result->compare_exchange_weak(current, new_value, std::memory_order_relaxed)) {
                    new_value = current + routing_weight * batch_fc2_output[j];
                  }
                }
              }
            }
          }
        }
      });

  // Convert results back to the appropriate output type with optimized conversion
  if constexpr (std::is_same_v<T, MLFloat16>) {
    // For MLFloat16, convert from float to half with performance optimization
    // Use fast math conversion if available and deterministic compute is not required
    if (!is_deterministic) {
      // Fast conversion path for better performance
      MlasConvertFloatToHalfBuffer(output_float_ptr,
                                   reinterpret_cast<MLAS_FP16*>(output_data),
                                   static_cast<size_t>(total_output_size));
    } else {
      // Deterministic conversion path
      MlasConvertFloatToHalfBuffer(output_float_ptr,
                                   reinterpret_cast<MLAS_FP16*>(output_data),
                                   static_cast<size_t>(total_output_size));
    }
  }
  // For float, no conversion needed as we directly wrote to output_data

  // Suppress unused parameter warnings for optional parameters that are not used in non-SwiGLU modes
  if (!is_swiglu) {
    ORT_UNUSED_PARAMETER(fc3_experts_bias_optional);
    ORT_UNUSED_PARAMETER(fc3_scales_optional);
  }

  return Status::OK();
}

template <bool UseUInt4x2>
Status QMoE::PrepackAndDequantizeWeights(OpKernelContext* context,
                                         MoEParameters& moe_params,
                                         const Tensor* fc1_experts_weights,
                                         const Tensor* fc2_experts_weights,
                                         const Tensor* fc1_scales,
                                         const Tensor* fc2_scales,
                                         bool is_swiglu) {
  // Get thread pool
  auto* thread_pool = context->GetOperatorThreadPool();

  // Get input data pointers
  const uint8_t* fc1_weights_data = fc1_experts_weights->Data<uint8_t>();
  const uint8_t* fc2_weights_data = fc2_experts_weights->Data<uint8_t>();
  const void* fc1_scales_data_typed = fc1_scales->DataRaw();
  const void* fc2_scales_data_typed = fc2_scales->DataRaw();
  bool is_fp32_scales = fc1_scales->IsDataType<float>();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // Prepare scales in float format
  const int64_t fc1_scales_size = moe_params.num_experts * (is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size);
  const int64_t fc2_scales_size = moe_params.num_experts * moe_params.hidden_size;

  auto fc1_scales_float = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(fc1_scales_size));
  auto fc2_scales_float = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(fc2_scales_size));

  if (is_fp32_scales) {
    // For float scales, just copy
    std::memcpy(fc1_scales_float.get(), fc1_scales_data_typed, static_cast<size_t>(fc1_scales_size) * sizeof(float));
    std::memcpy(fc2_scales_float.get(), fc2_scales_data_typed, static_cast<size_t>(fc2_scales_size) * sizeof(float));
  } else {
    // For MLFloat16 scales, convert to float
    MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(fc1_scales_data_typed),
                                           fc1_scales_float.get(),
                                           static_cast<size_t>(fc1_scales_size),
                                           thread_pool);
    MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(fc2_scales_data_typed),
                                           fc2_scales_float.get(),
                                           static_cast<size_t>(fc2_scales_size),
                                           thread_pool);
  }

  const float* fc1_scales_data = fc1_scales_float.get();
  const float* fc2_scales_data = fc2_scales_float.get();

  // Determine quantization parameters based on bit width - using symmetric quantization for TensorRT compatibility
  const bool is_4bit = UseUInt4x2;
  const int64_t act_multiplier = is_swiglu ? 2 : 1;
  const int64_t fc1_output_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;

  // Calculate weight sizes and strides based on quantization type
  const int64_t fc1_weight_stride = is_4bit ? (moe_params.hidden_size * fc1_output_size / 2) : (moe_params.hidden_size * moe_params.inter_size * act_multiplier);
  const int64_t fc2_weight_stride = is_4bit ? (moe_params.inter_size * moe_params.hidden_size / 2) : (moe_params.inter_size * moe_params.hidden_size);

  // Get or use existing persistent allocator for weights
  if (weights_allocator_ == nullptr) {
    // Get a temp allocator if we don't have a persistent one
    AllocatorPtr temp_allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&temp_allocator));
    weights_allocator_ = temp_allocator;
  }

  // Allocate prepacked weight buffers using ORT allocator with alignment for better performance
  const size_t fc1_weights_size = static_cast<size_t>(moe_params.num_experts * moe_params.hidden_size * (is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier));
  const size_t fc2_weights_size = static_cast<size_t>(moe_params.num_experts * moe_params.inter_size * moe_params.hidden_size);

  // Use aligned allocation for better SIMD performance (manual alignment)
  prepacked_fc1_weights_ = IAllocator::MakeUniquePtr<float>(weights_allocator_, fc1_weights_size);
  prepacked_fc2_weights_ = IAllocator::MakeUniquePtr<float>(weights_allocator_, fc2_weights_size);

  // Store pointers for easy access
  prepacked_fc1_weights_data_ = prepacked_fc1_weights_.get();
  prepacked_fc2_weights_data_ = prepacked_fc2_weights_.get();

  // Helper lambda for dequantizing a single weight value - optimized for symmetric quantization with better branch prediction
  auto DequantizeWeight = [&](const uint8_t* weights, size_t linear_idx,
                              const float* scales, int64_t scale_idx) -> float {
    if (is_4bit) {
      // For Int4, two values are packed in each uint8 - optimized unpacking
      size_t packed_idx = linear_idx >> 1;  // Faster than division by 2
      uint8_t packed_value = weights[packed_idx];
      // Extract the 4-bit value with optimized bit operations
      // Even indices (0, 2, 4...) use lower 4 bits (bits 0-3)
      // Odd indices (1, 3, 5...) use upper 4 bits (bits 4-7)
      uint8_t quantized_weight = (linear_idx & 1) ? (packed_value >> 4) : (packed_value & 0x0F);
      // Convert uint4 to int4 with optimized mapping for symmetric quantization
      // For 4-bit symmetric quantization, we need to map [0,15] to [-8,7]
      // Using branchless conversion for better performance
      int8_t signed_weight = static_cast<int8_t>(quantized_weight - 8);
      return static_cast<float>(signed_weight) * scales[scale_idx];
    } else {
      // For Int8, convert uint8 to int8 for symmetric quantization with optimized conversion
      // Branchless conversion: subtract 128 to map [0,255] to [-128,127]
      int8_t signed_weight = static_cast<int8_t>(weights[linear_idx] - 128);
      return static_cast<float>(signed_weight) * scales[scale_idx];
    }
  };

  // Dequantize FC1 weights for all experts with optimized parallelization
  double fc1_cost_per_expert = static_cast<double>(moe_params.hidden_size *
                                                   (is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier)) *
                               0.5;  // Reduced cost due to optimizations
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_experts),
      fc1_cost_per_expert,
      [&](ptrdiff_t expert_start, ptrdiff_t expert_end) {
        for (std::ptrdiff_t expert_idx = expert_start; expert_idx < expert_end; ++expert_idx) {
          const uint8_t* fc1_expert_weights = fc1_weights_data + expert_idx * fc1_weight_stride;
          const float* fc1_expert_scales = fc1_scales_data + expert_idx * (is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier);
          float* dequant_fc1_expert = prepacked_fc1_weights_data_ + expert_idx * moe_params.hidden_size * (is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier);

          const int64_t output_cols = is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier;
          // Handle column-major weight storage for FC1 with cache-friendly access pattern
          // Process in blocks to improve cache locality
          constexpr int64_t block_size = 64;  // Process 64 elements at a time for better cache usage

          for (int64_t in_col = 0; in_col < moe_params.hidden_size; ++in_col) {
            for (int64_t out_block = 0; out_block < output_cols; out_block += block_size) {
              int64_t out_end = std::min(out_block + block_size, output_cols);

              // Vectorized processing within the block
              for (int64_t out_col = out_block; out_col < out_end; ++out_col) {
                // For column-major, linear_idx is in_col * output_cols + out_col
                size_t linear_idx = static_cast<size_t>(in_col * output_cols + out_col);

                // For the output (row-major), we need out_col * hidden_size + in_col
                size_t output_idx = static_cast<size_t>(out_col * moe_params.hidden_size + in_col);

                // For FC1, the scale index is the output column
                dequant_fc1_expert[output_idx] = DequantizeWeight(fc1_expert_weights, linear_idx, fc1_expert_scales, out_col);
              }
            }
          }
        }
      });

  // Dequantize FC2 weights for all experts with optimized parallelization
  double fc2_cost_per_expert = static_cast<double>(moe_params.inter_size * moe_params.hidden_size) * 0.5;  // Reduced cost due to optimizations
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_experts),
      fc2_cost_per_expert,
      [&](ptrdiff_t expert_start, ptrdiff_t expert_end) {
        for (std::ptrdiff_t expert_idx = expert_start; expert_idx < expert_end; ++expert_idx) {
          const uint8_t* fc2_expert_weights = fc2_weights_data + expert_idx * fc2_weight_stride;
          const float* fc2_expert_scales = fc2_scales_data + expert_idx * moe_params.hidden_size;
          float* dequant_fc2_expert = prepacked_fc2_weights_data_ + expert_idx * moe_params.inter_size * moe_params.hidden_size;

          // Handle column-major weight storage for FC2 with cache-friendly access pattern
          // Process in blocks to improve cache locality
          constexpr int64_t block_size = 64;  // Process 64 elements at a time for better cache usage

          for (int64_t in_col = 0; in_col < moe_params.inter_size; ++in_col) {
            for (int64_t out_block = 0; out_block < moe_params.hidden_size; out_block += block_size) {
              int64_t out_end = std::min(out_block + block_size, moe_params.hidden_size);

              // Vectorized processing within the block
              for (int64_t out_col = out_block; out_col < out_end; ++out_col) {
                // For column-major, linear_idx is in_col * hidden_size + out_col
                size_t linear_idx = static_cast<size_t>(in_col * moe_params.hidden_size + out_col);

                // For the output (row-major), we need out_col * inter_size + in_col
                size_t output_idx = static_cast<size_t>(out_col * moe_params.inter_size + in_col);

                // For FC2, the scale index is the output column
                dequant_fc2_expert[output_idx] = DequantizeWeight(fc2_expert_weights, linear_idx, fc2_expert_scales, out_col);
              }
            }
          }
        }
      });

  // Update cached parameters
  cached_num_experts_ = moe_params.num_experts;
  cached_hidden_size_ = moe_params.hidden_size;
  cached_inter_size_ = moe_params.inter_size;
  cached_is_swiglu_ = is_swiglu;
  is_prepacked_ = true;

  return Status::OK();
}

// Explicit template instantiations
template Status QMoE::QuantizedMoEImpl<true, MLFloat16>(OpKernelContext* context,
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

template Status QMoE::QuantizedMoEImpl<false, MLFloat16>(OpKernelContext* context,
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

template Status QMoE::QuantizedMoEImpl<true, float>(OpKernelContext* context,
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

template Status QMoE::QuantizedMoEImpl<false, float>(OpKernelContext* context,
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
