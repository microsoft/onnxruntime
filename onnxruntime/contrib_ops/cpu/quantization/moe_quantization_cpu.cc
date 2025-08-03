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
    Status status = const_cast<QMoE*>(this)->PrepackAndDequantizeWeights<UseUInt4x2>(
        context, moe_params, fc1_experts_weights, fc2_experts_weights,
        fc1_scales, fc2_scales, is_swiglu);
    ORT_RETURN_IF_ERROR(status);
  }
  // Get thread pool
  auto* thread_pool = context->GetOperatorThreadPool();

  // Get input data pointers
  const T* input_data = input->Data<T>();
  const T* router_probs_data = router_probs->Data<T>();
  const T* fc1_bias_data = fc1_experts_bias_optional ? fc1_experts_bias_optional->Data<T>() : nullptr;
  const T* fc2_bias_data = fc2_experts_bias_optional ? fc2_experts_bias_optional->Data<T>() : nullptr;

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // Use arena allocator for better memory management and reduced fragmentation
  // This is especially beneficial for repeated kernel invocations
  AllocatorPtr arena_allocator;
  if (context->GetUseDeterministicCompute()) {
    // For deterministic compute, use the standard temp allocator
    arena_allocator = allocator;
  } else {
    // Try to get arena allocator for better performance
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&arena_allocator));
  }

  const int64_t num_threads = std::min<int64_t>(
      static_cast<int64_t>(concurrency::ThreadPool::DegreeOfParallelism(thread_pool)),
      moe_params.num_rows);

  const int64_t total_output_size = moe_params.num_rows * moe_params.hidden_size;
  std::fill_n(output_data, total_output_size, MLFloat16(0.0f));

  // Using prepacked weights - no need to convert scales

  // Optimized memory allocation: Use a single large buffer for all thread-local storage
  // This reduces allocation overhead and improves cache locality
  const int64_t fc1_buffer_size_per_thread = moe_params.inter_size * (is_swiglu ? 2 : 1);
  const int64_t fc2_buffer_size_per_thread = moe_params.hidden_size;
  const int64_t total_buffer_size_per_thread = fc1_buffer_size_per_thread + fc2_buffer_size_per_thread;

  // Single allocation for all thread buffers with proper alignment
  auto thread_buffers = IAllocator::MakeUniquePtr<float>(arena_allocator, static_cast<size_t>(num_threads * total_buffer_size_per_thread)); // Optimized for cache locality

  // Set up output buffer with memory optimization
  IAllocatorUniquePtr<float> output_float;
  float* output_float_ptr = nullptr;

  if constexpr (std::is_same_v<T, MLFloat16>) {
    // For MLFloat16, we need a separate float buffer
    // Use arena allocator for better memory management
    output_float = IAllocator::MakeUniquePtr<float>(arena_allocator, static_cast<size_t>(total_output_size));
    output_float_ptr = output_float.get();
  } else {
    // For float, we can write directly to output_data
    output_float = IAllocatorUniquePtr<float>(output_data, [](float*) {});
    output_float_ptr = output_data;
  }

  // Initialize output to zeros
  std::fill_n(output_float_ptr, total_output_size, 0.0f);

  // Prepare float buffers for input data and biases
  IAllocatorUniquePtr<float> input_float;
  IAllocatorUniquePtr<float> router_probs_float;

  // Pointers for easier access
  float* input_float_ptr = nullptr;
  float* router_probs_float_ptr = nullptr;

  // Pre-convert bias tensors to float (if they exist) with optimized memory usage
  const int64_t fc1_bias_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
  const int64_t fc2_bias_size = moe_params.hidden_size;

  // Allocate buffers for converted biases using arena allocator
  IAllocatorUniquePtr<float> fc1_bias_float;
  IAllocatorUniquePtr<float> fc2_bias_float;

  if (fc1_bias_data) {
    fc1_bias_float = IAllocator::MakeUniquePtr<float>(arena_allocator, static_cast<size_t>(moe_params.num_experts * fc1_bias_size));
  }

  if (fc2_bias_data) {
    fc2_bias_float = IAllocator::MakeUniquePtr<float>(arena_allocator, static_cast<size_t>(moe_params.num_experts * fc2_bias_size));
  }

  // Convert input and router_probs based on type with memory optimization
  if constexpr (std::is_same_v<T, MLFloat16>) {
    // For MLFloat16, convert to float - use arena allocator
    input_float = IAllocator::MakeUniquePtr<float>(arena_allocator, static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size));
    router_probs_float = IAllocator::MakeUniquePtr<float>(arena_allocator, static_cast<size_t>(moe_params.num_rows * moe_params.num_experts));

    input_float_ptr = input_float.get();
    router_probs_float_ptr = router_probs_float.get();

    // Convert MLFloat16 to float with optimized parallel conversion
    MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(input_data),
                                           input_float_ptr,
                                           static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size),
                                           thread_pool);

    MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(router_probs_data),
                                           router_probs_float_ptr,
                                           static_cast<size_t>(moe_params.num_rows * moe_params.num_experts),
                                           thread_pool);

    // Convert biases to float once (if they exist) with optimized conversion
    if (fc1_bias_data) {
      MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(fc1_bias_data),
                                             fc1_bias_float.get(),
                                             static_cast<size_t>(moe_params.num_experts * fc1_bias_size),
                                             thread_pool);
    }

    if (fc2_bias_data) {
      MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(fc2_bias_data),
                                             fc2_bias_float.get(),
                                             static_cast<size_t>(moe_params.num_experts * fc2_bias_size),
                                             thread_pool);
    }
  } else {
    // For float, point to original input and router_probs directly instead of copying
    input_float = IAllocatorUniquePtr<float>(const_cast<float*>(input_data), [](float*) {});
    router_probs_float = IAllocatorUniquePtr<float>(const_cast<float*>(router_probs_data), [](float*) {});

    // Set pointers to the original data
    input_float_ptr = const_cast<float*>(input_data);
    router_probs_float_ptr = const_cast<float*>(router_probs_data);

    // For float, just point to the original bias data directly without copying
    // No need to allocate or copy, just reuse the original pointers
    if (fc1_bias_data) {
      // Release previously allocated memory if any
      fc1_bias_float.reset();
      // Direct pointer to original data
      fc1_bias_float = IAllocatorUniquePtr<float>(const_cast<float*>(fc1_bias_data), [](float*) {});
    }

    if (fc2_bias_data) {
      // Release previously allocated memory if any
      fc2_bias_float.reset();
      // Direct pointer to original data
      fc2_bias_float = IAllocatorUniquePtr<float>(const_cast<float*>(fc2_bias_data), [](float*) {});
    }
  }

  // No need to initialize thread results - using direct output buffer

  // Determine activation related parameters
  const bool is_4bit = UseUInt4x2;
  const int64_t act_multiplier = is_swiglu ? 2 : 1;
  const int64_t fc1_output_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;

  // Use prepacked dequantized weights - no need to dequantize here
  const float* dequant_fc1_weights = prepacked_fc1_weights_data_;
  const float* dequant_fc2_weights = prepacked_fc2_weights_data_;

  // Process tokens in parallel with improved cost modeling for better load balancing
  // The cost model now accounts for expert filtering and vectorization improvements
  double cost_per_token = static_cast<double>(moe_params.hidden_size * moe_params.inter_size * moe_params.num_experts * 0.8);  // Reduced due to expert filtering
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_rows),
      cost_per_token,
      [&](ptrdiff_t start_token, ptrdiff_t end_token) {
        // Improved thread ID calculation for better cache locality
        const int64_t thread_id = start_token / std::max<int64_t>(1, (moe_params.num_rows + num_threads - 1) / num_threads);

        // Optimized buffer access: use single buffer with proper offsets for better cache locality
        float* thread_base_buffer = thread_buffers.get() + thread_id * total_buffer_size_per_thread;
        float* thread_fc1_output = thread_base_buffer;  // FC1 buffer at start
        float* thread_fc2_output = thread_base_buffer + fc1_buffer_size_per_thread;  // FC2 buffer after FC1

        // Process each token in this thread's range
        for (std::ptrdiff_t token_idx = start_token; token_idx < end_token; ++token_idx) {
          const float* token_input = input_float_ptr + static_cast<int64_t>(SafeInt<int64_t>(token_idx)) * moe_params.hidden_size;
          float* token_result = output_float_ptr + static_cast<int64_t>(SafeInt<int64_t>(token_idx)) * moe_params.hidden_size;

          // Prefetch next token's input for better cache performance
          if (token_idx + 1 < end_token) {
            const float* next_token_input = input_float_ptr + static_cast<int64_t>(SafeInt<int64_t>(token_idx + 1)) * moe_params.hidden_size;
            // Use compiler intrinsic for prefetching if available
            #ifdef _MSC_VER
            _mm_prefetch(reinterpret_cast<const char*>(next_token_input), _MM_HINT_T0);
            #elif defined(__GNUC__)
            __builtin_prefetch(next_token_input, 0, 3);
            #endif
          }

          // Process all experts for this token
          for (std::ptrdiff_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
            float routing_weight = router_probs_float_ptr[static_cast<int64_t>(SafeInt<int64_t>(token_idx)) * moe_params.num_experts + static_cast<int64_t>(SafeInt<int64_t>(expert_idx))];
            // Increased threshold for better performance - skip more low-impact experts
            if (routing_weight <= 1e-4f) continue;  // Skip experts with negligible routing weight

            // FC1: input -> intermediate using pre-dequantized weights + MLAS SGEMM
            const int64_t fc1_weight_offset = is_4bit ? (static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.hidden_size * fc1_output_size) : (static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.hidden_size * moe_params.inter_size * act_multiplier);
            const float* fc1_expert_weights = dequant_fc1_weights + fc1_weight_offset;

            // Bias size is always equal to output size (fc1_output_size), regardless of bit width
            const int64_t fc1_bias_size = fc1_output_size;

            // Use MLAS SGEMM for FC1 with optimized parameters
            MLAS_SGEMM_DATA_PARAMS fc1_params;
            fc1_params.A = token_input;
            fc1_params.lda = static_cast<size_t>(moe_params.hidden_size);
            fc1_params.B = fc1_expert_weights;
            // For column-major storage, leading dimension is the number of rows (fc1_output_size)
            fc1_params.ldb = static_cast<size_t>(fc1_output_size);
            fc1_params.C = thread_fc1_output;
            fc1_params.ldc = static_cast<size_t>(fc1_bias_size);
            fc1_params.alpha = 1.0f;
            fc1_params.beta = 0.0f;

            // Use CblasTrans for weights (B matrix) since they are stored in column-major format
            // Single-threaded GEMM for better cache utilization in parallel token processing
            MlasGemm(CblasNoTrans, CblasTrans, 1, static_cast<size_t>(fc1_bias_size), static_cast<size_t>(moe_params.hidden_size), fc1_params, nullptr);

            // Handle different activation types
            if (is_swiglu) {
              // Add bias if present
              if (fc1_bias_data) {
                // Use the pre-converted float bias data
                const float* fc1_expert_bias_float = fc1_bias_float.get() + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * fc1_bias_size;
                // Vectorized bias addition for better performance
                #pragma omp simd
                for (int64_t i = 0; i < fc1_bias_size; ++i) {
                  thread_fc1_output[i] += fc1_expert_bias_float[i];
                }
              }
              // Always use interleaved format for SwiGLU activation
              contrib::ApplySwiGLUActivation(thread_fc1_output, moe_params.inter_size, true);
            } else {
              // Standard activation (non-SwiGLU)
              if (fc1_bias_data) {
                // Use the pre-converted float bias data
                const float* fc1_expert_bias_float = fc1_bias_float.get() + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.inter_size;
                // Vectorized bias addition and activation for better performance
                #pragma omp simd
                for (int64_t i = 0; i < moe_params.inter_size; ++i) {
                  thread_fc1_output[i] += fc1_expert_bias_float[i];
                  thread_fc1_output[i] = ApplyActivation(thread_fc1_output[i], activation_type_);
                }
              } else {
                // Vectorized activation for better performance
                #pragma omp simd
                for (int64_t i = 0; i < moe_params.inter_size; ++i) {
                  thread_fc1_output[i] = ApplyActivation(thread_fc1_output[i], activation_type_);
                }
              }
            }

            // FC2: intermediate -> output using pre-dequantized weights + MLAS SGEMM
            const float* fc2_expert_weights = dequant_fc2_weights + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.inter_size * moe_params.hidden_size;

            // Use MLAS SGEMM for FC2 with optimized parameters
            MLAS_SGEMM_DATA_PARAMS fc2_params;
            fc2_params.A = thread_fc1_output;
            fc2_params.lda = static_cast<size_t>(moe_params.inter_size);
            fc2_params.B = fc2_expert_weights;
            // For column-major storage, leading dimension is the number of rows (moe_params.hidden_size)
            fc2_params.ldb = static_cast<size_t>(moe_params.hidden_size);
            fc2_params.C = thread_fc2_output;
            fc2_params.ldc = static_cast<size_t>(moe_params.hidden_size);
            fc2_params.alpha = 1.0f;
            fc2_params.beta = 0.0f;

            // Use CblasTrans for weights (B matrix) since they are stored in column-major format
            // Single-threaded GEMM for better cache utilization in parallel token processing
            MlasGemm(CblasNoTrans, CblasTrans, 1, static_cast<size_t>(moe_params.hidden_size), static_cast<size_t>(moe_params.inter_size), fc2_params, nullptr);

            // Add bias, apply routing weight, and accumulate to final result
            if (fc2_bias_data) {
              // Use the pre-converted float bias data
              const float* fc2_expert_bias_float = fc2_bias_float.get() + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.hidden_size;
              // Vectorized bias addition and routing weight application for better performance
              #pragma omp simd
              for (int64_t i = 0; i < moe_params.hidden_size; ++i) {
                token_result[i] += routing_weight * (thread_fc2_output[i] + fc2_expert_bias_float[i]);
              }
            } else {
              // Vectorized routing weight application for better performance
              #pragma omp simd
              for (int64_t i = 0; i < moe_params.hidden_size; ++i) {
                token_result[i] += routing_weight * thread_fc2_output[i];
              }
            }
          }
        }
      });

  // No need for accumulation since threads write directly to output_float

  // Convert results back to the appropriate output type, if needed
  if constexpr (std::is_same_v<T, MLFloat16>) {
    // For MLFloat16, convert from float to half
    // Note: MlasConvertFloatToHalfBuffer is already optimized internally
    MlasConvertFloatToHalfBuffer(output_float_ptr,
                                reinterpret_cast<MLAS_FP16*>(output_data),
                                static_cast<size_t>(total_output_size));
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

  // Get or create a persistent allocator for weights with memory optimization
  if (weights_allocator_ == nullptr) {
    // Try to get a more efficient allocator for weight storage
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

  // Helper lambda for dequantizing a single weight value - updated for symmetric quantization
  auto DequantizeWeight = [&](const uint8_t* weights, size_t linear_idx,
                              const float* scales, int64_t scale_idx) -> float {
    if (is_4bit) {
      // For Int4, two values are packed in each uint8
      size_t packed_idx = linear_idx / 2;
      uint8_t packed_value = weights[packed_idx];
      // Extract the 4-bit value correctly based on byte packing
      // For 4-bit quantized weights, each byte contains two 4-bit values
      // Even indices (0, 2, 4...) use lower 4 bits (bits 0-3)
      // Odd indices (1, 3, 5...) use upper 4 bits (bits 4-7)
      uint8_t quantized_weight = (linear_idx % 2 == 0) ? (packed_value & 0x0F) : ((packed_value >> 4) & 0x0F);
      // Convert uint4 to int4 with proper mapping for symmetric quantization
      // For 4-bit symmetric quantization, we need to map [0,15] to [-8,7]
      // Values [0,7] map to [0,7], values [8,15] map to [-8,-1]
      int8_t signed_weight;
      if (quantized_weight < 8) {
        signed_weight = static_cast<int8_t>(quantized_weight);
      } else {
        signed_weight = static_cast<int8_t>(quantized_weight - 16);  // Map [8, 15] to [-8, -1]
      }
      return static_cast<float>(signed_weight) * scales[scale_idx];
    } else {
      // For Int8, convert uint8 to int8 for symmetric quantization
      int8_t signed_weight = static_cast<int8_t>(weights[linear_idx]);
      return static_cast<float>(signed_weight) * scales[scale_idx];
    }
  };

  // Dequantize FC1 weights for all experts
  double fc1_cost_per_expert = static_cast<double>(moe_params.hidden_size *
    (is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier));
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_experts),
      fc1_cost_per_expert,
      [&](ptrdiff_t expert_start, ptrdiff_t expert_end) {
        for (std::ptrdiff_t expert_idx = expert_start; expert_idx < expert_end; ++expert_idx) {
          const uint8_t* fc1_expert_weights = fc1_weights_data + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * fc1_weight_stride;
          const float* fc1_expert_scales = fc1_scales_data + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * (is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier);
          float* dequant_fc1_expert = prepacked_fc1_weights_data_ + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.hidden_size * (is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier);

          const int64_t output_cols = is_4bit ? fc1_output_size : moe_params.inter_size * act_multiplier;
          // Handle column-major weight storage for FC1
          for (int64_t in_col = 0; in_col < moe_params.hidden_size; ++in_col) {
            for (int64_t out_col = 0; out_col < output_cols; ++out_col) {
              // For column-major, linear_idx is in_col * output_cols + out_col
              size_t linear_idx = static_cast<size_t>(in_col * output_cols + out_col);

              // For the output (row-major), we need out_col * hidden_size + in_col
              size_t output_idx = static_cast<size_t>(out_col * moe_params.hidden_size + in_col);

              // For FC1, the scale index is the output column
              dequant_fc1_expert[output_idx] = DequantizeWeight(fc1_expert_weights, linear_idx, fc1_expert_scales, out_col);
            }
          }
        }
      });

  // Dequantize FC2 weights for all experts
  double fc2_cost_per_expert = static_cast<double>(moe_params.inter_size * moe_params.hidden_size);
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_experts),
      fc2_cost_per_expert,
      [&](ptrdiff_t expert_start, ptrdiff_t expert_end) {
        for (std::ptrdiff_t expert_idx = expert_start; expert_idx < expert_end; ++expert_idx) {
          const uint8_t* fc2_expert_weights = fc2_weights_data + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * fc2_weight_stride;
          const float* fc2_expert_scales = fc2_scales_data + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.hidden_size;
          float* dequant_fc2_expert = prepacked_fc2_weights_data_ + static_cast<int64_t>(SafeInt<int64_t>(expert_idx)) * moe_params.inter_size * moe_params.hidden_size;

          // Handle column-major weight storage for FC2
          for (int64_t in_col = 0; in_col < moe_params.inter_size; ++in_col) {
            for (int64_t out_col = 0; out_col < moe_params.hidden_size; ++out_col) {
              // For column-major, linear_idx is in_col * hidden_size + out_col
              size_t linear_idx = static_cast<size_t>(in_col * moe_params.hidden_size + out_col);

              // For the output (row-major), we need out_col * inter_size + in_col
              size_t output_idx = static_cast<size_t>(out_col * moe_params.inter_size + in_col);

              // For FC2, the scale index is the output column
              dequant_fc2_expert[output_idx] = DequantizeWeight(fc2_expert_weights, linear_idx, fc2_expert_scales, out_col);
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
