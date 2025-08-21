// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/quantization/moe_quantization.h"
#include "core/providers/cuda/cuda_type_conversion.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
template <typename T, bool use_quint4x2>
struct ToCudaTypeWrapper : public ToCudaType<T> {};

template <>
struct ToCudaTypeWrapper<uint8_t, false> {
  using MappedType = uint8_t;
};

template <>
struct ToCudaTypeWrapper<uint8_t, true> {
  using MappedType = cutlass::uint4b_t;
};

}  // anonymous namespace

template <typename T>
QMoE<T>::QMoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
              "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);
}

template <typename T>
template <typename CudaWeightT>
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
                                 const Tensor* fc3_scales_optional,
                                 const cudaDeviceProp& device_prop) const {
  auto stream = context->GetComputeStream();

  const int sm = device_prop.major * 10 + device_prop.minor;

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  using CudaT = typename OrtToCudaType<T>::type;

  ort_fastertransformer::CutlassMoeFCRunner<CudaT, CudaWeightT> moe_runner(sm,
                                                                           activation_type_,
                                                                           fc3_experts_weights_optional != nullptr,
                                                                           normalize_routing_weights_,
                                                                           use_sparse_mixer_);

  size_t ws_size = moe_runner.getWorkspaceSize(
      static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.hidden_size),
      static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(moe_params.num_experts),
      static_cast<size_t>(k_));
  size_t fc2_output_size = k_ * moe_params.num_rows * moe_params.hidden_size * sizeof(CudaT);
  size_t expert_scales_size = k_ * moe_params.num_rows * sizeof(CudaT);
  size_t expanded_source_row_to_expanded_dest_row_size = k_ * moe_params.num_rows * sizeof(int);
  size_t expert_for_source_row_size = k_ * moe_params.num_rows * sizeof(int);

  IAllocatorUniquePtr<void> work_space = IAllocator::MakeUniquePtr<void>(allocator, ws_size, false, stream);
  IAllocatorUniquePtr<void> fc2_output = IAllocator::MakeUniquePtr<void>(allocator, fc2_output_size, false, stream);
  IAllocatorUniquePtr<void> expert_scales =
      IAllocator::MakeUniquePtr<void>(allocator, expert_scales_size, false, stream);
  IAllocatorUniquePtr<void> expanded_source_row_to_expanded_dest_row =
      IAllocator::MakeUniquePtr<void>(allocator, expanded_source_row_to_expanded_dest_row_size, false, stream);
  IAllocatorUniquePtr<void> expert_for_source_row =
      IAllocator::MakeUniquePtr<void>(allocator, expert_for_source_row_size, false, stream);

  moe_runner.run_moe_fc(
      reinterpret_cast<const CudaT*>(input->template Data<T>()),
      reinterpret_cast<const CudaT*>(router_probs->template Data<T>()),
      reinterpret_cast<const CudaWeightT*>(fc1_experts_weights->DataRaw()),
      fc1_scales == nullptr ? nullptr : reinterpret_cast<const CudaT*>(fc1_scales->template Data<T>()),
      fc1_experts_bias_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(fc1_experts_bias_optional->template Data<T>()),
      activation_type_,
      fc3_experts_weights_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaWeightT*>(fc3_experts_weights_optional->DataRaw()),
      fc3_scales_optional == nullptr ? nullptr
                                     : reinterpret_cast<const CudaT*>(fc3_scales_optional->template Data<T>()),
      fc3_experts_bias_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(fc3_experts_bias_optional->template Data<T>()),
      reinterpret_cast<const CudaWeightT*>(fc2_experts_weights->DataRaw()),
      fc2_scales == nullptr ? nullptr : reinterpret_cast<const CudaT*>(fc2_scales->template Data<T>()),
      static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.hidden_size),
      static_cast<int>(moe_params.inter_size), static_cast<int>(moe_params.num_experts),
      static_cast<int>(moe_params.local_num_experts), 0 /*local_experts_start_index_ used in sharded MoE*/,
      static_cast<int>(k_), reinterpret_cast<char*>(work_space.get()), reinterpret_cast<CudaT*>(fc2_output.get()),
      reinterpret_cast<CudaT*>(expert_scales.get()),
      reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
      reinterpret_cast<int*>(expert_for_source_row.get()), Stream(context));

  // DEBUG: Print raw router logits for first few rows (matching CPU)
  if (moe_params.num_rows > 0 && router_probs != nullptr) {
    cudaStreamSynchronize(Stream(context));

    const int debug_rows = std::min(static_cast<int>(moe_params.num_rows), 3);
    const int debug_experts = std::min(static_cast<int>(moe_params.num_experts), 8);

    // Safety check: Ensure router_probs tensor is large enough
    size_t router_probs_total_size = router_probs->Shape().Size();
    size_t required_size = static_cast<size_t>(debug_rows) * debug_experts;

    if (required_size <= router_probs_total_size) {
      std::vector<T> router_debug(required_size);

      cudaError_t err_router = cudaMemcpy(router_debug.data(),
                                          router_probs->template Data<T>(),
                                          required_size * sizeof(T),
                                          cudaMemcpyDeviceToHost);

      if (err_router == cudaSuccess) {
        for (int row = 0; row < debug_rows; ++row) {
          printf("[CUDA DEBUG] Row %d raw router logits: ", row);
          for (int e = 0; e < debug_experts; ++e) {
            printf("%.6f ", static_cast<float>(router_debug[row * debug_experts + e]));
          }
          printf("...\n");
        }
      } else {
        printf("[CUDA DEBUG] Router logits: cudaMemcpy failed (error %d)\n", err_router);
      }
    } else {
      printf("[CUDA DEBUG] Router logits: skipped (tensor too small: %zu vs required %zu)\n",
             router_probs_total_size, required_size);
    }
  }

  // DEBUG: Print expert selection results (matching CPU debug output)
  if (moe_params.num_rows > 0 && k_ > 0) {
    const int debug_rows = std::min(static_cast<int>(moe_params.num_rows), 3);
    const int debug_k = std::min(static_cast<int>(k_), 8);
    const size_t required_elements = static_cast<size_t>(debug_rows) * debug_k;

    // Safety check: Ensure we don't exceed allocated workspace sizes
    if (required_elements > 0) {
      std::vector<int> cuda_experts(required_elements);
      std::vector<T> cuda_scores(required_elements);

      cudaError_t err_experts = cudaMemcpy(cuda_experts.data(), expert_for_source_row.get(),
                                           required_elements * sizeof(int), cudaMemcpyDeviceToHost);
      cudaError_t err_scores = cudaMemcpy(cuda_scores.data(), expert_scales.get(),
                                          required_elements * sizeof(T), cudaMemcpyDeviceToHost);

      if (err_experts == cudaSuccess && err_scores == cudaSuccess) {
        for (int row = 0; row < debug_rows; ++row) {
          printf("[CUDA DEBUG] Row %d top-%d experts: ", row, debug_k);
          for (int i = 0; i < debug_k; ++i) {
            size_t idx = static_cast<size_t>(row) * debug_k + i;
            if (idx < required_elements) {
              int expert_idx = cuda_experts[idx];
              float score = static_cast<float>(cuda_scores[idx]);
              printf("(E%d:%.6f) ", expert_idx, score);
            }
          }
          printf("\n");

          printf("[CUDA DEBUG] Row %d Expert final weights: ", row);
          for (int i = 0; i < debug_k; ++i) {
            size_t idx = static_cast<size_t>(row) * debug_k + i;
            if (idx < required_elements) {
              int expert_idx = cuda_experts[idx];
              float score = static_cast<float>(cuda_scores[idx]);
              printf("Expert %d final weight: %.6f ", expert_idx, score);
            }
          }
          printf("\n");
        }
      } else {
        printf("[CUDA DEBUG] Expert selection: cudaMemcpy failed (experts: %d, scores: %d)\n",
               err_experts, err_scores);
      }
    }
  }

  // DEBUG: Print input activations (matching CPU debug)
  if (moe_params.num_rows > 0 && moe_params.hidden_size > 0 && input != nullptr) {
    const int debug_elements = std::min(static_cast<int>(moe_params.hidden_size), 8);

    // Safety check: Ensure input tensor is large enough
    size_t input_total_size = input->Shape().Size();

    if (static_cast<size_t>(debug_elements) <= input_total_size) {
      std::vector<T> input_debug(debug_elements);

      cudaError_t err_input = cudaMemcpy(input_debug.data(), input->template Data<T>(),
                                         debug_elements * sizeof(T), cudaMemcpyDeviceToHost);
      if (err_input == cudaSuccess) {
        printf("[CUDA DEBUG] Input activations (first token, first 8): ");
        for (int i = 0; i < debug_elements; ++i) {
          printf("%.6f ", static_cast<float>(input_debug[i]));
        }
        printf("...\n");
      } else {
        printf("[CUDA DEBUG] Input activations: cudaMemcpy failed (error %d)\n", err_input);
      }
    } else {
      printf("[CUDA DEBUG] Input activations: skipped (tensor too small: %zu vs required %d)\n",
             input_total_size, debug_elements);
    }
  }

  // DEBUG: Print FC1 and FC2 scales for SELECTED experts (matching CPU debug)
  if (fc1_scales && fc2_scales && moe_params.num_experts > 0) {
    cudaStreamSynchronize(Stream(context));

    std::vector<int> cuda_experts(4);
    std::vector<T> cuda_scores(4);

    // Safe copy with error checking
    cudaError_t err1 = cudaMemcpy(cuda_experts.data(), expert_for_source_row.get(), 4 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaError_t err2 = cudaMemcpy(cuda_scores.data(), expert_scales.get(), 4 * sizeof(T), cudaMemcpyDeviceToHost);

    if (err1 == cudaSuccess && err2 == cudaSuccess) {
      printf(
          "CUDA expert order: [%d, %d, %d, %d]",
          cuda_experts[0],
          cuda_experts[1], cuda_experts[2], cuda_experts[3]);
      printf(
          "CUDA expert scores: [%.6f, %.6f, %.6f, %.6f]",
          static_cast<float>(cuda_scores[0]),
          static_cast<float>(cuda_scores[1]),
          static_cast<float>(cuda_scores[2]), static_cast<float>(cuda_scores[3]));

      // Print scales for the selected experts
      const int debug_elements = std::min(static_cast<int>(moe_params.inter_size * 2), 8);
      size_t fc1_scales_total_size = fc1_scales->Shape().Size();
      size_t fc2_scales_total_size = fc2_scales->Shape().Size();

      for (int i = 0; i < 4; ++i) {
        int expert_id = cuda_experts[i];

        // FC1 scales for selected expert
        size_t fc1_expert_offset = static_cast<size_t>(expert_id) * moe_params.inter_size * 2;
        size_t fc1_required_size = fc1_expert_offset + debug_elements;

        if (fc1_required_size <= fc1_scales_total_size) {
          std::vector<T> fc1_scales_debug(debug_elements);
          cudaError_t err_fc1 = cudaMemcpy(fc1_scales_debug.data(),
                                           fc1_scales->template Data<T>() + fc1_expert_offset,
                                           debug_elements * sizeof(T), cudaMemcpyDeviceToHost);
          if (err_fc1 == cudaSuccess) {
            printf("[CUDA DEBUG] Expert %d FC1 scales (first 8): ", expert_id);
            for (int j = 0; j < debug_elements; ++j) {
              printf("%.6f ", static_cast<float>(fc1_scales_debug[j]));
            }
            printf("...");
          }
        }

        // FC2 scales for selected expert
        size_t fc2_expert_offset = static_cast<size_t>(expert_id) * moe_params.hidden_size;
        size_t fc2_required_size = fc2_expert_offset + std::min(static_cast<size_t>(moe_params.hidden_size), 8UL);

        if (fc2_required_size <= fc2_scales_total_size) {
          std::vector<T> fc2_scales_debug(std::min(static_cast<size_t>(moe_params.hidden_size), 8UL));
          cudaError_t err_fc2 = cudaMemcpy(fc2_scales_debug.data(),
                                           fc2_scales->template Data<T>() + fc2_expert_offset,
                                           fc2_scales_debug.size() * sizeof(T), cudaMemcpyDeviceToHost);
          if (err_fc2 == cudaSuccess) {
            printf("[CUDA DEBUG] Expert %d FC2 scales (first 8): ", expert_id);
            for (size_t j = 0; j < fc2_scales_debug.size(); ++j) {
              printf("%.6f ", static_cast<float>(fc2_scales_debug[j]));
            }
            printf("...");
          }
        }
      }
    } else {
      printf("CUDA debug: Memory copy failed (err1=%d, err2=%d)", err1, err2);
    }
  }

  // Debug: Print CUDA expert selection for first row (with safety checks)
  if (moe_params.num_rows > 0 && k_ >= 4) {
    // Synchronize stream to ensure kernel completion
    cudaStreamSynchronize(Stream(context));

    std::vector<int> cuda_experts(4);
    std::vector<T> cuda_scores(4);

    // Safe copy with error checking
    cudaError_t err1 = cudaMemcpy(cuda_experts.data(), expert_for_source_row.get(), 4 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaError_t err2 = cudaMemcpy(cuda_scores.data(), expert_scales.get(), 4 * sizeof(T), cudaMemcpyDeviceToHost);

    if (err1 == cudaSuccess && err2 == cudaSuccess) {
      printf("CUDA expert order: [%d, %d, %d, %d]\n",
             cuda_experts[0], cuda_experts[1], cuda_experts[2], cuda_experts[3]);
      printf("CUDA expert scores: [%.6f, %.6f, %.6f, %.6f]\n",
             static_cast<float>(cuda_scores[0]), static_cast<float>(cuda_scores[1]),
             static_cast<float>(cuda_scores[2]), static_cast<float>(cuda_scores[3]));
    } else {
      printf("CUDA debug: Memory copy failed (err1=%d, err2=%d)\n", err1, err2);
    }
  }

  // ===== COMPREHENSIVE WEIGHT LAYOUT DEBUGGING =====
  // Debug quantized weight memory layout and compare with CPU expectations
  if (fc1_experts_weights && fc2_experts_weights && fc1_scales && fc2_scales) {
    cudaStreamSynchronize(Stream(context));

    // Get selected experts
    std::vector<int> cuda_experts(4);
    cudaError_t err_experts = cudaMemcpy(cuda_experts.data(), expert_for_source_row.get(), 4 * sizeof(int), cudaMemcpyDeviceToHost);

    if (err_experts == cudaSuccess) {
      printf("\n[CUDA WEIGHT DEBUG] ===== WEIGHT LAYOUT ANALYSIS =====\n");
      printf("[CUDA WEIGHT DEBUG] Selected experts: [%d, %d, %d, %d]\n",
             cuda_experts[0], cuda_experts[1], cuda_experts[2], cuda_experts[3]);

      // Print tensor shapes and layout info
      printf("[CUDA WEIGHT DEBUG] FC1 weights shape: [%s]\n", fc1_experts_weights->Shape().ToString().c_str());
      printf("[CUDA WEIGHT DEBUG] FC1 scales shape: [%s]\n", fc1_scales->Shape().ToString().c_str());
      printf("[CUDA WEIGHT DEBUG] FC2 weights shape: [%s]\n", fc2_experts_weights->Shape().ToString().c_str());
      printf("[CUDA WEIGHT DEBUG] FC2 scales shape: [%s]\n", fc2_scales->Shape().ToString().c_str());

      // Calculate memory layout parameters
      size_t fc1_weights_per_expert = moe_params.inter_size * 2 * moe_params.hidden_size / 2;  // 4-bit packed
      size_t fc1_scales_per_expert = moe_params.inter_size * 2;
      size_t fc2_weights_per_expert = moe_params.hidden_size * moe_params.inter_size / 2;  // 4-bit packed
      size_t fc2_scales_per_expert = moe_params.hidden_size;

      printf("[CUDA WEIGHT DEBUG] Expected FC1 weights per expert: %zu bytes\n", fc1_weights_per_expert);
      printf("[CUDA WEIGHT DEBUG] Expected FC1 scales per expert: %zu elements\n", fc1_scales_per_expert);
      printf("[CUDA WEIGHT DEBUG] Expected FC2 weights per expert: %zu bytes\n", fc2_weights_per_expert);
      printf("[CUDA WEIGHT DEBUG] Expected FC2 scales per expert: %zu elements\n", fc2_scales_per_expert);

      // Debug first selected expert (Expert 5) in detail
      int expert_5 = cuda_experts[0];  // Should be 5
      printf("\n[CUDA WEIGHT DEBUG] === EXPERT %d DETAILED ANALYSIS ===\n", expert_5);

      // FC1 quantized weights raw bytes for Expert 5
      size_t fc1_weights_offset = static_cast<size_t>(expert_5) * fc1_weights_per_expert;
      const size_t debug_weight_bytes = std::min(fc1_weights_per_expert, 16UL);

      if (fc1_weights_offset + debug_weight_bytes <= fc1_experts_weights->Shape().Size()) {
        std::vector<uint8_t> fc1_weights_raw(debug_weight_bytes);
        cudaError_t err_weights = cudaMemcpy(fc1_weights_raw.data(),
                                             reinterpret_cast<const uint8_t*>(fc1_experts_weights->template Data<uint8_t>()) + fc1_weights_offset,
                                             debug_weight_bytes, cudaMemcpyDeviceToHost);
        if (err_weights == cudaSuccess) {
          printf("[CUDA WEIGHT DEBUG] Expert %d FC1 raw weight bytes (first 16): ", expert_5);
          for (size_t i = 0; i < debug_weight_bytes; ++i) {
            printf("%02x ", fc1_weights_raw[i]);
          }
          printf("\n");

          // Manually dequantize first few weights using CUDA logic
          printf("[CUDA WEIGHT DEBUG] Expert %d FC1 manual dequantization:\n", expert_5);

          // Get scales for this expert
          size_t fc1_scales_offset = static_cast<size_t>(expert_5) * fc1_scales_per_expert;
          std::vector<T> fc1_scales_expert(std::min(fc1_scales_per_expert, 8UL));
          cudaError_t err_scales = cudaMemcpy(fc1_scales_expert.data(),
                                              fc1_scales->template Data<T>() + fc1_scales_offset,
                                              fc1_scales_expert.size() * sizeof(T), cudaMemcpyDeviceToHost);

          if (err_scales == cudaSuccess) {
            printf("[CUDA WEIGHT DEBUG] Expert %d FC1 scales (first 8): ", expert_5);
            for (size_t i = 0; i < fc1_scales_expert.size(); ++i) {
              printf("%.6f ", static_cast<float>(fc1_scales_expert[i]));
            }
            printf("\n");

            // Manually dequantize first 8 weights (4-bit, so 4 bytes = 8 weights)
            printf("[CUDA WEIGHT DEBUG] Expert %d FC1 dequantized weights (first 8): ", expert_5);
            for (int i = 0; i < std::min(8, static_cast<int>(debug_weight_bytes * 2)); i += 2) {
              int byte_idx = i / 2;
              if (byte_idx < static_cast<int>(debug_weight_bytes)) {
                uint8_t packed_byte = fc1_weights_raw[byte_idx];

                // Extract 4-bit values (CUDA typically uses lower 4 bits first)
                int w0 = packed_byte & 0x0F;         // Lower 4 bits
                int w1 = (packed_byte >> 4) & 0x0F;  // Upper 4 bits

                // Dequantize with scale (assuming zero-point = 8 for 4-bit unsigned)
                if ((i / 2) < static_cast<int>(fc1_scales_expert.size())) {
                  float scale = static_cast<float>(fc1_scales_expert[i / 2]);
                  float dq0 = scale * (static_cast<float>(w0) - 8.0f);
                  float dq1 = scale * (static_cast<float>(w1) - 8.0f);
                  printf("%.6f %.6f ", dq0, dq1);
                }
              }
            }
            printf("\n");
          }
        }
      }

      // Compare with CPU layout expectations
      printf("\n[CUDA WEIGHT DEBUG] === CPU vs CUDA LAYOUT COMPARISON ===\n");
      printf("[CUDA WEIGHT DEBUG] CUDA uses CUTLASS column-major interleaved layout\n");
      printf("[CUDA WEIGHT DEBUG] CPU expects row-major packed layout\n");
      printf("[CUDA WEIGHT DEBUG] This difference may explain the computation discrepancies\n");

      // Print GEMM operation parameters
      printf("\n[CUDA WEIGHT DEBUG] === GEMM PARAMETERS ===\n");
      printf("[CUDA WEIGHT DEBUG] FC1 GEMM: A[%zu,%zu] * B[%zu,%zu] -> C[%zu,%zu]\n",
             static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.hidden_size),
             static_cast<size_t>(moe_params.hidden_size), moe_params.inter_size * 2,
             static_cast<size_t>(moe_params.num_rows), moe_params.inter_size * 2);
      printf("[CUDA WEIGHT DEBUG] FC2 GEMM: A[%zu,%zu] * B[%zu,%zu] -> C[%zu,%zu]\n",
             static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.inter_size),
             static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(moe_params.hidden_size),
             static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.hidden_size));

      printf("[CUDA WEIGHT DEBUG] CUDA GEMM uses: CUTLASS mixed-precision kernels\n");
      printf("[CUDA WEIGHT DEBUG] CPU GEMM uses: MLAS standard GEMM with manual dequantization\n");
      printf("[CUDA WEIGHT DEBUG] ================================================\n\n");
    }
  }

  Tensor* output = context->Output(0, input->Shape());

  ort_fastertransformer::finalize_moe_routing_kernelLauncher(
      reinterpret_cast<CudaT*>(fc2_output.get()), reinterpret_cast<CudaT*>(output->template MutableData<T>()),
      fc2_experts_bias_optional == nullptr
          ? nullptr
          : reinterpret_cast<const CudaT*>(fc2_experts_bias_optional->template Data<T>()),
      reinterpret_cast<CudaT*>(expert_scales.get()),
      reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.get()),
      reinterpret_cast<int*>(expert_for_source_row.get()), static_cast<int>(moe_params.num_rows),
      static_cast<int>(moe_params.hidden_size), static_cast<int>(k_), Stream(context));

  // DEBUG: Print FC2 GEMM outputs and final outputs (matching CPU debug)
  cudaStreamSynchronize(Stream(context));

  // Print FC2 output before final accumulation (for first few experts)
  if (moe_params.num_rows > 0 && k_ > 0) {
    const int debug_elements = std::min(static_cast<int>(moe_params.hidden_size), 8);
    const int debug_tokens = std::min(static_cast<int>(k_ * moe_params.num_rows), 4);

    std::vector<T> fc2_output_debug(debug_tokens * debug_elements);
    cudaError_t err_fc2 = cudaMemcpy(fc2_output_debug.data(), fc2_output.get(),
                                     debug_tokens * debug_elements * sizeof(T), cudaMemcpyDeviceToHost);
    if (err_fc2 == cudaSuccess) {
      printf("[CUDA DEBUG] FC2 GEMM output (first %d expert outputs, first 8 elements):\n", debug_tokens);
      for (int i = 0; i < debug_tokens; ++i) {
        printf("[CUDA DEBUG] FC2 output[%d]: ", i);
        for (int j = 0; j < debug_elements; ++j) {
          printf("%.6f ", static_cast<float>(fc2_output_debug[i * debug_elements + j]));
        }
        printf("...\n");
      }
    }
  }

  // Print final accumulated output
  if (moe_params.num_rows > 0 && output != nullptr) {
    const int debug_elements = std::min(static_cast<int>(moe_params.hidden_size), 8);
    const int debug_rows = std::min(static_cast<int>(moe_params.num_rows), 3);
    const size_t required_size = static_cast<size_t>(debug_rows) * debug_elements;

    // Safety check: Ensure output tensor is large enough
    size_t output_total_size = output->Shape().Size();

    if (required_size <= output_total_size) {
      std::vector<T> output_debug(required_size);
      cudaError_t err_output = cudaMemcpy(output_debug.data(), output->template MutableData<T>(),
                                          required_size * sizeof(T), cudaMemcpyDeviceToHost);
      if (err_output == cudaSuccess) {
        for (int row = 0; row < debug_rows; ++row) {
          printf("[CUDA DEBUG] Final output Row %d (first 8): ", row);
          for (int i = 0; i < debug_elements; ++i) {
            size_t idx = static_cast<size_t>(row) * debug_elements + i;
            if (idx < required_size) {
              printf("%.6f ", static_cast<float>(output_debug[idx]));
            }
          }
          printf("...\n");
        }
      } else {
        printf("[CUDA DEBUG] Final output: cudaMemcpy failed (error %d)\n", err_output);
      }
    } else {
      printf("[CUDA DEBUG] Final output: skipped (tensor too small: %zu vs required %zu)\n",
             output_total_size, required_size);
    }
  }

  return Status::OK();
}

template <typename T>
Status QMoE<T>::ComputeInternal(OpKernelContext* context) const {
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
      activation_type_ == ort_fastertransformer::ActivationType::SwiGLU));

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"  // Mute "maybe used uninitialized" warning for MoEParameters.
#endif

  if (expert_weight_bits_ == 4) {
    using CudaWeightT = typename ToCudaTypeWrapper<uint8_t, true>::MappedType;
    return QuantizedMoEImpl<CudaWeightT>(context, moe_params, input, router_probs,
                                         fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                         fc2_experts_bias_optional, fc3_experts_weights_optional,
                                         fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional,
                                         GetDeviceProp());
  } else {
    using CudaWeightT = typename ToCudaTypeWrapper<uint8_t, false>::MappedType;
    return QuantizedMoEImpl<CudaWeightT>(context, moe_params, input, router_probs,
                                         fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                         fc2_experts_bias_optional, fc3_experts_weights_optional,
                                         fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional,
                                         GetDeviceProp());
  }

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<MLFloat16>()),
    QMoE<MLFloat16>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE,
    kMSDomain,
    1,
    BFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<BFloat16>()),
    QMoE<BFloat16>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
