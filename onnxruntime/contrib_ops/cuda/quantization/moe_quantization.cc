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

  // Debug: Detailed CUDA intermediate values analysis
  std::vector<int> cuda_experts_detailed(debug_k);

  // Copy expert scales and indices (safe sizes)
  cudaError_t err_scales = cudaMemcpy(cuda_expert_scales_detailed.data(), expert_scales.get(),
                                      debug_k * sizeof(T), cudaMemcpyDeviceToHost);
  cudaError_t err_experts = cudaMemcpy(cuda_experts_detailed.data(), expert_for_source_row.get(),
                                       debug_k * sizeof(int), cudaMemcpyDeviceToHost);

  // Debug: Print input activations
  std::vector<T> input_debug(debug_elements);
  cudaError_t err_input = cudaMemcpy(input_debug.data(), input->template Data<T>(),
                                     debug_elements * sizeof(T), cudaMemcpyDeviceToHost);
  if (err_input == cudaSuccess) {
    printf("CUDA DEBUG: Input activations first %d values: ", debug_elements);
    for (int i = 0; i < debug_elements; i++) {
      printf("%.6f ", static_cast<float>(input_debug[i]));
    }
    printf("\n");
  }

  // Debug: Print router probabilities
  // Debug: Print FC1 and FC2 scales
  // CUDA Matrix Layout Investigation: Check FC1 dimensions and CUTLASS layout requirements

                 static_cast<int>(moe_params.hidden_size), ThreadblockK);
                 printf("CUDA LAYOUT DEBUG: FC1 quantized weights memory layout analysis:\n");
                 for (size_t i = 0; i < fc1_shape.NumDimensions(); i++) {
          cudaError_t err_fc1_weights = cudaMemcpy(fc1_weights_sample.data(),
            if (err_fc1_weights_64 == cudaSuccess) {
      printf("  FC1 weights at ThreadblockK boundary (%d): ", ThreadblockK);
      for (int i = 0; i < sample_size; i++) {
        printf("%d ", fc1_weights_sample[i]);
      }
      printf("\n");
            }
                 }
}
}
}

if (fc2_scales) {
  std::vector<T> fc2_scales_debug(debug_elements);
  cudaError_t err_fc2_scales = cudaMemcpy(fc2_scales_debug.data(), fc2_scales->template Data<T>(),
                                          debug_elements * sizeof(T), cudaMemcpyDeviceToHost);
  if (err_fc2_scales == cudaSuccess) {
    printf("CUDA DEBUG: FC2 scales first %d values: ", debug_elements);
    for (int i = 0; i < debug_elements; i++) {
      printf("%.6f ", static_cast<float>(fc2_scales_debug[i]));
    }
    printf("\n");
  }
}

if (err_scales == cudaSuccess && err_experts == cudaSuccess) {
  printf("CUDA DEBUG: Detailed intermediate values analysis (safe version):\n");

  // Debug: Print quantized weights for selected experts
  for (int k_idx = 0; k_idx < debug_k; k_idx++) {
    int expert_idx = cuda_experts_detailed[k_idx];

    // Debug FC1 quantized weights for this expert
    const size_t fc1_expert_size = static_cast<size_t>(moe_params.hidden_size) * moe_params.inter_size * 2;
    const size_t fc1_offset = expert_idx * fc1_expert_size;
    std::vector<uint8_t> fc1_weights_debug(debug_elements);

    cudaError_t err_fc1_weights = cudaMemcpy(fc1_weights_debug.data(),
                                             static_cast<const uint8_t*>(fc1_experts_weights->DataRaw()) + fc1_offset,
                                             debug_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (err_fc1_weights == cudaSuccess) {
      printf("CUDA DEBUG: Expert %d FC1 quantized weights first %d: ", expert_idx, debug_elements);
      for (int i = 0; i < debug_elements; i++) {
        printf("%d ", fc1_weights_debug[i]);
      }
      printf("\n");
    }

    // Debug FC2 quantized weights for this expert
    const size_t fc2_expert_size = static_cast<size_t>(moe_params.inter_size) * moe_params.hidden_size;
    const size_t fc2_offset = expert_idx * fc2_expert_size;
    std::vector<uint8_t> fc2_weights_debug(debug_elements);
    cudaError_t err_fc2_weights = cudaMemcpy(fc2_weights_debug.data(),
                                             static_cast<const uint8_t*>(fc2_experts_weights->DataRaw()) + fc2_offset,
                                             debug_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (err_fc2_weights == cudaSuccess) {
      printf("CUDA DEBUG: Expert %d FC2 quantized weights first %d: ", expert_idx, debug_elements);
      for (int i = 0; i < debug_elements; i++) {
        printf("%d ", fc2_weights_debug[i]);
      }
      printf("\n");
    }

    // --- ADDED: Print FC2 dequantized weights for Expert 5 (first 5 values) ---
    if (expert_idx == 5 && fc2_scales) {
      std::vector<T> fc2_dequantized_debug(5, T{});
      size_t fc2_scales_size = fc2_scales->Shape().Size();
      size_t fc2_weights_size = fc2_weights_debug.size();
      for (int i = 0; i < 5; i++) {
        size_t scale_idx = fc2_offset + i;
        if (scale_idx < fc2_scales_size && i < fc2_weights_size) {
          float scale = static_cast<float>(fc2_scales->template Data<T>()[scale_idx]);
          int8_t qval = static_cast<int8_t>(fc2_weights_debug[i]);
          fc2_dequantized_debug[i] = static_cast<T>(scale * (static_cast<float>(qval)));
        } else {
          printf("[OOB] ");
        }
      }
      printf("CUDA DEBUG: Expert 5 FC2 dequantized weights first 5: ");
      for (int i = 0; i < 5; i++) {
        printf("%.6f ", static_cast<float>(fc2_dequantized_debug[i]));
      }
      printf("\n");
    }

    // Debug expert-specific scales
    if (fc1_scales) {
      const size_t fc1_scales_offset = expert_idx * (moe_params.inter_size * 2);
      std::vector<T> expert_fc1_scales(5);
      cudaError_t err_expert_fc1_scales = cudaMemcpy(expert_fc1_scales.data(),
                                                     fc1_scales->template Data<T>() + fc1_scales_offset,
                                                     5 * sizeof(T), cudaMemcpyDeviceToHost);
      if (err_expert_fc1_scales == cudaSuccess) {
        printf("CUDA DEBUG: Expert %d FC1 scales first 5: ", expert_idx);
        for (int i = 0; i < 5; i++) {
          printf("%.6f ", static_cast<float>(expert_fc1_scales[i]));
        }
        printf("\n");
      }
    }

    if (fc2_scales) {
      const size_t fc2_scales_offset = expert_idx * moe_params.hidden_size;
      std::vector<T> expert_fc2_scales(5);
      cudaError_t err_expert_fc2_scales = cudaMemcpy(expert_fc2_scales.data(),
                                                     fc2_scales->template Data<T>() + fc2_scales_offset,
                                                     5 * sizeof(T), cudaMemcpyDeviceToHost);
      if (err_expert_fc2_scales == cudaSuccess) {
        printf("CUDA DEBUG: Expert %d FC2 scales first 5: ", expert_idx);
        for (int i = 0; i < 5; i++) {
          printf("%.6f ", static_cast<float>(expert_fc2_scales[i]));
        }
        printf("\n");
      }
    }
  }

  // Validate expert indices before using them
  bool valid_experts = true;
  for (int i = 0; i < debug_k; i++) {
    if (cuda_experts_detailed[i] < 0 || cuda_experts_detailed[i] >= moe_params.num_experts) {
      printf("CUDA DEBUG: Invalid expert index %d at position %d (num_experts=%d)\n",
             cuda_experts_detailed[i], i, static_cast<int>(moe_params.num_experts));
      valid_experts = false;
      break;
    }
  }

  if (valid_experts) {
    // Manual calculation like CPU version (using only expert scales)
    printf("CUDA DEBUG: Expert selection and weights (safe access):\n");

    for (int k_idx = 0; k_idx < debug_k; k_idx++) {
      int expert_idx = cuda_experts_detailed[k_idx];
      float expert_scale = static_cast<float>(cuda_expert_scales_detailed[k_idx]);

      printf("CUDA DEBUG: Expert %d, Row 0 - scale=%.6f\n", expert_idx, expert_scale);

      // CUDA Matrix Layout Investigation: Compare specific weight values with CPU
      if (fc1_experts_weights && (expert_idx == 5 || expert_idx == 8 || expert_idx == 9 || expert_idx == 24)) {
        printf("CUDA LAYOUT DEBUG: Expert %d FC1 - Quantized weight values at specific coordinates:\n", expert_idx);

        // Calculate expert's weight matrix base offset
        int64_t hidden_size = moe_params.hidden_size;
        int64_t inter_size = moe_params.inter_size;
        int64_t expert_weight_offset = expert_idx * (hidden_size * inter_size);

        // Sample quantized weights at specific logical coordinates for comparison with CPU
        const int coord_samples = 15;  // 3 rows x 5 cols
        std::vector<uint8_t> coord_weights(coord_samples);

        for (int row = 0; row < 3 && row * hidden_size < inter_size; row++) {
          for (int col = 0; col < 5 && col < hidden_size; col++) {
            // Calculate linear index in CUTLASS layout
            // This might need adjustment based on actual CUTLASS tile interleaving
            int64_t linear_idx = expert_weight_offset + row * hidden_size + col;

            if (linear_idx < fc1_experts_weights->Shape().Size()) {
              uint8_t weight_val;
              cudaError_t err = cudaMemcpy(&weight_val,
                                           fc1_experts_weights->template Data<uint8_t>() + linear_idx,
                                           sizeof(uint8_t), cudaMemcpyDeviceToHost);
              if (err == cudaSuccess) {
                printf("  [%d,%d] = %d", row, col, weight_val);
              } else {
                printf("  [%d,%d] = ERR", row, col);
              }
            } else {
              printf("  [%d,%d] = OOB", row, col);
            }
          }
          printf("\n");
        }
      }

      // Try to safely copy FC2 expert outputs for this expert
      // The fc2_output buffer contains outputs for all k*num_rows expert outputs
      // Layout: [expert_0_output, expert_1_output, ..., expert_k-1_output] for each row
      std::vector<T> expert_output_debug(debug_elements);
      const CudaT* expert_output_start = reinterpret_cast<const CudaT*>(fc2_output.get()) +
                                         k_idx * moe_params.hidden_size;  // k_idx position for row 0

      cudaError_t err_expert_output = cudaMemcpy(expert_output_debug.data(), expert_output_start,
                                                 debug_elements * sizeof(T), cudaMemcpyDeviceToHost);

      if (err_expert_output == cudaSuccess) {
        printf("CUDA DEBUG: Expert %d, Row 0 - first %d expert outputs: ", expert_idx, debug_elements);
        for (int i = 0; i < debug_elements; i++) {
          printf("%.6f ", static_cast<float>(expert_output_debug[i]));
        }
        printf("\n");

        // --- ADDED: Print FC2 GEMM output (before bias) for Expert 5 (first 5 values) ---
        if (expert_idx == 5) {
          printf("CUDA DEBUG: Expert 5 FC2 GEMM output (before bias) first 5: ");
          for (int i = 0; i < 5 && i < debug_elements; i++) {
            printf("%.6f ", static_cast<float>(expert_output_debug[i]));
          }
          printf("\n");
        }
      } else {
        printf("CUDA DEBUG: Expert %d - Failed to copy expert outputs (err=%d)\n", expert_idx, err_expert_output);
      }

      // CUDA FC2 Weights Layout Investigation for comparison with CPU
      if (fc2_experts_weights && (expert_idx == 5 || expert_idx == 8 || expert_idx == 9 || expert_idx == 24)) {
        printf("CUDA LAYOUT DEBUG: Expert %d FC2 - Quantized weight layout analysis:\n", expert_idx);

        // Get FC2 weights tensor shape
        const auto& fc2_shape = fc2_experts_weights->Shape();
        printf("  FC2 weights tensor shape: [");
        for (size_t i = 0; i < fc2_shape.NumDimensions(); i++) {
          printf("%lld", fc2_shape[i]);
          if (i < fc2_shape.NumDimensions() - 1) printf(", ");
        }
        printf("]\n");

        // Calculate FC2 expert weight matrix offset
        int64_t hidden_size = moe_params.hidden_size;
        int64_t inter_size = moe_params.inter_size;
        int64_t fc2_expert_offset = expert_idx * (inter_size * hidden_size);

        printf("  FC2 Expert %d offset: %lld, total elements: %lld\n",
               expert_idx, fc2_expert_offset, fc2_shape.Size());

        // Sample FC2 quantized weights at specific coordinates
        printf("  FC2 quantized weights at specific coordinates:\n");
        for (int row = 0; row < 3 && row < hidden_size; row++) {
          printf("  Row %d: ", row);
          for (int col = 0; col < 5 && col < inter_size; col++) {
            int64_t linear_idx = fc2_expert_offset + row * inter_size + col;

            if (linear_idx < fc2_shape.Size()) {
              uint8_t weight_val;
              cudaError_t err = cudaMemcpy(&weight_val,
                                           fc2_experts_weights->template Data<uint8_t>() + linear_idx,
                                           sizeof(uint8_t), cudaMemcpyDeviceToHost);
              if (err == cudaSuccess) {
                printf("[%d,%d]=%d ", row, col, weight_val);
              }
            }
          }
          printf("\n");
        }
      }

      // Try to safely copy bias values for this expert
      if (fc2_experts_bias_optional) {
        std::vector<T> expert_bias_debug(debug_elements);
        const CudaT* expert_bias_start = reinterpret_cast<const CudaT*>(fc2_experts_bias_optional->template Data<T>()) +
                                         expert_idx * moe_params.hidden_size;

        cudaError_t err_bias = cudaMemcpy(expert_bias_debug.data(), expert_bias_start,
                                          debug_elements * sizeof(T), cudaMemcpyDeviceToHost);

        if (err_bias == cudaSuccess) {
          printf("CUDA DEBUG: Expert %d, Row 0 - first %d FC2 biases: ", expert_idx, debug_elements);
          for (int i = 0; i < debug_elements; i++) {
            printf("%.6f ", static_cast<float>(expert_bias_debug[i]));
          }
          printf("\n");

          // Calculate outputs+bias and scaled contributions for manual verification
          if (err_expert_output == cudaSuccess) {
            printf("CUDA DEBUG: Expert %d, Row 0 - first %d outputs+bias: ", expert_idx, debug_elements);
            for (int i = 0; i < debug_elements; i++) {
              printf("%.6f ", static_cast<float>(expert_output_debug[i]) + static_cast<float>(expert_bias_debug[i]));
            }
            printf("\n");

            printf("CUDA DEBUG: Expert %d, Row 0 - first %d scaled outputs: ", expert_idx, debug_elements);
            for (int i = 0; i < debug_elements; i++) {
              printf("%.6f ", expert_scale * (static_cast<float>(expert_output_debug[i]) + static_cast<float>(expert_bias_debug[i])));
            }
            printf("\n");
          }
        } else {
          printf("CUDA DEBUG: Expert %d - Failed to copy bias values (err=%d)\n", expert_idx, err_bias);
        }
      }
    }

    // Manual calculation like CPU version with actual expert outputs
    if (fc2_experts_bias_optional) {
      printf("CUDA DEBUG: Manual calculation verification:\n");
      float cuda_manual_output[10] = {0.0f};  // Initialize to debug_elements size

      for (int k_idx = 0; k_idx < debug_k; k_idx++) {
        int expert_idx = cuda_experts_detailed[k_idx];
        float expert_scale = static_cast<float>(cuda_expert_scales_detailed[k_idx]);

        // Copy expert output and bias again for manual calculation
        std::vector<T> expert_output_debug(debug_elements);
        std::vector<T> expert_bias_debug(debug_elements);

        const CudaT* expert_output_start = reinterpret_cast<const CudaT*>(fc2_output.get()) +
                                           k_idx * moe_params.hidden_size;
        const CudaT* expert_bias_start = reinterpret_cast<const CudaT*>(fc2_experts_bias_optional->template Data<T>()) +
                                         expert_idx * moe_params.hidden_size;

        cudaError_t err_out = cudaMemcpy(expert_output_debug.data(), expert_output_start,
                                         debug_elements * sizeof(T), cudaMemcpyDeviceToHost);
        cudaError_t err_bias = cudaMemcpy(expert_bias_debug.data(), expert_bias_start,
                                          debug_elements * sizeof(T), cudaMemcpyDeviceToHost);

        if (err_out == cudaSuccess && err_bias == cudaSuccess) {
          for (int j = 0; j < debug_elements; j++) {
            float expert_output = static_cast<float>(expert_output_debug[j]);
            float bias_value = static_cast<float>(expert_bias_debug[j]);
            float scaled_contribution = expert_scale * (expert_output + bias_value);
            cuda_manual_output[j] += scaled_contribution;

            // Debug each step of accumulation
            if (j < 5) {  // Only print first 5 for readability
              printf("CUDA DEBUG: Expert %d pos[%d] - output=%.6f, bias=%.6f, output+bias=%.6f, scale=%.6f, contribution=%.6f, total=%.6f\n",
                     expert_idx, j, expert_output, bias_value, expert_output + bias_value, expert_scale, scaled_contribution, cuda_manual_output[j]);
            }
          }

          printf("CUDA DEBUG: Expert %d accumulation - output[0:%d] after: ", expert_idx, std::min(debug_elements, 5));
          for (int j = 0; j < std::min(debug_elements, 5); j++) {
            printf("%.6f ", cuda_manual_output[j]);
          }
          printf("\n");
        }
      }

      printf("CUDA DEBUG: Manual calculation output first %d values: ", std::min(debug_elements, 5));
      for (int j = 0; j < std::min(debug_elements, 5); j++) {
        printf("%.6f ", cuda_manual_output[j]);
      }
      printf("\n");
    }

  } else {
    printf("CUDA DEBUG: Skipping detailed analysis due to invalid expert indices\n");
  }

} else {
  printf("CUDA DEBUG: Failed to copy expert data (scales=%d, experts=%d)\n", err_scales, err_experts);
}

// Debug: Print final CUDA output after finalize_moe_routing_kernel
printf("CUDA DEBUG: Final output comparison:\n");
std::vector<T> final_output_debug(debug_elements);
cudaError_t err_final = cudaMemcpy(final_output_debug.data(), output->template MutableData<T>(),
                                   debug_elements * sizeof(T), cudaMemcpyDeviceToHost);
if (err_final == cudaSuccess) {
  printf("CUDA DEBUG: Final kernel output first %d values: ", debug_elements);
  for (int i = 0; i < debug_elements; i++) {
    printf("%.6f ", static_cast<float>(final_output_debug[i]));
  }
  printf("\n");
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
