// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/moe/moe_quantization_cpu.h"
#include "core/framework/allocator.h"
#include "core/common/float16.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cpu/activation/activations.h"
#include "core/common/safeint.h"
#include "core/common/narrow.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/util/math.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cpu/moe/moe_utils.h"
#include "contrib_ops/cpu/moe/moe_helper.h"

#include <atomic>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
inline uint8_t GetPackedZeroPointValue(int64_t num_bits, uint8_t zero_point) {
  ORT_ENFORCE(num_bits > 0 && num_bits <= 8 && (8 % num_bits) == 0,
              "num_bits must be a positive divisor of 8, got ", num_bits);
  const int64_t pack_size = 8 / num_bits;
  const uint8_t mask = static_cast<uint8_t>((1u << num_bits) - 1u);
  uint8_t packed_value = 0;

  for (int64_t i = 0; i < pack_size; ++i) {
    packed_value |= static_cast<uint8_t>((zero_point & mask) << (i * num_bits));
  }

  return packed_value;
}

inline int64_t GetOptimalBlockSize(int64_t total_elements, int num_threads) {
  if (total_elements <= 0 || num_threads <= 0) return 64;
  const int64_t l1_cache_elements = 8192;  // ~32KB / 4 bytes per float
  const int64_t divisor = std::max(1, num_threads > 1 ? 4 : 2);
  const int64_t base_block_size = l1_cache_elements / divisor;
  const int64_t max_block = std::max(int64_t{32}, total_elements / std::max(int64_t{1}, int64_t{4}));
  return std::clamp(base_block_size, int64_t{32}, std::min(int64_t{512}, max_block));
}

inline int64_t GetUnrollFactor(int64_t vector_size) {
  if (vector_size <= 0) return 2;
  if (vector_size >= 512) return 16;
  if (vector_size >= 128) return 8;
  if (vector_size >= 32) return 4;
  return 2;
}

inline bool ShouldUseMemcpy(int64_t size) {
  return size >= 64;
}

inline int64_t GetDequantBlockSize(int64_t features, int64_t total_work) {
  if (features <= 0 || total_work <= 0) return 16;
  const int64_t target_block_size = std::max(int64_t{16}, features / std::max(int64_t{1}, int64_t{8}));
  const int64_t work_based_size = std::max(int64_t{16}, total_work / std::max(int64_t{1}, int64_t{4}));
  return std::min(target_block_size, work_based_size);
}

bool CanUseMlasQ4Dequant(int64_t num_bits) {
  if (num_bits != 4) {
    return false;
  }

  return true;
}

bool CanUseMlasQ4Gemm(int64_t expert_weight_bits, int64_t block_size,
                      int64_t rows, int64_t cols, MLAS_BLK_QUANT_TYPE& out_qtype) {
  if (expert_weight_bits != 4) {
    return false;
  }

  if (block_size == 64) {
    out_qtype = BlkQ4Sym64;
  } else if (block_size == 128) {
    out_qtype = BlkQ4Sym128;
  } else if (block_size == 0 || block_size == 32) {
    out_qtype = BlkQ4Sym;
  } else {
    return false;
  }

  size_t expected_size = MlasQ4GemmPackBSize(out_qtype, static_cast<size_t>(rows), static_cast<size_t>(cols));
  return expected_size > 0;
}

bool CanUseMlasLutGemm(int64_t expert_weight_bits, int64_t block_size,
                       int64_t rows, int64_t cols) {
  if (expert_weight_bits != 2 || block_size <= 0) {
    return false;
  }

  if ((cols % block_size) != 0) {
    return false;
  }

  return MlasIsLutGemmAvailable(static_cast<size_t>(rows), static_cast<size_t>(cols),
                                static_cast<size_t>(expert_weight_bits), static_cast<size_t>(block_size));
}

}  // namespace

namespace onnxruntime {
namespace contrib {

constexpr const char* kUseMlasQ4GemmMoe = "ORT_USE_MLAS_Q4_GEMM_MOE";

template <typename TScale>
void DequantizeBlockWithMlas(const uint8_t* quantized_data,
                             const TScale* scales,
                             const uint8_t* zero_points,
                             int64_t block_size,
                             int64_t num_bits,
                             int64_t rows,
                             int64_t cols,
                             float* dequantized_data,
                             MLAS_THREADPOOL* thread_pool);

template <typename TScale>
Status ConvertToMlasQ4Format(const uint8_t* quantized_data,
                             const TScale* scales,
                             const uint8_t* zero_points,
                             int64_t block_size,
                             int64_t num_bits,
                             int64_t rows,
                             int64_t cols,
                             MLAS_BLK_QUANT_TYPE qtype,
                             AllocatorPtr allocator,
                             IAllocatorUniquePtr<uint8_t>& mlas_packed_buffer) {
  if (num_bits != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only 4-bit quantization supported for MLAS Q4 format conversion");
  }
  if (zero_points != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MLAS Q4 format conversion only supports symmetric quantization (zero_points must be null)");
  }

  auto temp_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(rows * cols));
  float* temp_float = temp_float_buffer.get();

  DequantizeBlockWithMlas(quantized_data, scales, zero_points, block_size, num_bits, rows, cols, temp_float, nullptr);

  // Transpose from N x K (weights) to K x N.
  // DirectQ4Gemm expects weights to be packed in a specific layout ([K, N] logically)
  auto transposed_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(rows * cols));
  float* transposed_float = transposed_float_buffer.get();
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      transposed_float[c * rows + r] = temp_float[r * cols + c];
    }
  }

  size_t packed_size = MlasQ4GemmPackBSize(qtype, static_cast<size_t>(rows), static_cast<size_t>(cols));
  if (packed_size == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MLAS Q4 packing not supported for this configuration");
  }

  mlas_packed_buffer = IAllocator::MakeUniquePtr<uint8_t>(allocator, packed_size);
  MlasQ4GemmPackB(qtype, mlas_packed_buffer.get(), transposed_float, static_cast<size_t>(rows), static_cast<size_t>(cols), static_cast<size_t>(rows));

  return Status::OK();
}

Status DirectQ4Gemm(const float* A,
                    const uint8_t* mlas_packed_B,
                    const float* bias,
                    float* C,
                    int64_t M,
                    int64_t N,
                    int64_t K,
                    MLAS_BLK_QUANT_TYPE qtype,
                    MLAS_THREADPOOL* thread_pool) {
  MLAS_Q4_GEMM_DATA_PARAMS params;
  params.A = A;
  params.lda = static_cast<size_t>(K);
  params.B = mlas_packed_B;
  params.Bias = bias;
  params.C = C;
  params.ldc = static_cast<size_t>(N);
  params.OutputProcessor = nullptr;

  MlasQ4GemmBatch(qtype, static_cast<size_t>(M), static_cast<size_t>(N), static_cast<size_t>(K), 1, &params, thread_pool);
  return Status::OK();
}

template <typename TScale>
const float* GetFloatScaleData(const TScale* scales_data,
                               size_t scale_count,
                               float* converted_scales) {
  if constexpr (std::is_same_v<TScale, float>) {
    ORT_UNUSED_PARAMETER(scale_count);
    ORT_UNUSED_PARAMETER(converted_scales);
    return scales_data;
  } else {
    ORT_ENFORCE(converted_scales != nullptr, "converted_scales buffer must be provided for non-float scale data.");
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(scales_data), converted_scales, scale_count);
    return converted_scales;
  }
}

template <typename TScale>
bool TryRunLutGemm(const float* activations,
                   float* output,
                   const uint8_t* weights_data,
                   const void* direct_lut_cache_ptr,
                   const TScale* scales_ptr,
                   const uint8_t* zp_ptr,
                   int64_t expert_idx,
                   int64_t rows,
                   int64_t cols,
                   int64_t packed_cols,
                   int64_t block_size,
                   int64_t blocks_per_row,
                   std::byte* thread_lut_packed_buffer,
                   float* thread_lut_scale_buffer,
                   int64_t num_expert_tokens,
                   MLAS_THREADPOOL* thread_pool) {
  if (direct_lut_cache_ptr == nullptr && weights_data == nullptr) {
    return false;
  }

  const void* packed_lut_b = direct_lut_cache_ptr;
  if (packed_lut_b == nullptr) {
    ORT_ENFORCE(thread_lut_packed_buffer != nullptr, "Thread-local LUT packed buffer is required.");
    const size_t scale_count = static_cast<size_t>(rows * blocks_per_row);
    const float* scales_fp32 = GetFloatScaleData(scales_ptr, scale_count, thread_lut_scale_buffer);
    MlasInitLutGemmKernelConfig(static_cast<size_t>(rows), static_cast<size_t>(cols), 2,
                                static_cast<size_t>(block_size), zp_ptr != nullptr);
    MlasLutGemmPack(static_cast<size_t>(rows), static_cast<size_t>(cols), 2,
                    static_cast<size_t>(block_size), zp_ptr != nullptr,
                    reinterpret_cast<const std::byte*>(weights_data + expert_idx * rows * packed_cols),
                    scales_fp32, zp_ptr, thread_lut_packed_buffer, thread_pool);
    packed_lut_b = thread_lut_packed_buffer;
  }

  MlasLutGemm(activations, static_cast<size_t>(block_size), packed_lut_b, output,
              static_cast<size_t>(cols), static_cast<size_t>(num_expert_tokens),
              static_cast<size_t>(rows), zp_ptr != nullptr, thread_pool);
  return true;
}

template <typename TScale>
Status BuildDirectLutPackedBCache(const uint8_t* quantized_data,
                                  const TScale* scales_data,
                                  const uint8_t* zero_points,
                                  int64_t num_experts,
                                  int64_t rows,
                                  int64_t cols,
                                  int64_t block_size,
                                  int64_t blocks_per_row,
                                  AllocatorPtr allocator,
                                  IAllocatorUniquePtr<void>& packed_b) {
  ORT_RETURN_IF_NOT(CanUseMlasLutGemm(2, block_size, rows, cols),
                    "LUT GEMM is not supported for rows=", rows, ", cols=", cols, ", block_size=", block_size, ".");

  const bool has_zero_points = (zero_points != nullptr);
  MlasInitLutGemmKernelConfig(static_cast<size_t>(rows), static_cast<size_t>(cols), 2,
                              static_cast<size_t>(block_size), has_zero_points);
  const size_t packed_size_per_expert = MlasLutGemmPackedSize(static_cast<size_t>(rows), static_cast<size_t>(cols), 2,
                                                              static_cast<size_t>(block_size), has_zero_points);
  ORT_RETURN_IF(packed_size_per_expert == 0, "Failed to compute LUT GEMM packed size.");

  const int64_t packed_cols = cols / 4;
  const size_t quantized_stride = static_cast<size_t>(rows * packed_cols);
  const size_t scales_stride = static_cast<size_t>(rows * blocks_per_row);
  const size_t zp_stride = has_zero_points ? static_cast<size_t>(rows * ((blocks_per_row + 3) / 4)) : 0;
  const size_t total_packed_size = packed_size_per_expert * static_cast<size_t>(num_experts);

  packed_b = IAllocator::MakeUniquePtr<void>(allocator, total_packed_size, true);
  auto* packed_b_ptr = static_cast<std::byte*>(packed_b.get());
  std::vector<float> scales_fp32(scales_stride);

  for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const uint8_t* expert_quantized = quantized_data + static_cast<size_t>(expert_idx) * quantized_stride;
    const TScale* expert_scales = scales_data + static_cast<size_t>(expert_idx) * scales_stride;
    const uint8_t* expert_zero_points = has_zero_points ? zero_points + static_cast<size_t>(expert_idx) * zp_stride : nullptr;
    const float* expert_scales_fp32 = GetFloatScaleData(expert_scales, scales_stride, scales_fp32.data());

    MlasLutGemmPack(static_cast<size_t>(rows), static_cast<size_t>(cols), 2, static_cast<size_t>(block_size),
                    has_zero_points, reinterpret_cast<const std::byte*>(expert_quantized), expert_scales_fp32,
                    expert_zero_points, packed_b_ptr + static_cast<size_t>(expert_idx) * packed_size_per_expert, nullptr);
  }

  return Status::OK();
}

template <typename TScale>
void DequantizeBlockWithMlas(const uint8_t* quantized_data,
                             const TScale* scales,
                             const uint8_t* zero_points,
                             int64_t block_size,
                             int64_t num_bits,
                             int64_t rows,
                             int64_t cols,
                             float* dequantized_data,
                             MLAS_THREADPOOL* thread_pool) {
  ORT_UNUSED_PARAMETER(thread_pool);
  const float default_zp_8bit = 128.0f;
  const int64_t zp_pack_size = 8 / num_bits;

  if (CanUseMlasQ4Dequant(num_bits) && zero_points == nullptr) {
    // Use optimized symmetric 4-bit dequantization
    const float default_zp_4bit = 8.0f;
    const int64_t packed_cols = (cols + 1) / 2;
    const int64_t blocks_per_row = (block_size > 0) ? ((cols + block_size - 1) / block_size) : 1;

    if (block_size == 0) {
      for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row_data = quantized_data + r * packed_cols;
        float* row_output = dequantized_data + r * cols;
        const float scale = static_cast<float>(scales[r]);

        int64_t c = 0;
        for (; c + 8 <= cols; c += 8) {
          const uint8_t packed_val0 = row_data[(c + 0) / 2];
          const uint8_t packed_val1 = row_data[(c + 2) / 2];
          const uint8_t packed_val2 = row_data[(c + 4) / 2];
          const uint8_t packed_val3 = row_data[(c + 6) / 2];

          row_output[c + 0] = scale * (static_cast<float>(packed_val0 & 0x0F) - default_zp_4bit);
          row_output[c + 1] = scale * (static_cast<float>(packed_val0 >> 4) - default_zp_4bit);
          row_output[c + 2] = scale * (static_cast<float>(packed_val1 & 0x0F) - default_zp_4bit);
          row_output[c + 3] = scale * (static_cast<float>(packed_val1 >> 4) - default_zp_4bit);
          row_output[c + 4] = scale * (static_cast<float>(packed_val2 & 0x0F) - default_zp_4bit);
          row_output[c + 5] = scale * (static_cast<float>(packed_val2 >> 4) - default_zp_4bit);
          row_output[c + 6] = scale * (static_cast<float>(packed_val3 & 0x0F) - default_zp_4bit);
          row_output[c + 7] = scale * (static_cast<float>(packed_val3 >> 4) - default_zp_4bit);
        }

        for (; c < cols; c += 2) {
          const uint8_t packed_val = row_data[c / 2];
          const uint8_t val0 = packed_val & 0x0F;
          const uint8_t val1 = packed_val >> 4;

          row_output[c] = scale * (static_cast<float>(val0) - default_zp_4bit);
          if (c + 1 < cols) {
            row_output[c + 1] = scale * (static_cast<float>(val1) - default_zp_4bit);
          }
        }
      }
      return;
    } else {
      for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row_data = quantized_data + r * packed_cols;
        float* row_output = dequantized_data + r * cols;

        for (int64_t block_start = 0; block_start < cols; block_start += block_size) {
          const int64_t block_end = std::min(block_start + block_size, cols);
          const int64_t block_idx = std::min(block_start / block_size, blocks_per_row - 1);
          const int64_t scale_idx = r * blocks_per_row + block_idx;
          const float scale = static_cast<float>(scales[scale_idx]);

          int64_t c = block_start;
          for (; c + 4 <= block_end; c += 4) {
            const uint8_t packed_val0 = row_data[(c + 0) / 2];
            const uint8_t packed_val1 = row_data[(c + 2) / 2];

            row_output[c + 0] = scale * (static_cast<float>(packed_val0 & 0x0F) - default_zp_4bit);
            row_output[c + 1] = scale * (static_cast<float>(packed_val0 >> 4) - default_zp_4bit);
            row_output[c + 2] = scale * (static_cast<float>(packed_val1 & 0x0F) - default_zp_4bit);
            row_output[c + 3] = scale * (static_cast<float>(packed_val1 >> 4) - default_zp_4bit);
          }

          for (; c < block_end; c += 2) {
            const uint8_t packed_val = row_data[c / 2];
            const uint8_t val0 = packed_val & 0x0F;
            const uint8_t val1 = packed_val >> 4;

            row_output[c] = scale * (static_cast<float>(val0) - default_zp_4bit);
            if (c + 1 < block_end) {
              row_output[c + 1] = scale * (static_cast<float>(val1) - default_zp_4bit);
            }
          }
        }
      }
      return;
    }
  }

  // Generic dequantization logic for 8-bit (symmetric/asymmetric) and 4-bit (asymmetric)
  if (num_bits == 8) {
    const int64_t blocks_per_row = (block_size > 0) ? ((cols + block_size - 1) / block_size) : 1;
    if (block_size == 0) {
      // 8-bit, row-wise
      for (int64_t r = 0; r < rows; ++r) {
        const float scale = static_cast<float>(scales[r]);
        const uint8_t zero_pt = (zero_points == nullptr) ? static_cast<uint8_t>(default_zp_8bit) : zero_points[r];
        MlasDequantizeLinear(
            quantized_data + r * cols,
            dequantized_data + r * cols,
            static_cast<size_t>(cols),
            scale,
            zero_pt);
      }
    } else {
      // 8-bit, block-wise
      for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* row_data = quantized_data + r * cols;
        float* row_output = dequantized_data + r * cols;
        const uint8_t* row_zp_data = (zero_points == nullptr) ? nullptr : zero_points + r * blocks_per_row;

        int64_t c = 0;
        for (int64_t block_start = 0; block_start < cols; block_start += block_size) {
          const int64_t block_end = std::min(block_start + block_size, cols);
          const int64_t block_idx = std::min(block_start / block_size, blocks_per_row - 1);
          const int64_t scale_idx = r * blocks_per_row + block_idx;
          const float scale = static_cast<float>(scales[scale_idx]);
          const float zp = (row_zp_data == nullptr) ? default_zp_8bit : static_cast<float>(row_zp_data[block_idx]);

          for (c = block_start; c + 4 <= block_end; c += 4) {
            row_output[c] = scale * (static_cast<float>(row_data[c]) - zp);
            row_output[c + 1] = scale * (static_cast<float>(row_data[c + 1]) - zp);
            row_output[c + 2] = scale * (static_cast<float>(row_data[c + 2]) - zp);
            row_output[c + 3] = scale * (static_cast<float>(row_data[c + 3]) - zp);
          }
          for (; c < block_end; ++c) {
            row_output[c] = scale * (static_cast<float>(row_data[c]) - zp);
          }
        }
      }
    }
  } else if (num_bits == 2 || num_bits == 4) {
    const uint8_t value_mask = static_cast<uint8_t>((1u << num_bits) - 1u);
    const uint8_t default_zero_point = static_cast<uint8_t>(1u << (num_bits - 1));
    const uint8_t default_zp_packed = GetPackedZeroPointValue(num_bits, default_zero_point);
    const int64_t pack_size = 8 / num_bits;
    const int64_t packed_cols = (cols + pack_size - 1) / pack_size;
    const int64_t blocks_per_row = (block_size > 0) ? ((cols + block_size - 1) / block_size) : 1;
    const int64_t blocks_per_row_packed = (blocks_per_row + zp_pack_size - 1) / zp_pack_size;

    for (int64_t r = 0; r < rows; ++r) {
      const uint8_t* row_data = quantized_data + r * packed_cols;
      float* row_output = dequantized_data + r * cols;

      if (block_size > 0) {
        const uint8_t* row_zp_data = (zero_points == nullptr) ? nullptr : zero_points + r * blocks_per_row_packed;
        for (int64_t block_start = 0; block_start < cols; block_start += block_size) {
          const int64_t block_end = std::min(block_start + block_size, cols);
          const int64_t block_idx = std::min(block_start / block_size, blocks_per_row - 1);
          const int64_t scale_idx = r * blocks_per_row + block_idx;
          const float scale = static_cast<float>(scales[scale_idx]);

          const uint8_t packed_zp = (row_zp_data == nullptr) ? default_zp_packed : row_zp_data[block_idx / zp_pack_size];
          const int zp_shift = static_cast<int>((block_idx % zp_pack_size) * num_bits);
          const float zp = static_cast<float>((packed_zp >> zp_shift) & value_mask);

          for (int64_t c = block_start; c < block_end; ++c) {
            const uint8_t packed_val = row_data[c / pack_size];
            const int shift = static_cast<int>((c % pack_size) * num_bits);
            const uint8_t value = static_cast<uint8_t>((packed_val >> shift) & value_mask);
            row_output[c] = scale * (static_cast<float>(value) - zp);
          }
        }
      } else {
        const uint8_t packed_zp = (zero_points == nullptr) ? default_zp_packed : zero_points[r / zp_pack_size];
        const int zp_shift = static_cast<int>((r % zp_pack_size) * num_bits);
        const float zp = static_cast<float>((packed_zp >> zp_shift) & value_mask);
        const float scale = static_cast<float>(scales[r]);

        for (int64_t c = 0; c < cols; ++c) {
          const uint8_t packed_val = row_data[c / pack_size];
          const int shift = static_cast<int>((c % pack_size) * num_bits);
          const uint8_t value = static_cast<uint8_t>((packed_val >> shift) & value_mask);
          row_output[c] = scale * (static_cast<float>(value) - zp);
        }
      }
    }
  }
}

template <typename TScale>
void DequantizeBlock(const uint8_t* quantized_data,
                     const TScale* scales,
                     const uint8_t* zero_points,
                     int64_t block_size,
                     int64_t num_bits,
                     int64_t rows,
                     int64_t cols,
                     float* dequantized_data,
                     MLAS_THREADPOOL* thread_pool = nullptr) {
  DequantizeBlockWithMlas(quantized_data, scales, zero_points, block_size, num_bits, rows, cols, dequantized_data, thread_pool);
}

template <typename TScale>
void DequantizePrePacked(const uint8_t* prepacked_data,
                         const TScale* scales,
                         const uint8_t* zero_points,
                         int64_t block_size,
                         int64_t rows,
                         int64_t cols,
                         float* dequantized_data,
                         const gsl::span<const int64_t>& scale_dims) {
  // TODO(tlwu): Generalize this helper if we add prepacked 2-bit QMoE support.
  // The current prepack path is intentionally 4-bit-only.
  // prepacked_data is [cols, rows] (transposed, unpacked)
  // dequantized_data is [cols, rows] (transposed)
  // scales, zero_points correspond to original [rows, cols] layout

  const float default_zp_4bit = 8.0f;
  const int64_t blocks_per_row = (block_size > 0) ? ((cols + block_size - 1) / block_size) : 1;
  const int64_t zp_pack_size = 2;  // Always 2 for 4-bit

  // Iterate over Columns (K) then Rows (N) because prepacked_data is [K, N]
  for (int64_t c = 0; c < cols; ++c) {
    for (int64_t r = 0; r < rows; ++r) {
      uint8_t val = prepacked_data[c * rows + r];

      int64_t block_idx = (block_size > 0) ? (c / block_size) : 0;
      if (block_size > 0) block_idx = std::min(block_idx, blocks_per_row - 1);

      int64_t scale_idx;
      if (scale_dims.size() == 3 && scale_dims[2] > 1) {  // block-wise
        scale_idx = r * blocks_per_row + block_idx;
      } else {  // per-channel
        scale_idx = r;
      }

      float scale = static_cast<float>(scales[scale_idx]);
      float zp = default_zp_4bit;

      if (zero_points != nullptr) {
        int64_t zp_idx;
        bool is_lower_nibble;

        if (scale_dims.size() == 3 && scale_dims[2] > 1) {  // block-wise
          int64_t zp_blocks_packed = (blocks_per_row + zp_pack_size - 1) / zp_pack_size;
          zp_idx = r * zp_blocks_packed + block_idx / 2;
          is_lower_nibble = (block_idx % 2 == 0);
        } else {
          zp_idx = r / 2;
          is_lower_nibble = (r % 2 == 0);
        }

        uint8_t packed_zp = zero_points[zp_idx];
        zp = is_lower_nibble ? static_cast<float>(packed_zp & 0x0F) : static_cast<float>(packed_zp >> 4);
      }

      dequantized_data[c * rows + r] = scale * (static_cast<float>(val) - zp);
    }
  }
}

template <typename TScale>
Status BuildDirectQ4PackedBCache(const uint8_t* prepacked_weights,
                                 const TScale* scales_data,
                                 int64_t num_experts,
                                 int64_t rows,
                                 int64_t cols,
                                 int64_t block_size,
                                 const gsl::span<const int64_t>& scales_dims,
                                 MLAS_BLK_QUANT_TYPE qtype,
                                 AllocatorPtr allocator,
                                 IAllocatorUniquePtr<void>& packed_b) {
  const size_t packed_size = MlasQ4GemmPackBSize(qtype, static_cast<size_t>(rows), static_cast<size_t>(cols));
  if (packed_size == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to compute MLAS Q4 packed size for cache");
  }

  const bool is_block_wise = (scales_dims.size() == 3 && scales_dims[2] > 1);
  const int64_t scales_expert_stride = is_block_wise ? (rows * scales_dims[2]) : rows;
  const size_t prepacked_expert_stride = static_cast<size_t>(rows * cols);
  const size_t total_packed_size = packed_size * static_cast<size_t>(num_experts);

  packed_b = IAllocator::MakeUniquePtr<void>(allocator, total_packed_size, true);
  uint8_t* packed_b_ptr = static_cast<uint8_t*>(packed_b.get());

  std::vector<float> dequantized_transposed(static_cast<size_t>(rows * cols));
  for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const uint8_t* expert_prepacked = prepacked_weights + static_cast<size_t>(expert_idx) * prepacked_expert_stride;
    const TScale* expert_scales = scales_data + expert_idx * scales_expert_stride;

    DequantizePrePacked(expert_prepacked, expert_scales, nullptr, block_size, rows, cols,
                        dequantized_transposed.data(), scales_dims);

    MlasQ4GemmPackB(qtype, packed_b_ptr + expert_idx * packed_size, dequantized_transposed.data(),
                    static_cast<size_t>(rows), static_cast<size_t>(cols), static_cast<size_t>(rows));
  }

  return Status::OK();
}

template <typename T>
Status QMoECPU<T>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                           /*out*/ bool& is_packed,
                           /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;

  // If scales are prepacked, they are constant initializers.
  if (input_idx == 3) {
    return Status::OK();
  }
  if (input_idx == 6) {
    return Status::OK();
  }

  // Only support PrePack for FC1 (2) and FC2 (5) weights.
  // 4-bit uses the existing unpacked-transposed/Q4 cache path.
  // 2-bit block-wise uses a direct LUT GEMM packed cache when supported.
  if (expert_weight_bits_ != 4 && expert_weight_bits_ != 2) {
    return Status::OK();
  }

  if (input_idx == 2 || input_idx == 5) {
    const auto& shape = tensor.Shape();
    const int64_t num_experts = shape[0];
    const int64_t rows = shape[1];
    const int64_t cols_packed = shape[2];
    const int64_t pack_size = 8 / expert_weight_bits_;
    const int64_t cols = cols_packed * pack_size;

    if (input_idx == 2) {
      fc1_shape_ = shape;
    } else if (input_idx == 5) {
      fc2_shape_ = shape;
    }

    if (expert_weight_bits_ == 2) {
      if (block_size_ <= 0 || (cols % block_size_) != 0 || !prepacked_weights) {
        return Status::OK();
      }

      const int scales_idx = (input_idx == 2) ? 3 : 6;
      const int zp_idx = (input_idx == 2) ? 11 : 12;
      const Tensor* scales_tensor = nullptr;
      if (!Info().TryGetConstantInput(scales_idx, &scales_tensor) || scales_tensor == nullptr) {
        return Status::OK();
      }

      const auto& scales_dims = scales_tensor->Shape().GetDims();
      if (scales_dims.size() != 3 || scales_dims[2] <= 1) {
        return Status::OK();
      }

      const bool has_zp_input = zp_idx < static_cast<int>(Info().node().InputDefs().size()) &&
                                Info().node().InputDefs()[zp_idx]->Exists();
      const Tensor* zp_tensor = nullptr;
      if (has_zp_input && !Info().TryGetConstantInput(zp_idx, &zp_tensor)) {
        return Status::OK();
      }

      if (!CanUseMlasLutGemm(expert_weight_bits_, block_size_, rows, cols)) {
        return Status::OK();
      }

      IAllocatorUniquePtr<void> lut_cache_buffer;
      const uint8_t* zp_data = zp_tensor ? zp_tensor->Data<uint8_t>() : nullptr;
      ORT_RETURN_IF_ERROR(BuildDirectLutPackedBCache(static_cast<const uint8_t*>(tensor.DataRaw()),
                                                     scales_tensor->Data<T>(),
                                                     zp_data,
                                                     num_experts,
                                                     rows,
                                                     cols,
                                                     block_size_,
                                                     scales_dims[2],
                                                     alloc,
                                                     lut_cache_buffer));

      const size_t cache_size = MlasLutGemmPackedSize(static_cast<size_t>(rows), static_cast<size_t>(cols), 2,
                                                      static_cast<size_t>(block_size_), zp_data != nullptr) *
                                static_cast<size_t>(num_experts);
      prepacked_weights->buffers_.push_back(std::move(lut_cache_buffer));
      prepacked_weights->buffer_sizes_.push_back(cache_size);
      is_packed = true;

      auto dims = shape.GetDims();
      size_t rank_bytes = sizeof(int64_t);
      size_t dims_bytes = dims.size() * sizeof(int64_t);
      size_t shape_size = rank_bytes + dims_bytes;

      auto shape_buffer = IAllocator::MakeUniquePtr<void>(alloc, shape_size);
      int64_t* buffer_data = static_cast<int64_t*>(shape_buffer.get());
      *buffer_data = static_cast<int64_t>(dims.size());
      memcpy(buffer_data + 1, dims.data(), dims_bytes);

      prepacked_weights->buffers_.push_back(std::move(shape_buffer));
      prepacked_weights->buffer_sizes_.push_back(shape_size);
      return Status::OK();
    }

    size_t packed_size = static_cast<size_t>(num_experts * rows * cols);
    auto packed_buffer = IAllocator::MakeUniquePtr<void>(alloc, packed_size, true);
    uint8_t* dst_base = static_cast<uint8_t*>(packed_buffer.get());
    const uint8_t* src_base = static_cast<const uint8_t*>(tensor.DataRaw());

    for (int64_t i = 0; i < num_experts; ++i) {
      const uint8_t* src = src_base + i * rows * cols_packed;
      uint8_t* dst = dst_base + i * rows * cols;

      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          uint8_t packed_val = src[r * cols_packed + (c / 2)];
          uint8_t val = (c % 2 == 0) ? (packed_val & 0x0F) : (packed_val >> 4);

          dst[c * rows + r] = val;
        }
      }
    }

    if (prepacked_weights) {
      prepacked_weights->buffers_.push_back(std::move(packed_buffer));
      prepacked_weights->buffer_sizes_.push_back(packed_size);
      is_packed = true;

      // Pack Shape (Buffer 1)
      auto dims = shape.GetDims();
      size_t rank_bytes = sizeof(int64_t);
      size_t dims_bytes = dims.size() * sizeof(int64_t);
      size_t shape_size = rank_bytes + dims_bytes;

      auto shape_buffer = IAllocator::MakeUniquePtr<void>(alloc, shape_size);
      int64_t* buffer_data = static_cast<int64_t*>(shape_buffer.get());
      *buffer_data = static_cast<int64_t>(dims.size());
      memcpy(buffer_data + 1, dims.data(), dims_bytes);

      prepacked_weights->buffers_.push_back(std::move(shape_buffer));
      prepacked_weights->buffer_sizes_.push_back(shape_size);

      // Try build MLAS Q4 cache if scales are available
      if (use_mlas_q4_gemm_) {
        const Tensor* scales_tensor = nullptr;
        MLAS_BLK_QUANT_TYPE qtype = BlkQ4Sym;
        int scales_idx = -1;
        int zp_idx = -1;

        if (input_idx == 2) {  // FC1
          scales_idx = 3;
          zp_idx = 11;
        } else if (input_idx == 5) {  // FC2
          scales_idx = 6;
          zp_idx = 12;
        }

        if (scales_idx != -1 &&
            (zp_idx >= static_cast<int>(Info().node().InputDefs().size()) || !Info().node().InputDefs()[zp_idx]->Exists()) &&
            Info().TryGetConstantInput(scales_idx, &scales_tensor) &&
            scales_tensor != nullptr &&
            CanUseMlasQ4Gemm(expert_weight_bits_, block_size_, rows, cols, qtype)) {
          IAllocatorUniquePtr<void> cache_buffer;
          const auto& scales_dims = scales_tensor->Shape().GetDims();
          const T* scales_data = scales_tensor->Data<T>();
          // Use the simple packed buffer we just created (buffer 0) as input
          const uint8_t* simple_packed = dst_base;

          if (BuildDirectQ4PackedBCache(simple_packed, scales_data, num_experts, rows, cols,
                                        block_size_, scales_dims, qtype,
                                        alloc, cache_buffer)
                  .IsOK()) {
            // Store the MLAS Q4 cache as buffer 2 (after unpacked weights and shape).
            size_t cache_size = MlasQ4GemmPackBSize(qtype, static_cast<size_t>(rows), static_cast<size_t>(cols)) * static_cast<size_t>(num_experts);
            prepacked_weights->buffers_.push_back(std::move(cache_buffer));
            prepacked_weights->buffer_sizes_.push_back(cache_size);
          }
        }
      }
    }
  }

  return Status::OK();
}

template <typename T>
Status QMoECPU<T>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                             gsl::span<const size_t> /*prepacked_buffer_sizes*/,
                                             int input_idx,
                                             /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;

  if (expert_weight_bits_ == 2) {
    if ((input_idx == 2 || input_idx == 5) && !prepacked_buffers.empty()) {
      auto parse_shape = [&](TensorShape& shape) {
        if (prepacked_buffers.size() > 1) {
          int64_t* buffer_data = static_cast<int64_t*>(prepacked_buffers[1].get());
          int64_t rank = buffer_data[0];
          std::vector<int64_t> dims(static_cast<size_t>(rank));
          memcpy(dims.data(), buffer_data + 1, static_cast<size_t>(rank) * sizeof(int64_t));
          shape = TensorShape(dims);
        }
      };

      if (input_idx == 2) {
        packed_fc1_lut_cache_ = std::move(prepacked_buffers[0]);
        parse_shape(fc1_shape_);
      } else {
        packed_fc2_lut_cache_ = std::move(prepacked_buffers[0]);
        parse_shape(fc2_shape_);
      }

      used_shared_buffers = true;
    }

    return Status::OK();
  }

  if (expert_weight_bits_ != 4) {
    return Status::OK();
  }

  if ((input_idx == 2 || input_idx == 5) && !prepacked_buffers.empty()) {
    auto parse_shape = [&](TensorShape& shape) {
      if (prepacked_buffers.size() > 1) {
        int64_t* buffer_data = static_cast<int64_t*>(prepacked_buffers[1].get());
        int64_t rank = buffer_data[0];
        std::vector<int64_t> dims(static_cast<size_t>(rank));
        memcpy(dims.data(), buffer_data + 1, static_cast<size_t>(rank) * sizeof(int64_t));
        shape = TensorShape(dims);
      }
    };

    if (input_idx == 2) {
      packed_fc1_ = std::move(prepacked_buffers[0]);
      parse_shape(fc1_shape_);
      if (prepacked_buffers.size() > 2) {
        packed_fc1_mlas_cache_ = std::move(prepacked_buffers[2]);
      }
    } else if (input_idx == 5) {
      packed_fc2_ = std::move(prepacked_buffers[0]);
      parse_shape(fc2_shape_);
      if (prepacked_buffers.size() > 2) {
        packed_fc2_mlas_cache_ = std::move(prepacked_buffers[2]);
      }
    }
    used_shared_buffers = true;
  }

  return Status::OK();
}

template <typename T>
QMoECPU<T>::QMoECPU(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info),
      MoEBaseCPU(op_kernel_info) {
  ORT_ENFORCE(activation_type_ != ActivationType::SwiGLU || swiglu_fusion_ == 1,
              "CPU QMoE only supports interleaved SwiGLU format. Please set swiglu_fusion=1.");
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 2 || expert_weight_bits_ == 4 || expert_weight_bits_ == 8,
              "Attribute 'expert_weight_bits' must be 2, 4, or 8.");
  block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", 0);
  ORT_ENFORCE(block_size_ >= 0);

  if (block_size_ > 0) {
    ORT_ENFORCE(block_size_ >= 16, "block_size must be >= 16 when provided.");
    ORT_ENFORCE((block_size_ & (block_size_ - 1)) == 0, "block_size must be a power of 2.");
  }

  const auto use_mlas_q4_gemm = ParseEnvironmentVariable<bool>(kUseMlasQ4GemmMoe);
  if (use_mlas_q4_gemm.has_value()) {
    use_mlas_q4_gemm_ = *use_mlas_q4_gemm;
    use_mlas_q4_gemm_overridden_ = true;
  } else {
    // Default policy: enable fast path unless this run hits a known accuracy-loss configuration.
    use_mlas_q4_gemm_ = true;
    use_mlas_q4_gemm_overridden_ = false;
  }
}

template <typename T>
Status QMoECPU<T>::Compute(OpKernelContext* context) const {
  const ComputeInputs inputs{
      context->Input<Tensor>(0),
      context->Input<Tensor>(1),
      ((packed_fc1_ != nullptr) || (packed_fc1_lut_cache_ != nullptr)) ? nullptr : context->Input<Tensor>(2),
      context->Input<Tensor>(3),
      context->Input<Tensor>(4),
      ((packed_fc2_ != nullptr) || (packed_fc2_lut_cache_ != nullptr)) ? nullptr : context->Input<Tensor>(5),
      context->Input<Tensor>(6),
      context->Input<Tensor>(7),
      context->Input<Tensor>(8),
      context->Input<Tensor>(9),
      context->Input<Tensor>(10),
      context->Input<Tensor>(11),
      context->Input<Tensor>(12),
      context->Input<Tensor>(13),
      context->Input<Tensor>(14),
  };

  const bool has_fc1_prepacked = (packed_fc1_ != nullptr) || (packed_fc1_lut_cache_ != nullptr);
  const bool has_fc2_prepacked = (packed_fc2_ != nullptr) || (packed_fc2_lut_cache_ != nullptr);

  const TensorShape* fc1_shape_ptr = has_fc1_prepacked ? &fc1_shape_ : (inputs.fc1_experts_weights ? &inputs.fc1_experts_weights->Shape() : nullptr);
  const TensorShape* fc2_shape_ptr = has_fc2_prepacked ? &fc2_shape_ : (inputs.fc2_experts_weights ? &inputs.fc2_experts_weights->Shape() : nullptr);
  const TensorShape* fc3_shape_ptr = inputs.fc3_experts_weights ? &inputs.fc3_experts_weights->Shape() : nullptr;

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(moe_helper::CheckInputs<Tensor>(
      moe_params, inputs.input, inputs.router_probs,
      fc1_shape_ptr, inputs.fc1_experts_bias, inputs.fc1_scales, inputs.fc1_zero_points,
      fc2_shape_ptr, inputs.fc2_experts_bias, inputs.fc2_scales, inputs.fc2_zero_points,
      fc3_shape_ptr, inputs.fc3_experts_bias, inputs.fc3_scales, inputs.fc3_zero_points,
      8 / expert_weight_bits_,
      activation_type_ == ActivationType::SwiGLU,
      block_size_));

  if (fc3_shape_ptr || inputs.fc3_experts_bias || inputs.fc3_scales || inputs.fc3_zero_points) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "FC3 gating is not yet implemented on CPU for QMoE");
  }

  return ComputeCommon(context, inputs, moe_params);
}

template <typename T>
Status QMoECPU<T>::ComputeCommon(OpKernelContext* context, const ComputeInputs& inputs, const MoEParameters& moe_params) const {
  const auto* input = inputs.input;
  const auto* router_probs = inputs.router_probs;
  const auto* fc1_experts_weights = inputs.fc1_experts_weights;
  const auto* fc1_scales = inputs.fc1_scales;
  const auto* fc1_experts_bias = inputs.fc1_experts_bias;
  const auto* fc2_experts_weights = inputs.fc2_experts_weights;
  const auto* fc2_scales = inputs.fc2_scales;
  const auto* fc2_experts_bias = inputs.fc2_experts_bias;
  const auto* fc1_zero_points = inputs.fc1_zero_points;
  const auto* fc2_zero_points = inputs.fc2_zero_points;
  const auto* router_weights = inputs.router_weights;

  const auto& input_shape = input->Shape();
  const int64_t num_tokens = moe_params.num_rows;
  const int64_t hidden_size = moe_params.hidden_size;
  const int64_t inter_size = moe_params.inter_size;
  const int64_t num_experts = moe_params.num_experts;
  const int64_t fc1_out_features = inter_size * (swiglu_fusion_ > 0 ? 2 : 1);

  auto* output = context->Output(0, input_shape);
  auto* tp = context->GetOperatorThreadPool();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  const size_t output_buffer_size = static_cast<size_t>(output->Shape().Size());

  const T* input_data = input->template Data<T>();

  IAllocatorUniquePtr<float> router_logits_float_buffer;
  const float* router_logits_float;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    router_logits_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * num_experts));
    router_logits_float = router_logits_float_buffer.get();
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(router_probs->template Data<T>()),
                                 const_cast<float*>(router_logits_float),
                                 static_cast<size_t>(num_tokens * num_experts));
  } else {
    router_logits_float = reinterpret_cast<const float*>(router_probs->template Data<T>());
  }

  // Handle optional router_weights input for separate selection/aggregation tensors
  const bool has_router_weights = (router_weights != nullptr);
  IAllocatorUniquePtr<float> router_weights_float_buffer;
  const float* router_weights_float = nullptr;
  if (has_router_weights) {
    const auto& rw_shape = router_weights->Shape();
    if (rw_shape.NumDimensions() != 2 || rw_shape[0] != num_tokens || rw_shape[1] != num_experts) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'router_weights' is expected to have shape (",
                             num_tokens, ", ", num_experts, "), got ", rw_shape);
    }
    if constexpr (std::is_same_v<T, MLFloat16>) {
      router_weights_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * num_experts));
      router_weights_float = router_weights_float_buffer.get();
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(router_weights->template Data<T>()),
                                   const_cast<float*>(router_weights_float),
                                   static_cast<size_t>(num_tokens * num_experts));
    } else {
      router_weights_float = reinterpret_cast<const float*>(router_weights->template Data<T>());
    }
  }

  auto route_expert_ptr = IAllocator::MakeUniquePtr<int>(allocator, static_cast<size_t>(num_tokens * k_));
  int* route_expert = route_expert_ptr.get();
  auto route_scale_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * k_));
  float* route_scale = route_scale_ptr.get();

  const int max_threads = tp ? concurrency::ThreadPool::DegreeOfParallelism(tp) : 1;
  const int64_t thread_divisor = std::max(1, max_threads * 4);
  const int64_t min_work_per_thread = std::max(int64_t{32}, static_cast<int64_t>(num_tokens / thread_divisor));
  const int optimal_routing_threads = (tp == nullptr || num_tokens < min_work_per_thread) ? 1 : std::min(narrow<int>(num_tokens / std::max(int64_t{1}, min_work_per_thread)), max_threads);
  const int num_routing_threads = std::max(1, optimal_routing_threads);

  std::vector<std::vector<std::vector<int64_t>>> thread_local_expert_token_maps(num_routing_threads);
  for (auto& map : thread_local_expert_token_maps) {
    map.resize(static_cast<size_t>(num_experts));
    for (auto& expert_tokens : map) {
      expert_tokens.reserve(32);
    }
  }

  concurrency::ThreadPool::TrySimpleParallelFor(tp, num_routing_threads, [&](std::ptrdiff_t thread_id) {
    auto work = concurrency::ThreadPool::PartitionWork(narrow<int>(thread_id), num_routing_threads, static_cast<std::ptrdiff_t>(num_tokens));
    auto& local_expert_token_map = thread_local_expert_token_maps[thread_id];

    std::vector<std::pair<float, int64_t>> sorted_logits(static_cast<size_t>(num_experts));
    std::vector<float> top_k_exp(static_cast<size_t>(k_));

    for (int64_t i = work.start; i < work.end; ++i) {
      const float* logits = router_logits_float + i * num_experts;

      for (size_t j = 0; j < narrow<size_t>(num_experts); ++j) {
        sorted_logits[j] = {logits[j], j};
      }
      std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + static_cast<std::ptrdiff_t>(k_),
                        sorted_logits.end(), std::greater<>());

      if (has_router_weights) {
        // When router_weights is provided, use it for aggregation weights instead of softmax of router_probs.
        // Gather weights from router_weights at the selected expert indices.
        // Note: top_k_exp is reused here as a scratch buffer for the gathered weights.
        const float* weights_row = router_weights_float + i * num_experts;
        if (normalize_routing_weights_) {
          float weight_sum = 0.0f;
          for (size_t j = 0; j < narrow<size_t>(k_); ++j) {
            int64_t expert_idx = sorted_logits[j].second;
            top_k_exp[j] = weights_row[expert_idx];
            weight_sum += top_k_exp[j];
          }
          const float inv_weight_sum = (weight_sum == 0.0f) ? 0.0f : (1.0f / weight_sum);
          for (size_t j = 0; j < narrow<size_t>(k_); ++j) {
            int64_t expert_idx = sorted_logits[j].second;
            int64_t route_idx = i * k_ + narrow<int64_t>(j);
            route_expert[route_idx] = narrow<int>(expert_idx);
            route_scale[route_idx] = top_k_exp[j] * inv_weight_sum;
            if (route_scale[route_idx] > 1e-8f) {
              local_expert_token_map[static_cast<size_t>(expert_idx)].push_back(route_idx);
            }
          }
        } else {
          for (size_t j = 0; j < narrow<size_t>(k_); ++j) {
            int64_t expert_idx = sorted_logits[j].second;
            int64_t route_idx = i * k_ + narrow<int64_t>(j);
            route_expert[route_idx] = narrow<int>(expert_idx);
            route_scale[route_idx] = weights_row[expert_idx];
            if (route_scale[route_idx] > 1e-8f) {
              local_expert_token_map[static_cast<size_t>(expert_idx)].push_back(route_idx);
            }
          }
        }
      } else {
        // Default path: compute softmax weights from router_probs for aggregation.
        float max_logit = sorted_logits[0].first;

        float sum_exp = 0.0f;
        for (size_t j = 0; j < narrow<size_t>(k_); ++j) {
          top_k_exp[j] = std::exp(sorted_logits[j].first - max_logit);
          sum_exp += top_k_exp[j];
        }

        const float inv_sum = (sum_exp == 0.0f) ? 0.0f : (1.0f / sum_exp);
        for (size_t j = 0; j < narrow<size_t>(k_); ++j) {
          int64_t expert_idx = sorted_logits[j].second;
          int64_t route_idx = i * k_ + narrow<int64_t>(j);
          route_expert[route_idx] = narrow<int>(expert_idx);
          route_scale[route_idx] = top_k_exp[j] * inv_sum;
          if (route_scale[route_idx] > 1e-8f) {  // Use small threshold to avoid zero weights
            local_expert_token_map[static_cast<size_t>(expert_idx)].push_back(route_idx);
          }
        }
      }
    }
  });

  std::vector<std::vector<int64_t>> expert_token_map(static_cast<size_t>(num_experts));
  for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    size_t total_tokens_for_expert = 0;
    for (int t = 0; t < num_routing_threads; ++t) {
      total_tokens_for_expert += thread_local_expert_token_maps[t][static_cast<size_t>(expert_idx)].size();
    }
    expert_token_map[static_cast<size_t>(expert_idx)].reserve(total_tokens_for_expert);

    for (int t = 0; t < num_routing_threads; ++t) {
      auto& local_tokens = thread_local_expert_token_maps[t][static_cast<size_t>(expert_idx)];
      if (!local_tokens.empty()) {
        expert_token_map[static_cast<size_t>(expert_idx)].insert(
            expert_token_map[static_cast<size_t>(expert_idx)].end(),
            local_tokens.begin(), local_tokens.end());
      }
    }
  }

  IAllocatorUniquePtr<float> input_float_buffer;
  const float* input_float;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    input_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_tokens * hidden_size));
    input_float = input_float_buffer.get();
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(input_data),
                                 const_cast<float*>(input_float),
                                 static_cast<size_t>(num_tokens * hidden_size));
  } else {
    input_float = reinterpret_cast<const float*>(input_data);
  }

  const int max_expert_threads = tp ? concurrency::ThreadPool::DegreeOfParallelism(tp) : 1;
  const int64_t total_expert_work = std::accumulate(expert_token_map.begin(), expert_token_map.end(), 0LL,
                                                    [](int64_t sum, const std::vector<int64_t>& tokens) { return sum + static_cast<int64_t>(tokens.size()); });
  const int64_t expert_thread_divisor = std::max(1, max_expert_threads * 8);
  const int64_t min_expert_work_per_thread = std::max(int64_t{16}, total_expert_work / expert_thread_divisor);

  int num_expert_threads = (tp == nullptr || total_expert_work < min_expert_work_per_thread) ? 1 : std::min(narrow<int>(total_expert_work / std::max(int64_t{1}, min_expert_work_per_thread)), std::min(narrow<int>(num_experts), max_expert_threads));
  if (num_expert_threads == 0) num_expert_threads = 1;

  auto thread_local_outputs_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_expert_threads) * output_buffer_size);
  float* thread_local_outputs = thread_local_outputs_ptr.get();
  std::memset(thread_local_outputs, 0, static_cast<size_t>(num_expert_threads) * output_buffer_size * sizeof(float));

  size_t max_tokens_per_expert = 0;
  for (const auto& tokens : expert_token_map) {
    max_tokens_per_expert = std::max(max_tokens_per_expert, tokens.size());
  }

  const auto align_size = [](size_t size) -> size_t {
    return (size + 63) & ~63;
  };

  const size_t A1_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(hidden_size));
  const size_t C1_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(fc1_out_features));
  const size_t A2_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(inter_size));
  const size_t C2_size = align_size(static_cast<size_t>(max_tokens_per_expert) * static_cast<size_t>(hidden_size));
  const size_t B1_dequant_size = align_size(static_cast<size_t>(fc1_out_features) * static_cast<size_t>(hidden_size));
  const size_t B2_dequant_size = align_size(static_cast<size_t>(hidden_size) * static_cast<size_t>(inter_size));

  const size_t workspace_elements_per_thread = A1_size + C1_size + A2_size + C2_size +
                                               B1_dequant_size + B2_dequant_size;

  auto workspace_ptr = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_expert_threads) * workspace_elements_per_thread);
  float* workspace = workspace_ptr.get();

  auto bias_conversion_buffers_ptr = IAllocator::MakeUniquePtr<float>(allocator,
                                                                      static_cast<size_t>(num_expert_threads) * (static_cast<size_t>(fc1_out_features) + static_cast<size_t>(hidden_size)));
  float* bias_conversion_buffers = bias_conversion_buffers_ptr.get();

  const auto& fc1_scales_dims = fc1_scales->Shape().GetDims();
  const auto& fc2_scales_dims = fc2_scales->Shape().GetDims();
  const bool is_fc1_block_wise = (fc1_scales_dims.size() == 3 && fc1_scales_dims[2] > 1);
  const bool is_fc2_block_wise = (fc2_scales_dims.size() == 3 && fc2_scales_dims[2] > 1);

  const uint8_t* fc1_weights_data = (packed_fc1_ != nullptr) ? nullptr : fc1_experts_weights->template Data<uint8_t>();
  const uint8_t* fc2_weights_data = (packed_fc2_ != nullptr) ? nullptr : fc2_experts_weights->template Data<uint8_t>();
  const T* fc1_scales_data = fc1_scales->template Data<T>();
  const T* fc2_scales_data = fc2_scales->template Data<T>();
  const T* fc1_bias_data = fc1_experts_bias ? fc1_experts_bias->template Data<T>() : nullptr;
  const T* fc2_bias_data = fc2_experts_bias ? fc2_experts_bias->template Data<T>() : nullptr;
  const uint8_t* fc1_zp_data = fc1_zero_points ? fc1_zero_points->template Data<uint8_t>() : nullptr;
  const uint8_t* fc2_zp_data = fc2_zero_points ? fc2_zero_points->template Data<uint8_t>() : nullptr;

  // Known loss-prone case from parity testing: 4-bit symmetric path (row-wise and block-wise).
  const bool known_accuracy_loss_case = (expert_weight_bits_ == 4) &&
                                        (fc1_zp_data == nullptr) && (fc2_zp_data == nullptr);
  const bool use_mlas_q4_gemm_effective = use_mlas_q4_gemm_overridden_
                                              ? use_mlas_q4_gemm_
                                              : (use_mlas_q4_gemm_ && !known_accuracy_loss_case);
  const bool use_mlas_lut_gemm_effective = (expert_weight_bits_ == 2);

  const int64_t pack_unit = (8 / expert_weight_bits_);
  const int64_t fc1_packed_cols = (hidden_size + pack_unit - 1) / pack_unit;
  const int64_t fc2_packed_cols = (inter_size + pack_unit - 1) / pack_unit;
  const bool has_fc1_bias = (fc1_bias_data != nullptr);
  const bool has_fc2_bias = (fc2_bias_data != nullptr);

  // Calculate strides for zero-point tensors
  const int64_t zp_pack_size = 8 / expert_weight_bits_;
  int64_t fc1_zp_expert_stride = 0;
  int64_t fc2_zp_expert_stride = 0;

  if (is_fc1_block_wise) {
    const int64_t fc1_blocks_per_row = (hidden_size + block_size_ - 1) / block_size_;
    const int64_t fc1_zp_blocks_packed = (fc1_blocks_per_row + zp_pack_size - 1) / zp_pack_size;
    fc1_zp_expert_stride = fc1_out_features * fc1_zp_blocks_packed;
  } else {
    fc1_zp_expert_stride = (fc1_out_features + zp_pack_size - 1) / zp_pack_size;
  }

  if (is_fc2_block_wise) {
    const int64_t fc2_blocks_per_row = (inter_size + block_size_ - 1) / block_size_;
    const int64_t fc2_zp_blocks_packed = (fc2_blocks_per_row + zp_pack_size - 1) / zp_pack_size;
    fc2_zp_expert_stride = hidden_size * fc2_zp_blocks_packed;
  } else {
    fc2_zp_expert_stride = (hidden_size + zp_pack_size - 1) / zp_pack_size;
  }

  MLAS_BLK_QUANT_TYPE fc1_direct_qtype = BlkQ4Sym;
  MLAS_BLK_QUANT_TYPE fc2_direct_qtype = BlkQ4Sym;
  const bool can_use_fc1_lut_gemm = use_mlas_lut_gemm_effective &&
                                    is_fc1_block_wise &&
                                    CanUseMlasLutGemm(expert_weight_bits_, block_size_, fc1_out_features, hidden_size);
  const bool can_use_fc2_lut_gemm = use_mlas_lut_gemm_effective &&
                                    is_fc2_block_wise &&
                                    CanUseMlasLutGemm(expert_weight_bits_, block_size_, hidden_size, inter_size);

  // Use pre-packed MLAS cache if available
  const void* fc1_direct_q4_cache_ptr = nullptr;
  if (use_mlas_q4_gemm_effective && packed_fc1_mlas_cache_ && fc1_zp_data == nullptr &&
      CanUseMlasQ4Gemm(expert_weight_bits_, is_fc1_block_wise ? block_size_ : 0, fc1_out_features, hidden_size, fc1_direct_qtype)) {
    fc1_direct_q4_cache_ptr = packed_fc1_mlas_cache_.get();
  }

  const void* fc2_direct_q4_cache_ptr = nullptr;
  if (use_mlas_q4_gemm_effective && packed_fc2_mlas_cache_ && fc2_zp_data == nullptr &&
      CanUseMlasQ4Gemm(expert_weight_bits_, is_fc2_block_wise ? block_size_ : 0, hidden_size, inter_size, fc2_direct_qtype)) {
    fc2_direct_q4_cache_ptr = packed_fc2_mlas_cache_.get();
  }

  const void* fc1_direct_lut_cache_ptr = can_use_fc1_lut_gemm ? packed_fc1_lut_cache_.get() : nullptr;
  const void* fc2_direct_lut_cache_ptr = can_use_fc2_lut_gemm ? packed_fc2_lut_cache_.get() : nullptr;
  const size_t fc1_lut_packed_size = (can_use_fc1_lut_gemm && fc1_direct_lut_cache_ptr == nullptr)
                                         ? MlasLutGemmPackedSize(static_cast<size_t>(fc1_out_features),
                                                                 static_cast<size_t>(hidden_size),
                                                                 2,
                                                                 static_cast<size_t>(block_size_),
                                                                 fc1_zp_data != nullptr)
                                         : 0;
  const size_t fc2_lut_packed_size = (can_use_fc2_lut_gemm && fc2_direct_lut_cache_ptr == nullptr)
                                         ? MlasLutGemmPackedSize(static_cast<size_t>(hidden_size),
                                                                 static_cast<size_t>(inter_size),
                                                                 2,
                                                                 static_cast<size_t>(block_size_),
                                                                 fc2_zp_data != nullptr)
                                         : 0;
  const size_t lut_packed_scratch_size_per_thread = std::max(fc1_lut_packed_size, fc2_lut_packed_size);
  IAllocatorUniquePtr<std::byte> lut_packed_buffers_ptr;
  std::byte* lut_packed_buffers = nullptr;
  if (lut_packed_scratch_size_per_thread > 0) {
    lut_packed_buffers_ptr = IAllocator::MakeUniquePtr<std::byte>(allocator,
                                                                  static_cast<size_t>(num_expert_threads) * lut_packed_scratch_size_per_thread,
                                                                  true);
    lut_packed_buffers = lut_packed_buffers_ptr.get();
  }

  const size_t lut_scale_count_per_thread = std::max(static_cast<size_t>(is_fc1_block_wise ? fc1_out_features * fc1_scales_dims[2] : 0),
                                                     static_cast<size_t>(is_fc2_block_wise ? hidden_size * fc2_scales_dims[2] : 0));
  IAllocatorUniquePtr<float> lut_scale_conversion_buffers_ptr;
  float* lut_scale_conversion_buffers = nullptr;
  if constexpr (!std::is_same_v<T, float>) {
    if (lut_scale_count_per_thread > 0) {
      lut_scale_conversion_buffers_ptr = IAllocator::MakeUniquePtr<float>(allocator,
                                                                          static_cast<size_t>(num_expert_threads) * lut_scale_count_per_thread,
                                                                          true);
      lut_scale_conversion_buffers = lut_scale_conversion_buffers_ptr.get();
    }
  }

  std::vector<std::pair<int64_t, size_t>> expert_workload;
  size_t total_work = 0;

  for (int64_t i = 0; i < num_experts; ++i) {
    const size_t token_count = expert_token_map[static_cast<size_t>(i)].size();
    if (token_count > 0) {
      expert_workload.emplace_back(i, token_count);
      total_work += token_count;
    }
  }

  if (total_work < 48) {
    num_expert_threads = 1;
  } else if (total_work < 192) {
    num_expert_threads = std::min(num_expert_threads, 2);
  } else if (total_work < 512) {
    num_expert_threads = std::min(num_expert_threads, 4);
  }

  std::sort(expert_workload.begin(), expert_workload.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  std::vector<std::vector<int64_t>> expert_batches(num_expert_threads);
  size_t thread_idx = 0;
  for (const auto& work : expert_workload) {
    expert_batches[thread_idx].push_back(work.first);
    thread_idx = (thread_idx + 1) % static_cast<size_t>(num_expert_threads);
  }

  concurrency::ThreadPool::TrySimpleParallelFor(tp, num_expert_threads, [&](std::ptrdiff_t thread_id_pd) {
    const int thread_id = narrow<int>(thread_id_pd);
    const auto& expert_batch = expert_batches[static_cast<size_t>(thread_id)];

    float* thread_workspace = workspace + static_cast<size_t>(thread_id) * workspace_elements_per_thread;
    std::byte* thread_lut_packed_buffer = (lut_packed_buffers == nullptr)
                                              ? nullptr
                                              : (lut_packed_buffers + static_cast<size_t>(thread_id) * lut_packed_scratch_size_per_thread);
    float* thread_lut_scale_buffer = (lut_scale_conversion_buffers == nullptr)
                                         ? nullptr
                                         : (lut_scale_conversion_buffers + static_cast<size_t>(thread_id) * lut_scale_count_per_thread);

    float* thread_bias1_buffer = bias_conversion_buffers + static_cast<size_t>(thread_id) * (static_cast<size_t>(fc1_out_features) + static_cast<size_t>(hidden_size));
    float* thread_bias2_buffer = thread_bias1_buffer + static_cast<size_t>(fc1_out_features);

    for (int64_t expert_idx : expert_batch) {
      bool fc2_bias_added_by_mlas = false;
      const auto& routes = expert_token_map[static_cast<size_t>(expert_idx)];
      if (routes.empty()) {
        continue;
      }

      const int64_t num_expert_tokens = static_cast<int64_t>(routes.size());

      float* A1 = thread_workspace;
      float* C1 = A1 + A1_size;
      float* A2 = C1 + C1_size;
      float* C2 = A2 + A2_size;
      float* B1_dequant = C2 + C2_size;
      float* B2_dequant = B1_dequant + B1_dequant_size;

      const int64_t dynamic_block_size = GetOptimalBlockSize(num_expert_tokens, tp ? concurrency::ThreadPool::DegreeOfParallelism(tp) : 1);
      const int64_t num_blocks = (num_expert_tokens + dynamic_block_size - 1) / dynamic_block_size;

      if (num_expert_tokens >= 8 && num_blocks > 1 && tp != nullptr) {
        concurrency::ThreadPool::TrySimpleParallelFor(tp, narrow<int>(num_blocks), [&](std::ptrdiff_t block_idx) {
          const int64_t start_idx = block_idx * dynamic_block_size;
          const int64_t end_idx = std::min(start_idx + dynamic_block_size, num_expert_tokens);

          for (int64_t i = start_idx; i < end_idx; ++i) {
            const int64_t token_idx = routes[static_cast<size_t>(i)] / k_;
            const float* src = input_float + token_idx * hidden_size;
            float* dst = A1 + i * hidden_size;

            std::memcpy(dst, src, static_cast<size_t>(hidden_size) * sizeof(float));
          }
        });
      } else {
        for (int64_t i = 0; i < num_expert_tokens; ++i) {
          const int64_t token_idx = routes[static_cast<size_t>(i)] / k_;
          const float* src = input_float + token_idx * hidden_size;
          float* dst = A1 + i * hidden_size;

          if (ShouldUseMemcpy(hidden_size)) {
            std::memcpy(dst, src, static_cast<size_t>(hidden_size) * sizeof(float));
          } else {
            const size_t unroll_factor = narrow<size_t>(GetUnrollFactor(hidden_size));
            size_t j = 0;
            for (; j + unroll_factor <= narrow<size_t>(hidden_size); j += unroll_factor) {
              for (size_t k = 0; k < unroll_factor; ++k) {
                dst[j + k] = src[j + k];
              }
            }
            for (; j < narrow<size_t>(hidden_size); ++j) {
              dst[j] = src[j];
            }
          }
        }
      }

      const T* fc1_scales_ptr;
      const uint8_t* fc1_zp_ptr;

      if (is_fc1_block_wise) {
        const int64_t fc1_blocks_per_row = fc1_scales_dims[2];
        fc1_scales_ptr = fc1_scales_data + expert_idx * fc1_out_features * fc1_blocks_per_row;
        fc1_zp_ptr = (fc1_zp_data == nullptr) ? nullptr : fc1_zp_data + expert_idx * fc1_zp_expert_stride;
      } else {
        fc1_scales_ptr = fc1_scales_data + expert_idx * fc1_out_features;
        fc1_zp_ptr = (fc1_zp_data == nullptr) ? nullptr : fc1_zp_data + expert_idx * fc1_zp_expert_stride;
      }

      const int64_t dequant_block_size = GetDequantBlockSize(fc1_out_features, num_expert_tokens);
      const int64_t num_dequant_blocks = (fc1_out_features + dequant_block_size - 1) / dequant_block_size;

      const size_t m = static_cast<size_t>(num_expert_tokens);
      const size_t n = static_cast<size_t>(fc1_out_features);
      const size_t k = static_cast<size_t>(hidden_size);

      MLAS_BLK_QUANT_TYPE q_type = BlkQ4Sym;  // Initialize to default
      bool use_direct_q4_gemm = use_mlas_q4_gemm_effective &&
                                ((fc1_direct_q4_cache_ptr != nullptr) ||
                                 ((packed_fc1_ == nullptr) && (fc1_zp_data == nullptr) &&
                                  CanUseMlasQ4Gemm(expert_weight_bits_, is_fc1_block_wise ? block_size_ : 0,
                                                   fc1_out_features, hidden_size, q_type)));

      if (can_use_fc1_lut_gemm &&
          TryRunLutGemm(A1, C1, fc1_weights_data, fc1_direct_lut_cache_ptr,
                        fc1_scales_ptr, fc1_zp_ptr, expert_idx,
                        fc1_out_features, hidden_size, fc1_packed_cols,
                        block_size_, fc1_scales_dims[2],
                        thread_lut_packed_buffer, thread_lut_scale_buffer,
                        num_expert_tokens, tp)) {
        goto fc1_bias_handling;
      }

      if (packed_fc1_ != nullptr) {
        if (use_mlas_q4_gemm_effective && fc1_zp_data == nullptr &&
            CanUseMlasQ4Gemm(expert_weight_bits_, is_fc1_block_wise ? block_size_ : 0,
                             fc1_out_features, hidden_size, q_type)) {
          if (fc1_direct_q4_cache_ptr != nullptr) {
            float* fc1_bias_float = nullptr;
            if (has_fc1_bias) {
              const T* B1_bias = fc1_bias_data + expert_idx * fc1_out_features;
              if constexpr (std::is_same_v<T, MLFloat16>) {
                MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B1_bias), thread_bias1_buffer, static_cast<size_t>(fc1_out_features));
              } else {
                std::memcpy(thread_bias1_buffer, B1_bias, static_cast<size_t>(fc1_out_features) * sizeof(float));
              }
              fc1_bias_float = thread_bias1_buffer;
            }

            size_t packed_size = MlasQ4GemmPackBSize(q_type, static_cast<size_t>(fc1_out_features), static_cast<size_t>(hidden_size));
            const uint8_t* packed_b = static_cast<const uint8_t*>(fc1_direct_q4_cache_ptr) + expert_idx * packed_size;

            Status gemm_status = DirectQ4Gemm(A1, packed_b, fc1_bias_float, C1,
                                              num_expert_tokens, fc1_out_features, hidden_size, fc1_direct_qtype, tp);
            if (gemm_status.IsOK()) {
              goto fc1_gemm_done;
            }
          }
        }

        // Fallback: Dequantize from PrePacked (transposed, unpacked) -> MlasGemm
        const uint8_t* current_packed_ptr = static_cast<const uint8_t*>(packed_fc1_.get()) + expert_idx * fc1_out_features * hidden_size;

        DequantizePrePacked(current_packed_ptr, fc1_scales_ptr, fc1_zp_ptr,
                            is_fc1_block_wise ? block_size_ : 0,
                            fc1_out_features, hidden_size,
                            B1_dequant, fc1_scales_dims);

        // Use MlasGemm with B1_dequant (which is already float transposed)
        MlasGemm(CblasNoTrans, CblasNoTrans,
                 m, n, k,
                 1.0f, A1, k,
                 B1_dequant, n,
                 0.0f, C1, n,
                 tp, &mlas_backend_kernel_selector_config_);

        goto fc1_bias_handling;
      }

      if (use_direct_q4_gemm) {
        IAllocatorUniquePtr<uint8_t> mlas_packed_fc1;
        Status convert_status = ConvertToMlasQ4Format(
            fc1_weights_data + expert_idx * fc1_out_features * fc1_packed_cols,
            fc1_scales_ptr,
            fc1_zp_ptr,  // This will be nullptr
            is_fc1_block_wise ? block_size_ : 0,
            expert_weight_bits_,
            fc1_out_features,
            hidden_size,
            q_type,
            allocator,
            mlas_packed_fc1);

        if (convert_status.IsOK()) {
          float* fc1_bias_float = nullptr;

          if (has_fc1_bias) {
            const T* B1_bias = fc1_bias_data + expert_idx * fc1_out_features;
            fc1_bias_float = thread_bias1_buffer;

            if constexpr (std::is_same_v<T, MLFloat16>) {
              MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B1_bias), fc1_bias_float, static_cast<size_t>(fc1_out_features));
            } else {
              for (int64_t i = 0; i < fc1_out_features; ++i) {
                fc1_bias_float[i] = static_cast<float>(B1_bias[i]);
              }
            }
          }

          Status gemm_status = DirectQ4Gemm(A1, mlas_packed_fc1.get(), fc1_bias_float, C1,
                                            num_expert_tokens, fc1_out_features, hidden_size, q_type, tp);

          if (gemm_status.IsOK()) {
            goto fc1_gemm_done;
          }
        }
        // If direct Q4 GEMM failed, fall back to traditional approach
      }

      // Traditional approach: dequantize + regular GEMM
      if (num_dequant_blocks > 1 && fc1_out_features >= 32) {
        concurrency::ThreadPool::TrySimpleParallelFor(tp, narrow<int>(num_dequant_blocks), [&](std::ptrdiff_t block_idx) {
          const int64_t start_row = block_idx * dequant_block_size;
          const int64_t end_row = std::min(start_row + dequant_block_size, fc1_out_features);
          const auto offset = expert_idx * fc1_out_features * fc1_packed_cols + start_row * fc1_packed_cols;

          const T* current_scales_ptr = fc1_scales_ptr + (is_fc1_block_wise ? start_row * fc1_scales_dims[2] : start_row);
          const uint8_t* current_zp_ptr = nullptr;
          if (fc1_zp_ptr != nullptr) {
            if (is_fc1_block_wise) {
              const int64_t fc1_blocks_per_row = (hidden_size + block_size_ - 1) / block_size_;
              const int64_t fc1_zp_blocks_packed = (fc1_blocks_per_row + zp_pack_size - 1) / zp_pack_size;
              current_zp_ptr = fc1_zp_ptr + start_row * fc1_zp_blocks_packed;
            } else {
              current_zp_ptr = fc1_zp_ptr + start_row / zp_pack_size;
            }
          }

          DequantizeBlock(fc1_weights_data + offset,
                          current_scales_ptr,
                          current_zp_ptr,
                          is_fc1_block_wise ? block_size_ : 0, expert_weight_bits_,
                          end_row - start_row, hidden_size, B1_dequant + start_row * hidden_size, tp);
        });
      } else {
        DequantizeBlock(fc1_weights_data + expert_idx * fc1_out_features * fc1_packed_cols,
                        fc1_scales_ptr,
                        fc1_zp_ptr,
                        is_fc1_block_wise ? block_size_ : 0, expert_weight_bits_,
                        fc1_out_features, hidden_size, B1_dequant, tp);
      }

      MlasGemm(CblasNoTrans, CblasTrans,
               m, n, k,
               1.0f, A1, k,
               B1_dequant, k,
               0.0f, C1, n,
               tp, &mlas_backend_kernel_selector_config_);

    fc1_bias_handling:

      if (has_fc1_bias) {
        const T* B1_bias = fc1_bias_data + expert_idx * fc1_out_features;
        if constexpr (std::is_same_v<T, MLFloat16>) {
          MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B1_bias), thread_bias1_buffer, static_cast<size_t>(fc1_out_features));
        } else {
          if (ShouldUseMemcpy(fc1_out_features)) {
            std::memcpy(thread_bias1_buffer, B1_bias, static_cast<size_t>(fc1_out_features) * sizeof(float));
          } else {
            const size_t unroll_factor = static_cast<size_t>(GetUnrollFactor(fc1_out_features));
            size_t j = 0;
            for (; j + unroll_factor <= static_cast<size_t>(fc1_out_features); j += unroll_factor) {
              for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
                thread_bias1_buffer[j + loop_k] = static_cast<float>(B1_bias[j + loop_k]);
              }
            }
            for (; j < static_cast<size_t>(fc1_out_features); ++j) {
              thread_bias1_buffer[j] = static_cast<float>(B1_bias[j]);
            }
          }
        }

        for (int64_t i = 0; i < num_expert_tokens; ++i) {
          float* C1_row = C1 + i * fc1_out_features;
          const size_t unroll_factor = static_cast<size_t>(GetUnrollFactor(fc1_out_features));

          size_t j = 0;
          for (; j + unroll_factor <= static_cast<size_t>(fc1_out_features); j += unroll_factor) {
            for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
              C1_row[j + loop_k] += thread_bias1_buffer[j + loop_k];
            }
          }
          for (; j < static_cast<size_t>(fc1_out_features); ++j) {
            C1_row[j] += thread_bias1_buffer[j];
          }
        }
      }

    fc1_gemm_done:

      if (activation_type_ == ActivationType::SwiGLU) {
        const int64_t activation_threshold = std::max(int64_t{4}, 256 / std::max(int64_t{1}, inter_size));
        if (num_expert_tokens >= activation_threshold && tp != nullptr) {
          const int64_t activation_block_size = std::max(int64_t{1}, std::min(int64_t{64}, activation_threshold));
          const int64_t num_activation_blocks = (num_expert_tokens + activation_block_size - 1) / activation_block_size;

          if (num_activation_blocks > 1) {
            concurrency::ThreadPool::TrySimpleParallelFor(tp, narrow<int>(num_activation_blocks), [&](std::ptrdiff_t block_idx) {
              const int64_t start_token = block_idx * activation_block_size;
              const int64_t end_token = std::min(start_token + activation_block_size, num_expert_tokens);

              for (int64_t i = start_token; i < end_token; ++i) {
                const float* C1_token = C1 + i * fc1_out_features;
                float* A2_token = A2 + i * inter_size;
                ApplySwiGLUActivation(C1_token, A2_token, inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
              }
            });
          } else {
            for (int64_t i = 0; i < num_expert_tokens; ++i) {
              const float* C1_token = C1 + i * fc1_out_features;
              float* A2_token = A2 + i * inter_size;
              ApplySwiGLUActivation(C1_token, A2_token, inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
            }
          }
        } else {
          for (int64_t i = 0; i < num_expert_tokens; ++i) {
            const float* C1_token = C1 + i * fc1_out_features;
            float* A2_token = A2 + i * inter_size;
            ApplySwiGLUActivation(C1_token, A2_token, inter_size, true, activation_alpha_, activation_beta_, swiglu_limit_);
          }
        }
      } else {
        ApplyActivationVectorized(C1, num_expert_tokens * fc1_out_features);
        std::copy(C1, C1 + (num_expert_tokens * fc1_out_features), A2);
      }

      const T* fc2_scales_ptr;
      const uint8_t* fc2_zp_ptr;

      if (is_fc2_block_wise) {
        const int64_t fc2_blocks_per_row = fc2_scales_dims[2];
        fc2_scales_ptr = fc2_scales_data + expert_idx * hidden_size * fc2_blocks_per_row;
        fc2_zp_ptr = (fc2_zp_data == nullptr) ? nullptr : fc2_zp_data + expert_idx * fc2_zp_expert_stride;
      } else {
        fc2_scales_ptr = fc2_scales_data + expert_idx * hidden_size;
        fc2_zp_ptr = (fc2_zp_data == nullptr) ? nullptr : fc2_zp_data + expert_idx * fc2_zp_expert_stride;
      }

      const int64_t fc2_dequant_block_size = GetDequantBlockSize(hidden_size, num_expert_tokens);
      const int64_t num_fc2_dequant_blocks = (hidden_size + fc2_dequant_block_size - 1) / fc2_dequant_block_size;

      const size_t m2 = static_cast<size_t>(num_expert_tokens);
      const size_t n2 = static_cast<size_t>(hidden_size);
      const size_t k2 = static_cast<size_t>(inter_size);

      MLAS_BLK_QUANT_TYPE q_type2 = BlkQ4Sym;  // Initialize to default
      bool use_direct_q4_gemm_fc2 = use_mlas_q4_gemm_effective &&
                                    ((fc2_direct_q4_cache_ptr != nullptr) ||
                                     ((packed_fc2_ == nullptr) && (fc2_zp_data == nullptr) &&
                                      CanUseMlasQ4Gemm(expert_weight_bits_, is_fc2_block_wise ? block_size_ : 0,
                                                       hidden_size, inter_size, q_type2)));

      if (can_use_fc2_lut_gemm &&
          TryRunLutGemm(A2, C2, fc2_weights_data, fc2_direct_lut_cache_ptr,
                        fc2_scales_ptr, fc2_zp_ptr, expert_idx,
                        hidden_size, inter_size, fc2_packed_cols,
                        block_size_, fc2_scales_dims[2],
                        thread_lut_packed_buffer, thread_lut_scale_buffer,
                        num_expert_tokens, tp)) {
        goto fc2_gemm_done;
      }

      if (packed_fc2_ != nullptr) {
        if (use_mlas_q4_gemm_effective && fc2_zp_data == nullptr &&
            CanUseMlasQ4Gemm(expert_weight_bits_, is_fc2_block_wise ? block_size_ : 0,
                             hidden_size, inter_size, q_type2)) {
          if (fc2_direct_q4_cache_ptr != nullptr) {
            float* fc2_bias_float = nullptr;
            if (has_fc2_bias) {
              const T* B2_bias = fc2_bias_data + expert_idx * hidden_size;
              if constexpr (std::is_same_v<T, MLFloat16>) {
                MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B2_bias), thread_bias2_buffer, static_cast<size_t>(hidden_size));
              } else {
                std::memcpy(thread_bias2_buffer, B2_bias, static_cast<size_t>(hidden_size) * sizeof(float));
              }
              fc2_bias_float = thread_bias2_buffer;
            }

            size_t packed_size = MlasQ4GemmPackBSize(q_type2, static_cast<size_t>(hidden_size), static_cast<size_t>(inter_size));
            const uint8_t* packed_b = static_cast<const uint8_t*>(fc2_direct_q4_cache_ptr) + expert_idx * packed_size;

            Status gemm_status = DirectQ4Gemm(A2, packed_b, fc2_bias_float, C2,
                                              num_expert_tokens, hidden_size, inter_size, fc2_direct_qtype, tp);
            if (gemm_status.IsOK()) {
              fc2_bias_added_by_mlas = true;
              goto fc2_gemm_done;
            }
          }
        }

        // Dequantize from PrePacked (transposed, unpacked)
        const uint8_t* current_packed_ptr = static_cast<const uint8_t*>(packed_fc2_.get()) + expert_idx * hidden_size * inter_size;

        DequantizePrePacked(current_packed_ptr, fc2_scales_ptr, fc2_zp_ptr,
                            is_fc2_block_wise ? block_size_ : 0,
                            hidden_size, inter_size,
                            B2_dequant, fc2_scales_dims);

        // Fallback
        MlasGemm(CblasNoTrans, CblasNoTrans,
                 m2, n2, k2,
                 1.0f, A2, k2,
                 B2_dequant, n2,
                 0.0f, C2, n2,
                 tp, &mlas_backend_kernel_selector_config_);

        goto fc2_gemm_done;
      }

      if (use_direct_q4_gemm_fc2) {
        IAllocatorUniquePtr<uint8_t> mlas_packed_fc2;
        Status convert_status = ConvertToMlasQ4Format(
            fc2_weights_data + expert_idx * hidden_size * fc2_packed_cols,
            fc2_scales_ptr,
            fc2_zp_ptr,  // This will be nullptr
            is_fc2_block_wise ? block_size_ : 0,
            expert_weight_bits_,
            hidden_size,
            inter_size,
            q_type2,
            allocator,
            mlas_packed_fc2);

        if (convert_status.IsOK()) {
          float* fc2_bias_float = nullptr;

          if (has_fc2_bias) {
            const T* B2_bias = fc2_bias_data + expert_idx * hidden_size;
            fc2_bias_float = thread_bias2_buffer;

            if constexpr (std::is_same_v<T, MLFloat16>) {
              MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B2_bias), fc2_bias_float, static_cast<size_t>(hidden_size));
            } else {
              for (int64_t i = 0; i < hidden_size; ++i) {
                fc2_bias_float[i] = static_cast<float>(B2_bias[i]);
              }
            }
          }

          Status gemm_status = DirectQ4Gemm(A2, mlas_packed_fc2.get(), fc2_bias_float, C2,
                                            num_expert_tokens, hidden_size, inter_size, q_type2, tp);

          if (gemm_status.IsOK()) {
            fc2_bias_added_by_mlas = true;
            goto fc2_gemm_done;
          }
        }

        // If direct Q4 GEMM failed, fall back to traditional approach
      }

      // Traditional approach: dequantize + regular GEMM
      if (num_fc2_dequant_blocks > 1 && hidden_size >= 32) {
        concurrency::ThreadPool::TrySimpleParallelFor(tp, narrow<int>(num_fc2_dequant_blocks), [&](std::ptrdiff_t block_idx) {
          const int64_t start_row = block_idx * fc2_dequant_block_size;
          const int64_t end_row = std::min(start_row + fc2_dequant_block_size, hidden_size);
          const auto offset = expert_idx * hidden_size * fc2_packed_cols + start_row * fc2_packed_cols;

          const T* current_scales_ptr = fc2_scales_ptr + (is_fc2_block_wise ? start_row * fc2_scales_dims[2] : start_row);
          const uint8_t* current_zp_ptr = nullptr;
          if (fc2_zp_ptr != nullptr) {
            if (is_fc2_block_wise) {
              const int64_t fc2_blocks_per_row = (inter_size + block_size_ - 1) / block_size_;
              const int64_t fc2_zp_blocks_packed = (fc2_blocks_per_row + zp_pack_size - 1) / zp_pack_size;
              current_zp_ptr = fc2_zp_ptr + start_row * fc2_zp_blocks_packed;
            } else {
              current_zp_ptr = fc2_zp_ptr + start_row / zp_pack_size;
            }
          }

          DequantizeBlock(fc2_weights_data + offset,
                          current_scales_ptr,
                          current_zp_ptr,
                          is_fc2_block_wise ? block_size_ : 0, expert_weight_bits_,
                          end_row - start_row, inter_size, B2_dequant + start_row * inter_size, tp);
        });
      } else {
        DequantizeBlock(fc2_weights_data + expert_idx * hidden_size * fc2_packed_cols,
                        fc2_scales_ptr,
                        fc2_zp_ptr,
                        is_fc2_block_wise ? block_size_ : 0, expert_weight_bits_,
                        hidden_size, inter_size, B2_dequant, tp);
      }

      MlasGemm(CblasNoTrans, CblasTrans,
               m2, n2, k2,
               1.0f, A2, k2,
               B2_dequant, k2,
               0.0f, C2, n2,
               tp, &mlas_backend_kernel_selector_config_);

    fc2_gemm_done:

      if (has_fc2_bias && !fc2_bias_added_by_mlas) {
        const T* B2_bias = fc2_bias_data + expert_idx * hidden_size;
        if constexpr (std::is_same_v<T, MLFloat16>) {
          MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLFloat16*>(B2_bias), thread_bias2_buffer, static_cast<size_t>(hidden_size));
        } else {
          if (ShouldUseMemcpy(hidden_size)) {
            std::memcpy(thread_bias2_buffer, B2_bias, static_cast<size_t>(hidden_size) * sizeof(float));
          } else {
            const size_t unroll_factor = narrow<size_t>(GetUnrollFactor(hidden_size));
            size_t j = 0;
            for (; j + unroll_factor <= narrow<size_t>(hidden_size); j += unroll_factor) {
              for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
                thread_bias2_buffer[j + loop_k] = static_cast<float>(B2_bias[j + loop_k]);
              }
            }
            for (; j < narrow<size_t>(hidden_size); ++j) {
              thread_bias2_buffer[j] = static_cast<float>(B2_bias[j]);
            }
          }
        }
      }

      for (int64_t i = 0; i < num_expert_tokens; ++i) {
        const int64_t route_idx = routes[static_cast<size_t>(i)];
        const int64_t token_idx = route_idx / k_;
        const float weight = route_scale[route_idx];

        if (token_idx < 0 || token_idx >= num_tokens) continue;

        const size_t buffer_offset = static_cast<size_t>(token_idx) * static_cast<size_t>(hidden_size);
        if (buffer_offset + static_cast<size_t>(hidden_size) > output_buffer_size) continue;

        float* dest = thread_local_outputs + static_cast<size_t>(thread_id) * output_buffer_size + buffer_offset;
        const float* src = C2 + i * hidden_size;

        if (has_fc2_bias && !fc2_bias_added_by_mlas) {
          const size_t unroll_factor = narrow<size_t>(GetUnrollFactor(hidden_size));
          size_t j = 0;
          for (; j + unroll_factor <= narrow<size_t>(hidden_size); j += unroll_factor) {
            for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
              dest[j + loop_k] += weight * (src[j + loop_k] + thread_bias2_buffer[j + loop_k]);
            }
          }
          for (; j < narrow<size_t>(hidden_size); ++j) {
            dest[j] += weight * (src[j] + thread_bias2_buffer[j]);
          }
        } else {
          const size_t unroll_factor = narrow<size_t>(GetUnrollFactor(hidden_size));
          size_t j = 0;
          for (; j + unroll_factor <= narrow<size_t>(hidden_size); j += unroll_factor) {
            for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
              dest[j + loop_k] += weight * src[j + loop_k];
            }
          }
          for (; j < narrow<size_t>(hidden_size); ++j) {
            dest[j] += weight * src[j];
          }
        }
      }
    }
  });

  auto accumulate = [&](float* buffer) {
    std::memset(buffer, 0, output_buffer_size * sizeof(float));

    const int max_acc_threads = tp ? concurrency::ThreadPool::DegreeOfParallelism(tp) : 1;
    const size_t acc_thread_divisor = std::max(size_t{1}, static_cast<size_t>(max_acc_threads) * 8);
    const size_t min_elements_per_thread = std::max(size_t{32}, output_buffer_size / acc_thread_divisor);
    const int optimal_acc_threads = (tp == nullptr || output_buffer_size < min_elements_per_thread) ? 1 : std::min(narrow<int>(output_buffer_size / std::max(size_t{1}, min_elements_per_thread)), max_acc_threads);
    const int num_acc_threads = std::max(1, optimal_acc_threads);

    if (num_acc_threads > 1) {
      concurrency::ThreadPool::TrySimpleParallelFor(tp, num_acc_threads, [&](std::ptrdiff_t acc_thread_id) {
        const size_t elements_per_thread = output_buffer_size / static_cast<size_t>(num_acc_threads);
        const size_t start_idx = static_cast<size_t>(acc_thread_id) * elements_per_thread;
        const size_t end_idx = (acc_thread_id == num_acc_threads - 1) ? output_buffer_size : start_idx + elements_per_thread;

        for (int i = 0; i < num_expert_threads; ++i) {
          const size_t thread_offset = static_cast<size_t>(i) * output_buffer_size;
          const float* src = thread_local_outputs + thread_offset + start_idx;
          float* dst = buffer + start_idx;

          size_t j = 0;
          const size_t chunk_size = end_idx - start_idx;
          const size_t unroll_factor = static_cast<size_t>(GetUnrollFactor(static_cast<int64_t>(chunk_size)));
          for (; j + unroll_factor <= chunk_size; j += unroll_factor) {
            for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
              dst[j + loop_k] += src[j + loop_k];
            }
          }
          for (; j < chunk_size; ++j) {
            dst[j] += src[j];
          }
        }
      });
    } else {
      for (int i = 0; i < num_expert_threads; ++i) {
        const size_t thread_offset = static_cast<size_t>(i) * output_buffer_size;
        const float* src = thread_local_outputs + thread_offset;

        size_t j = 0;
        const size_t unroll_factor = narrow<size_t>(GetUnrollFactor(narrow<int64_t>(output_buffer_size)));
        for (; j + unroll_factor <= output_buffer_size; j += unroll_factor) {
          for (size_t loop_k = 0; loop_k < unroll_factor; ++loop_k) {
            buffer[j + loop_k] += src[j + loop_k];
          }
        }
        for (; j < output_buffer_size; ++j) {
          buffer[j] += src[j];
        }
      }
    }
  };

  if constexpr (std::is_same_v<T, MLFloat16>) {
    auto final_output_float_ptr = IAllocator::MakeUniquePtr<float>(allocator, output_buffer_size);
    float* final_output_float = final_output_float_ptr.get();
    accumulate(final_output_float);

    MlasConvertFloatToHalfBuffer(final_output_float,
                                 reinterpret_cast<MLFloat16*>(output->template MutableData<T>()),
                                 static_cast<size_t>(output_buffer_size));
  } else {
    accumulate(output->template MutableData<T>());
  }

  return Status::OK();
}

template <typename T>
void QMoECPU<T>::ApplyActivationVectorized(float* data, int64_t size) const {
  for (int64_t i = 0; i < size; ++i) {
    data[i] = ApplyActivation(data[i], activation_type_);
  }
}

template QMoECPU<float>::QMoECPU(const OpKernelInfo& op_kernel_info);

template Status QMoECPU<float>::Compute(OpKernelContext* context) const;
template Status QMoECPU<float>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc, bool& is_packed, PrePackedWeights* prepacked_weights);
template Status QMoECPU<float>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, gsl::span<const size_t> prepacked_buffer_sizes, int input_idx, bool& used_shared_buffers);
template QMoECPU<MLFloat16>::QMoECPU(const OpKernelInfo& op_kernel_info);
template Status QMoECPU<MLFloat16>::Compute(OpKernelContext* context) const;
template Status QMoECPU<MLFloat16>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc, bool& is_packed, PrePackedWeights* prepacked_weights);
template Status QMoECPU<MLFloat16>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, gsl::span<const size_t> prepacked_buffer_sizes, int input_idx, bool& used_shared_buffers);

// Kernel Registration
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE, kMSDomain, 1, float, kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    QMoECPU<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE, kMSDomain, 1, MLFloat16, kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<MLFloat16>()),
    QMoECPU<MLFloat16>);

}  // namespace contrib
}  // namespace onnxruntime
