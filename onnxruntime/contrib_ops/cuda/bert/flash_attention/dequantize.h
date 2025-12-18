#pragma once

#include "contrib_ops/cuda/bert/flash_attention/utils.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

// Set to 1 to enable debug prints for this kernel
#define DEQUANT_DEBUG 0

namespace onnxruntime {
namespace flash {

using namespace cute;

// ============================================================================
// KV Cache Dequantization Utilities for Flash Attention
// ============================================================================
//
// This file implements on-the-fly dequantization of quantized KV cache for
// Flash Attention kernels. It supports INT4 and INT8 quantization with both
// per-tensor and per-channel quantization modes.
//
// QUANTIZATION SCHEME:
// -------------------
// INT4: Symmetric signed quantization, range [-8, 7]
//   - Quantization:   q = clamp(round(x / scale), -8, 7)
//   - Storage:        packed = (q + 8) & 0x0F  [converts to unsigned 0-15]
//   - Packing:        2 values per byte (nibbles)
//   - Dequantization: x = ((packed & 0x0F) - 8) * scale
//
// INT8: Symmetric signed quantization, range [-128, 127]
//   - Quantization:   q = clamp(round(x / scale), -128, 127)
//   - Dequantization: x = q * scale
//
// SCALE TENSOR FORMAT:
// -------------------
// Scale tensors are always FP16/BF16 (type T), even when cache is INT4/INT8.
//
// PER_TENSOR mode (QUANT_TYPE=1):
//   - Single scale for entire tensor
//   - Shape: [1] or broadcastable to cache shape
//   - Access: scale[0]
//
// PER_CHANNEL mode (QUANT_TYPE=2):
//   - One scale per head element (channel)
//   - Shape: [num_heads_k, head_size] or broadcastable
//   - Access: scale[h_k_idx * head_size + coord_k]
//   - Where h_k_idx = head index, coord_k = element index within head
//
// BIT PACKING (INT4 only):
// -----------------------
// Each byte stores 2 values:
//   - Low nibble (bits 0-3):  even-indexed element (coord_k % 2 == 0)
//   - High nibble (bits 4-7): odd-indexed element (coord_k % 2 == 1)
//
// For odd head_size, the last nibble is padded with 0 (which represents -8+8=0 after bias).
//
// TEMPLATE PARAMETERS:
// -------------------
// QUANT_TYPE: 0 = none, 1 = PER_TENSOR, 2 = PER_CHANNEL
// BIT_WIDTH:  4 or 8
// ScaleType:  FP16 (half) or BF16 (bfloat16)
// ============================================================================

template <
    bool Is_even_MN = true, bool Is_even_K = true, int QUANT_TYPE = 0, int BIT_WIDTH = 0,
    typename TiledCopy, typename SrcTensor, typename DstTensor,
    typename CoordTensor, typename PredTensor, typename ScaleType>
__noinline__ __device__ void copy_and_dequantize(
    TiledCopy const& tiled_copy,
    SrcTensor const& gmem_src,       // Thread-local view of quantized source tensor in gmem
    DstTensor& smem_dst,             // Thread-local view of dequantized destination tensor in smem
    CoordTensor const& identity_MN,  // Thread-local view of an identity tensor for global coordinates
    PredTensor const& predicate_K,   // Predicate for the K dimension
    const int max_MN,
    const ScaleType* scale,
    const int d,
    const int h_k_idx) {
#if DEQUANT_DEBUG
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("[DEQUANT_DEBUG] Enter copy_and_dequantize\n");
    printf("  gmem_ptr: %p, smem_ptr: %p, scale_ptr: %p\n", raw_pointer_cast(gmem_src.data()), raw_pointer_cast(smem_dst.data()), scale);
    printf("  d: %d, h_k_idx: %d, max_MN: %d, BIT_WIDTH: %d, QUANT_TYPE: %d\n", d, h_k_idx, max_MN, BIT_WIDTH, QUANT_TYPE);
  }
#endif

  using DQuantType = typename DstTensor::value_type;  // The dequantized type (e.g., half)
  using QType = typename SrcTensor::value_type;       // The quantized type (e.g., int8_t or uint8_t for int4)

  constexpr int R = SrcTensor::layout_type::rank;
  constexpr int K_Tiles = []() {
    if constexpr (R >= 3) {
      using ShapeType = decltype(std::declval<typename SrcTensor::layout_type>().shape());
      return size<2>(ShapeType{});
    } else {
      return 1;
    }
  }();

#pragma unroll
  for (int m = 0; m < size<1>(gmem_src); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < K_Tiles; ++k) {
        if (Is_even_K || predicate_K(k)) {
          // Define a lambda to handle the core dequantization logic for a given slice
          auto process_slice = [&](auto gmem_slice, auto smem_slice, auto identity_slice) {
            // Step 1 & 2: Load raw quantized data from GMEM to registers.
            Tensor tRrQ_raw = make_tensor<QType>(shape(gmem_slice));

            if constexpr (Is_even_MN) {
              copy(tiled_copy, gmem_slice, tRrQ_raw);
            } else {
              if constexpr (BIT_WIDTH == 4) {
                fill(tRrQ_raw, static_cast<QType>(0x88));
              } else {
                clear(tRrQ_raw);
              }
              // Create a predicate tensor for valid MN coordinates
              auto tPred = make_tensor<bool>(shape(gmem_slice));
#pragma unroll
              for (int i = 0; i < size(tPred); ++i) {
                tPred(i) = get<1>(identity_slice(i)) < max_MN;
              }
              copy_if(tiled_copy, tPred, gmem_slice, tRrQ_raw);
            }

            // Step 3: Dequantize the data now residing in registers.
            Tensor tRsK = make_tensor<DQuantType>(shape(smem_slice));

            if constexpr (BIT_WIDTH == 8) {
              auto const* tRrQ_quant = reinterpret_cast<int8_t const*>(&tRrQ_raw(0));

#pragma unroll 1
              for (int i = 0; i < size(tRsK); ++i) {
                float val = static_cast<float>(tRrQ_quant[i]);
                float current_scale = 1.0f;
                int coord_k = get<1>(identity_slice(i));

                if constexpr (QUANT_TYPE == 1) {  // PER_TENSOR
                  current_scale = static_cast<float>(scale[0]);
                } else if constexpr (QUANT_TYPE == 2) {  // PER_CHANNEL
                  if (coord_k < d) {
                    int scale_idx = h_k_idx * d + coord_k;
#if DEQUANT_DEBUG
                    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && m == 0 && k == 0 && i == 0) {
                      printf("[FlashDequant-8bit] Thread=%d CoordK=%d ScaleIdx=%d Raw=%f Scale=%f Result=%f\n",
                             threadIdx.x, coord_k, scale_idx, val, current_scale, val * current_scale);
                    }
#endif
                    current_scale = static_cast<float>(scale[scale_idx]);
                  }
                }
                tRsK(i) = static_cast<DQuantType>(val * current_scale);
              }
            } else if constexpr (BIT_WIDTH == 4) {
              auto const* tRrQ_packed = reinterpret_cast<uint8_t const*>(&tRrQ_raw(0));

#pragma unroll 1
              for (int i = 0; i < size(tRsK); ++i) {
                // With Duplicate Layout in kernel, we load bytes as [B0, B0, B1, B1...].
                // So for i=0, we want B0. index 0.
                // For i=1, we want B0. index 1.
                // For i=2, we want B1. index 2.
                int coord_k = get<1>(identity_slice(i));
#if DEQUANT_DEBUG
                if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && i == 0) {
                  printf("[DequantDebug] BIT_WIDTH=%d QUANT_TYPE=%d i=%d coord_k=%d d=%d packed=0x%x\n",
                         BIT_WIDTH, QUANT_TYPE, i, coord_k, d, (int)tRrQ_packed[i]);
                }
#endif
                uint8_t packed_val = tRrQ_packed[i];
                // Unpack INT4 values from nibbles and remove +8 bias applied during quantization.
                // Even elements are in low nibble (bits 0-3), odd elements in high nibble (bits 4-7).
                // The -8 bias restores the original signed range [-8, 7].
                int8_t unpacked_val = (coord_k % 2 == 0)
                                          ? static_cast<int8_t>((packed_val & 0x0F) - 8)
                                          : static_cast<int8_t>((packed_val >> 4) - 8);

                float val = static_cast<float>(unpacked_val);
                float current_scale = 1.0f;

                if constexpr (QUANT_TYPE == 1) {  // PER_TENSOR
                  current_scale = static_cast<float>(scale[0]);
                } else if constexpr (QUANT_TYPE == 2) {  // PER_CHANNEL
                  if (coord_k < d) {
                    int scale_idx = h_k_idx * d + coord_k;
                    current_scale = static_cast<float>(scale[scale_idx]);
#if DEQUANT_DEBUG
                    if (threadIdx.x == 0 && i < 4) {
                      printf("[FlashDequant-4bit] B=(%d,%d,%d) T=%d i=%d CK=%d SIdx=%d Pkd=0x%x Unp=%d Sc=%f Res=%f\n",
                             blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, i, coord_k, scale_idx, packed_val, unpacked_val, current_scale, val * current_scale);
                    }
#endif
                  }
                }
                tRsK(i) = static_cast<DQuantType>(val * current_scale);
              }
            }
            // Step 4: Perform an efficient copy from registers to shared memory.
            copy(tRsK, smem_slice);
          };

          if constexpr (R >= 3) {
            process_slice(gmem_src(_, m, k), smem_dst(_, m, k), identity_MN(_, m, k));
          } else {
            process_slice(gmem_src(_, m), smem_dst(_, m), identity_MN(_, m));
          }

        } else {
          if constexpr (R >= 3) {
            clear(smem_dst(_, m, k));
          } else {
            clear(smem_dst(_, m));
          }
        }
      }
    } else {
      if constexpr (R >= 3) {
        clear(smem_dst(_, m, _));
      } else {
        clear(smem_dst(_, m));
      }
    }
  }
}

}  // namespace flash
}  // namespace onnxruntime
