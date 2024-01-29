// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define DequantizeLinearBlockWise operator, it is basically
// dequantize input tensor and unpack it into float/half tensor.
//

#include <sys/types.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {
namespace GPTQPacking {
static std::unique_ptr<int32_t> l_g_idx = std::make_unique<int32_t>(4096*128);

template <typename ZERO_TYPE>
void GeneralDequant(concurrency::ThreadPool* pool, const int32_t* qweight_i32, const float* scale,
                    const ZERO_TYPE* qzeros_i32, const int32_t* g_idx,
                    float* output, uint32_t mat_k, uint32_t mat_n, int bits, int group_size) {
  const int32_t* group_idx = g_idx;
  if (g_idx == nullptr) {
    int32_t* sg_idx = l_g_idx.get();
    for (uint32_t i = 0; i < mat_k; i++) {
      sg_idx[i] = i / group_size;
    }
    group_idx = sg_idx;
  }
  const uint32_t max_num_in_bits = (1 << bits) - 1;
  const uint32_t qzeros_size_n = (mat_n * bits + 31) / 32;
  constexpr int32_t kThreadBlockSize = 64;
  concurrency::ThreadPool::TrySimpleParallelFor(
      pool, static_cast<std::ptrdiff_t>(mat_k / kThreadBlockSize),
      [&](std::ptrdiff_t row_block) {
        row_block*=kThreadBlockSize;
        //for (uint32_t row_block = 0; row_block < mat_k; row_block+=kThreadBlockSize) {
        for (uint32_t sub_row = 0; sub_row < mat_k; sub_row += 1) {
          uint32_t row = row_block + sub_row;
          for (uint32_t col = 0; col < mat_n; col++) {
            uint32_t reorder_idx = group_idx[row];

            uint32_t weight_int = 0;
            uint8_t wv1 = 0;

            int start_bits = row * bits;
            int first = start_bits / 32;
            int end_bits = (start_bits + bits);
            int second = end_bits / 32;
            start_bits = start_bits % 32;
            end_bits = end_bits % 32;

            weight_int = qweight_i32[first * mat_n + col];
            wv1 = (weight_int >> start_bits) & max_num_in_bits;
            if (first != second) {
              weight_int = qweight_i32[second * mat_n + col];
              wv1 |= (weight_int & ((1 << end_bits) - 1)) << (32 - start_bits);
            }

            float f_zeros = 8;
            if constexpr (std::is_same_v<ZERO_TYPE, float>) {
              f_zeros = qzeros_i32[reorder_idx * mat_n + col];
            }else{
              uint32_t zero_v1 = 0x88888888;
              uint8_t zv1 = 8;
              if (qzeros_i32 != nullptr) {
                int start_bits = col * bits;
                int first = start_bits / 32;
                int end_bits = (start_bits + bits);
                int second = end_bits / 32;
                start_bits = start_bits % 32;
                end_bits = end_bits % 32;

                zero_v1 = qzeros_i32[reorder_idx * qzeros_size_n + first];
                zv1 = (zero_v1 >> start_bits) & max_num_in_bits;
                if (first != second) {
                  zero_v1 = qzeros_i32[reorder_idx * qzeros_size_n + second];
                  zv1 |= (zero_v1 & ((1 << end_bits) - 1)) << (32 - start_bits);
                }
                f_zeros = float(zv1);
              }
            }

            float out_real =(float(wv1)-f_zeros) * scale[reorder_idx * mat_n + col];
            if (fabs(output[row * mat_n + col] - out_real) > 0.001) {
              printf("error %f %f\n", output[row * mat_n + col], out_real);
            }
          }
        }
        //}
      });
}

template void GeneralDequant<float>(concurrency::ThreadPool* pool, const int32_t* qweight_i32, const float* scale,
                                    const float* qzeros_i32, const int32_t* g_idx,
                                    float* output, uint32_t mat_k, uint32_t mat_n, int bits, int group_size);
template void GeneralDequant<uint32_t>(concurrency::ThreadPool* pool, const int32_t* qweight_i32, const float* scale,
                                       const uint32_t* qzeros_i32, const int32_t* g_idx,
                                       float* output, uint32_t mat_k, uint32_t mat_n, int bits, int group_size);
template <typename SCALE_TYPE>
void DequantWeightNbitGidx(concurrency::ThreadPool* pool,
                           const int32_t* qweight_i32, const SCALE_TYPE* scale,
                           const uint32_t* qzeros_i32, const int32_t* g_dix,
                           SCALE_TYPE* output,
                           uint32_t mat_k, uint32_t mat_n, int) {
  //assert(bits == 4);
  constexpr uint32_t kWBITS = 4;
  constexpr uint32_t kOUTCOLBLOCK = 64;
  const uint32_t kCompressedSize = 32 / 4;
  constexpr uint32_t kOUTCOLBLOCK_PER_GROUP = kOUTCOLBLOCK / kCompressedSize;
  constexpr uint32_t kThreadBlockSize = 64;
  const int qzeros_size_n = (mat_n + kCompressedSize - 1) / kCompressedSize;
  concurrency::ThreadPool::TrySimpleParallelFor(
      pool, static_cast<std::ptrdiff_t>(mat_k / kThreadBlockSize),
      [&](std::ptrdiff_t row_block) {
        //for (uint32_t row_block = 0; row_block < mat_k; row_block += kThreadBlockSize) {
          for (uint32_t sub_row = 0; sub_row < kThreadBlockSize; sub_row += kCompressedSize) {
            uint32_t zeros_u32[64];
            for (uint32_t col_block = 0; col_block < mat_n; col_block += kOUTCOLBLOCK) {
              for (uint32_t i = 0; i < kOUTCOLBLOCK_PER_GROUP; i++) {
                std::copy_n(qzeros_i32 + g_dix[row_block + sub_row + i] * qzeros_size_n + col_block / kCompressedSize,
                            kOUTCOLBLOCK_PER_GROUP, zeros_u32 + i * kOUTCOLBLOCK_PER_GROUP);
              }
              for (uint32_t sub_col_block = 0; sub_col_block < kOUTCOLBLOCK_PER_GROUP; sub_col_block++) {
                uint32_t qweight_g8[kCompressedSize];
                std::copy_n(qweight_i32 + (row_block + sub_row) / kCompressedSize * mat_n +
                            col_block + sub_col_block * kOUTCOLBLOCK_PER_GROUP, kCompressedSize, qweight_g8);
                for (uint32_t row_sub_idx = 0; row_sub_idx < kCompressedSize; row_sub_idx++) {
                  int reorder_idx = g_dix[row_block + sub_row + row_sub_idx];
                  for (uint32_t sub_idx = 0; sub_idx < kOUTCOLBLOCK_PER_GROUP; sub_idx++) {
                    int col_idx = col_block + sub_idx + sub_col_block * kOUTCOLBLOCK_PER_GROUP;
                    auto qweight = int32_t((qweight_g8[sub_idx] >> (row_sub_idx * kWBITS)) & 0xf);
                    int32_t qzeros = int32_t(0xf & (zeros_u32[sub_idx + row_sub_idx * kOUTCOLBLOCK_PER_GROUP] >> (row_sub_idx * kWBITS)));
                    output[(row_block + sub_row + row_sub_idx) * mat_n + col_idx] = float(qweight - qzeros) * scale[reorder_idx * mat_n + col_idx];
                  }
                }
              }
            }
          }
        //}
      });
}

template void
DequantWeightNbitGidx<float>(concurrency::ThreadPool* pool,
                             const int32_t* qweight_i32, const float* scale,
                             const uint32_t* qzeros_i32, const int32_t* g_dix,
                             float* output,
                             uint32_t mat_k, uint32_t mat_n, int bits);


template <typename SCALE_TYPE, typename ZEROT>
void DequantWeightNbit(
    concurrency::ThreadPool* pool,
    const int32_t* qweight_i32,
    const SCALE_TYPE* scale,
    const ZEROT* qzeros_i32,
    SCALE_TYPE* output,
    uint32_t mat_k,
    uint32_t mat_n,
    uint32_t,
    uint32_t groupsize) {
  //assert(bits == 4);
  constexpr uint32_t kWBITS = 4;
  constexpr uint32_t kOUTCOLBLOCK = 64;
  const uint32_t kCompressedSize = 32 / 4;
  const int qzeros_size_n = (mat_n + kCompressedSize - 1) / kCompressedSize;
  concurrency::ThreadPool::TrySimpleParallelFor(
      pool, static_cast<std::ptrdiff_t>(mat_k/groupsize),
      [&](std::ptrdiff_t row_block) {
        row_block *= groupsize;
        //for (uint32_t row_block = 0; row_block < mat_k; row_block += groupsize) {
        for (uint32_t col_block = 0; col_block < mat_n; col_block += kOUTCOLBLOCK) {
          for (uint32_t sub_col_block = 0; sub_col_block < kOUTCOLBLOCK; sub_col_block += kCompressedSize) {
            for (uint32_t inner_group_idx = 0; inner_group_idx < groupsize; inner_group_idx += kCompressedSize) {
              uint32_t qweight_g8[kCompressedSize];
              std::copy_n(qweight_i32 + (row_block + inner_group_idx) / kCompressedSize * mat_n + col_block + sub_col_block, kCompressedSize, qweight_g8);
              uint32_t zeros_u32 = qzeros_i32[row_block / groupsize * qzeros_size_n + (col_block + sub_col_block) / kCompressedSize];
              uint8_t zeros_u8_8[kCompressedSize];
              for (uint32_t i = 0; i < kCompressedSize; i++) {
                zeros_u8_8[i] = (zeros_u32 >> (i * kWBITS)) & 0xf;
              }
              for (uint32_t row_sub_idx = 0; row_sub_idx < kCompressedSize; row_sub_idx++) {
                const SCALE_TYPE* scale_p = &scale[row_block / groupsize * mat_n + col_block + sub_col_block];
                for (uint32_t sub_col_idx = 0; sub_col_idx < kCompressedSize; sub_col_idx++) {
                  int col_idx = col_block + sub_col_block + sub_col_idx;
                  auto qweight = int32_t((qweight_g8[sub_col_idx] >> (row_sub_idx * kWBITS)) & 0xf);
                  output[(row_block + inner_group_idx + row_sub_idx) * mat_n + col_idx] = float(qweight - int32_t(zeros_u8_8[sub_col_idx])) * (*scale_p++);
                }
              }
            }
          }
        }
        //}
      });
}



template void DequantWeightNbit<float, float>(
    concurrency::ThreadPool* pool,
    const int32_t* qweight_i32,
    const float* scale,
    const float* qzeros_i32,
    float* output,
    uint32_t mat_k,
    uint32_t mat_n,
    uint32_t bits,
    uint32_t groupsize);

template void DequantWeightNbit<float, uint32_t>(
    concurrency::ThreadPool* pool,
    const int32_t* qweight_i32,
    const float* scale,
    const uint32_t* qzeros_i32,
    float* output,
    uint32_t mat_k,
    uint32_t mat_n,
    uint32_t bits,
    uint32_t groupsize);
}  // namespace GPTQPacking
}  // namespace contrib
}  // namespace onnxruntime
