/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    blkq4_fp16_gemm_sm80.h
 *
 * Abstract:
 *   Bridge between gtest code and gemm kernel implementation.
 *   Gemm kernel requires CUTLASS header files, which causes strange
 *   compilation errors with RE2 header files, which are required
 *   by gtest.
 */

#pragma once

#include <random>

#include "core/util/matrix_layout.h"
#include "core/common/common.h"
#include "core/mickey/blk_q4/f16_prepack_sm80.h"
#include "test/cuda_host/blkq4_fp16_quant_sm80.h"

namespace onnxruntime {
namespace cuda {
namespace test {

Status sm80_supported();

/**
 * @brief Generate a set of quantized weights, scales and offsets
 *        and dequantized weights for testing quantization and
 *        dequantization. All outputs are column major layout.
 *
 * @tparam ElementT The type of the dequantized weights.
 * @tparam block_size The block size of the quantization.
 * @tparam col_blocking Whether to use column blocking (all elements of
 *                      a block comes from a single column) or row blocking
 * @tparam has_offsets Whether to generate offsets.
 *
 * @param[in]  rows The number of rows of the weight matrix.
 * @param[in]  columns The number of columns of the weight matrix.
 * @param[out] dequants The dequantized weights, column major layout.
 * @param[out] q_weights The quantized weights, column major layout.
 * @param[out] q_scales The scales, column major layout.
 * @param[out] q_zp The zero points, column major layout.
 */
template <typename ElementT, int block_size, bool col_blocking, bool has_offsets>
inline void blkq4_weights_gen(
    int rows, int columns,
    std::vector<ElementT>& dequants,
    std::vector<uint8_t>& q_weights,
    std::vector<ElementT>& q_scales,
    std::vector<uint8_t>& q_zp) {
  using Base = onnxruntime::cuda::BlockwiseQuantization<
      ElementT,
      block_size,
      4,
      col_blocking>;

  using QuantBlocking = typename Base::QuantBlocking;
  using ElementW = typename Base::ElementW;
  using LayoutWPack = typename Base::LayoutWPack;
  using ElementQOffset = typename Base::ElementQOffset;

  static_assert(std::is_same<ElementW, uint8_t>::value);
  static_assert(std::is_same<ElementQOffset, uint8_t>::value);
  static_assert(std::is_same<LayoutWPack, ColumnMajorLayout>::value);

  unsigned int seed = 28571;  // Replace with desired seed value
  std::seed_seq seq{seed};
  std::mt19937 gen(seq);
  std::uniform_int_distribution<uint32_t> dis(0, 8192);

  const auto q_weight_shape = Base::get_quant_weights_shape(rows, columns);
  const auto meta_shape = Base::get_quant_meta_shape(rows, columns);
  const auto zp_shape = make_Position((meta_shape[0] + 1) / 2, meta_shape[1]);

  //
  // For testing quantization and dequantization, it is not straight
  // forward to avoid flaky tests due to rounding errors. The way we
  // try to achieve this is to:
  // 1. Generate a set of quantized weights, scales and offsets
  // 2. Dequantize the weights
  // 3. Quantize the dequantized weights
  // 4. Compare the dequantied-and-then-quantized weights with
  //    the original quantized weights
  //
  // Random filling of the initial values are key to get this right.
  // For weights, we must ensure each block gets a full range of
  // values, i.e. must contain 0 and 15. And for scales, they must
  // all be positive.
  //

  q_weights.resize(q_weight_shape.product());
  MatrixRef<ElementW, ColumnMajorLayout, true> tensor_q_weight(
      q_weights, make_Position(rows / 2, columns));
  int v = 7;
  for (int c = 0; c < tensor_q_weight.shape()[1]; c++) {
    for (int r = 0; r < tensor_q_weight.shape()[0]; ++r) {
      uint8_t v0 = static_cast<uint8_t>(v);
      v = (v + 5) % 16;
      if (v == 11 || v == 7 || v == 3) {
        // making the cycle 13 instead of 16, avoiding same values in a row
        v = (v + 5) % 16;
      }
      uint8_t v1 = 0;
      if (r + 1 < rows) {
        v1 = static_cast<uint8_t>(v);
        v = (v + 5) % 16;
        if (v == 11 || v == 7 || v == 3) {
          // making the cycle 13 instead of 16, avoiding same values in a row
          v = (v + 5) % 16;
        }
      }

      tensor_q_weight.at(r, c) = ElementW((v1 << 4) | v0);
    }
  }

  q_scales.resize(meta_shape.product());
  for (size_t i = 0; i < q_scales.size(); i++) {
    uint32_t v = dis(gen);
    uint32_t m = (v % 63) + 1;
    uint32_t e = (v >> 6) % 4;
    q_scales[i] = ElementT(m / static_cast<float>(1 << (2 + e)));
  }
  MatrixRef<ElementT, ColumnMajorLayout, true> tensor_scale(
      q_scales, meta_shape);

  MatrixRef<ElementQOffset, ColumnMajorLayout, true> tensor_offset;
  if constexpr (has_offsets) {
    q_zp.resize(zp_shape.product());
    tensor_offset = MatrixRef<ElementQOffset, ColumnMajorLayout, true>(
        q_zp, zp_shape);
    for (int c = 0; c < zp_shape[1]; c++) {
      for (int r = 0; r < zp_shape[0]; ++r) {
        uint8_t v0 = dis(gen) % 16;
        uint8_t v1 = 8;
        if (r * 2 + 1 < meta_shape[0]) {
          v1 = dis(gen) % 16;
        }
        tensor_offset.at(r, c) = static_cast<uint8_t>(v0 | (v1 << 4));
      }
    }
  }

  dequants.resize(rows * columns);
  MatrixRef<ElementT, ColumnMajorLayout> tensor_dequant(dequants, make_Position(rows, columns));

  // Dequantize weights and save into matrix B
  for (int col = 0; col < tensor_dequant.shape()[1]; ++col) {
    for (int row = 0; row < tensor_dequant.shape()[0]; ++row) {
      auto weight_cord = make_Position(row / 2, col);
      auto scale_cord = make_Position(row / QuantBlocking::kRow, col / QuantBlocking::kColumn);
      uint8_t offset = 8;
      if constexpr (has_offsets) {
        if (scale_cord[0] % 2 == 0) {
          offset = tensor_offset.at(scale_cord[0] / 2, scale_cord[1]) & 0x0f;
        } else {
          offset = tensor_offset.at(scale_cord[0] / 2, scale_cord[1]) >> 4;
        }
      }
      int w = 0;
      if (row % 2 == 0) {
        w = int(tensor_q_weight.at(weight_cord) & 0x0f);
      } else {
        w = int(tensor_q_weight.at(weight_cord) >> 4);
      }
      float scale = float(tensor_scale.at(scale_cord));
      float dequant = scale * float(w - offset);
      tensor_dequant.at(row, col) = ElementT(dequant);
      // Prints for help debugging in case of test failure
      // fprintf(stderr, "(%2d,%2d)= %2d, %2d, %f, %f\n", row, col, w, offset, scale, dequant);
    }
  }
}

template <
    int block_size,
    bool column_wise_blocking,
    bool small_m,
    bool has_offsets>
void run_blkq4_gemm(int m, int n, int k);

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
