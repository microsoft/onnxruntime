/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    blkq4_fp16_quant_sm80.h
 *
 * Abstract:
 *   Oracle computation for blockwise 4b quantization for fp16
 *   gemm kernel specifically for Ampere GPUs. This is used for
 *   testing the cuda kernel implementation in
 *   (test/providers/cuda/test_cases)
 *   and for testing the cuda op prepack code in (test/optimizer)
 */

#pragma once

#include <random>
#include "core/mickey/blk_q4/f16_prepack_sm80.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace test {

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
    uint32_t vscale = dis(gen);
    uint32_t m = (vscale % 63) + 1;
    uint32_t e = (vscale >> 6) % 4;
    q_scales[i] = ElementT(m / static_cast<float>(1 << (2 + e)));
  }
  MatrixRef<ElementT, ColumnMajorLayout, true> tensor_scale(
      q_scales, meta_shape);

  MatrixRef<ElementQOffset, ColumnMajorLayout, true> tensor_offset;
  if constexpr (has_offsets) {
    const auto zp_shape = make_Position((meta_shape[0] + 1) / 2, meta_shape[1]);
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
      // fprintf(stderr, "%f~%d-%d|%f, ", dequant, w, offset, scale);
    }
    // fprintf(stderr, "\n");
  }
}

static inline void sm80_prepack_weights_ref(
    int rows,
    int columns,
    const MatrixRef<uint8_t const, ColumnMajorLayout, true>& tensor_weight,
    const MatrixRef<uint8_t, ColumnMajorLayout, true>& tensor_weight_prepacked) {
  ORT_ENFORCE(tensor_weight.shape()[0] == rows / 2 && tensor_weight.shape()[1] == columns,
              "Unexpected tensor_weight shape! Expected: (", rows / 2, ", ", columns, "), Got: (",
              tensor_weight.shape()[0], ", ", tensor_weight.shape()[1], ").");
  ORT_ENFORCE(tensor_weight_prepacked.shape()[0] == rows && tensor_weight_prepacked.shape()[1] == columns / 2,
              "tensor_weight_prepacked shape is not compatible with prepacked weight shape");

  auto t0_base = make_Position(0, 0);
  auto t1_base = make_Position(4, 0);
  auto t2_base = make_Position(0, 8);
  auto t3_base = make_Position(4, 8);
  for (int col_dtile = 0; col_dtile < columns / 16; ++col_dtile) {
    for (int row_dtile = 0; row_dtile < rows / 16; ++row_dtile) {
      // Packing from a 8x16 tile to a 16x8 tile
      auto dtile_base = make_Position(row_dtile * 8, col_dtile * 16);
      auto packed_tile_base = make_Position(row_dtile * 16, col_dtile * 8);
      for (int col = 0; col < 8; ++col) {
        for (int row = 0; row < 4; ++row) {
          auto cord = make_Position(row, col);
          auto packed_cord = packed_tile_base + make_Position(row * 4, col);  // packed tile is 16x8
          uint8_t buf[4];
          buf[0] = tensor_weight.at(dtile_base + t0_base + cord);
          buf[1] = tensor_weight.at(dtile_base + t1_base + cord);
          buf[2] = tensor_weight.at(dtile_base + t2_base + cord);
          buf[3] = tensor_weight.at(dtile_base + t3_base + cord);

          // [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7] so that each pair of adjacent weights
          // are in different b16 register at the same positions. This makes it easier to convert to
          // fp16x2 format in a b32 register

          tensor_weight_prepacked.at(packed_cord) = (buf[0] & 0x0f) | ((buf[1] & 0x0f) << 4);
          tensor_weight_prepacked.at(packed_cord + make_Position(1, 0)) = (buf[2] & 0x0f) | ((buf[3] & 0x0f) << 4);
          tensor_weight_prepacked.at(packed_cord + make_Position(2, 0)) = ((buf[0] & 0xf0) >> 4) | (buf[1] & 0xf0);
          tensor_weight_prepacked.at(packed_cord + make_Position(3, 0)) = ((buf[2] & 0xf0) >> 4) | (buf[3] & 0xf0);
        }
      }
    }
  }
}

template <
    typename ScaleElementT,
    typename Layout,
    typename QuantBlocking>
inline void sm80_prepack_quant_scales_ref(
    int rows,
    int columns,
    const MatrixRef<ScaleElementT const, Layout, true>& tensor_scale,
    const MatrixRef<ScaleElementT, Layout, true>& tensor_scale_prepacked) {
  ORT_ENFORCE(tensor_scale.shape()[0] == (rows / QuantBlocking::kRow) && tensor_scale.shape()[1] ==
                                                                             (columns / QuantBlocking::kColumn),
              "Unexpected tensor_scale shape! Expected: (",
              rows / QuantBlocking::kRow, ", ", columns / QuantBlocking::kColumn, ")");
  ORT_ENFORCE(tensor_scale_prepacked.shape() == tensor_scale.shape());

  // Only prepacking scale and offset tensors for a often used special case:
  //    16b gemm (2 elements per 32b register, operand tile shape 8x8)
  //    2 B operand tiles per mma instruction stacked on k dimension
  //    (1,n) quantization blocking
  if constexpr (sizeof(ScaleElementT) != 2 || QuantBlocking::kRow != 1) {
    ORT_THROW(
        "sm80_prepack_quant_scales_ref should only be called for "
        " row-wise block quantization on 16b float values.");
  }

  // In Ampere tensor op, each operand B tile is 8 x 8, in a warp of 32 threads, each thread
  // holds a fragment of the tile containing 2 elements in the k dimension. Most often we use
  // mma instruction shape of 16x8x16, which means 2 B tiles are stacked in the k dimension,
  // as shown below (T stands for thread):
  // T0, T4, T8, T12
  // T1, T5, T9, T13
  // T2, T6, T10, T14
  // T3, T7, T11, T15
  // T0, T4, T8, T12
  // T1, T5, T9, T13
  // T2, T6, T10, T14
  // T3, T7, T11, T15
  //
  // We need to deliver quantization scale and offset elements to the corresponding threads,
  // so we can perform dequantization efficiently. With a column major layout, each thread
  // needs two separate loads for a mma instruction, due to the tile fragment layout shown
  // above. To reduce the number of loads, we rearrange each column as below, so we can use
  // a single load to load fragments for two tiles:
  // T0        T0
  // T1        T0
  // T2        T1
  // T3   =>   T1
  // T0        T2
  // T1        T2
  // T2        T3
  // T3        T3

  for (int col = 0; col < tensor_scale.shape()[1]; ++col) {
    for (int row_blk = 0; row_blk < tensor_scale.shape()[0]; row_blk += 16) {
      for (int thread_id = 0; thread_id < 4; thread_id++) {
        const int dst_idx = row_blk + thread_id * 4;
        const int src_idx = row_blk + thread_id * 2;
        tensor_scale_prepacked.at(dst_idx + 0, col) = tensor_scale.at(src_idx + 0, col);
        tensor_scale_prepacked.at(dst_idx + 1, col) = tensor_scale.at(src_idx + 1, col);
        tensor_scale_prepacked.at(dst_idx + 2, col) = tensor_scale.at(src_idx + 8, col);
        tensor_scale_prepacked.at(dst_idx + 3, col) = tensor_scale.at(src_idx + 9, col);
      }
    }
  }
}

template <typename Layout, typename QuantBlocking>
inline void sm80_prepack_quant_offsets_ref(
    int rows,
    int columns,
    MatrixRef<uint8_t const, Layout, true> tensor_offset,
    MatrixRef<uint8_t, Layout, true> tensor_offset_prepacked) {
  const auto meta_shape = make_Position(rows / QuantBlocking::kRow, columns / QuantBlocking::kColumn);
  const auto zp_shape = make_Position((meta_shape[0] + 1) / 2, meta_shape[1]);
  ORT_ENFORCE(tensor_offset_prepacked.shape() == meta_shape,
              "Unexpected tensor_offset_prepacked shape (",
              tensor_offset_prepacked.shape()[0], ",", tensor_offset_prepacked.shape()[1],
              ")! Expected: (", meta_shape[0], ", ", meta_shape[1], ")");
  ORT_ENFORCE(tensor_offset.shape() == zp_shape,
              "Unexpected tensor_offset shape (",
              tensor_offset.shape()[0], ",", tensor_offset.shape()[1],
              ")! Expected: (", zp_shape[0], ", ", zp_shape[1], ")");

  // Only prepacking scale and offset tensors for a often used special case:
  //    16b gemm (2 elements per 32b register, operand tile shape 8x8)
  //    2 B operand tiles per mma instruction stacked on k dimension
  //    (1,n) quantization blocking
  if constexpr (QuantBlocking::kRow != 1) {
    ORT_THROW("sm80_prepack_quant_offsets_ref should only be called for row-wise block quantization.");
  }
  // In Ampere tensor op, each operand B tile is 8 x 8, in a warp of 32 threads, each thread
  // holds a fragment of the tile containing 2 elements in the k dimension. Most often we use
  // mma instruction shape of 16x8x16, which means 2 B tiles are stacked in the k dimension,
  // as shown below (T stands for thread):
  // T0, T4, T8, T12
  // T1, T5, T9, T13
  // T2, T6, T10, T14
  // T3, T7, T11, T15
  // T0, T4, T8, T12
  // T1, T5, T9, T13
  // T2, T6, T10, T14
  // T3, T7, T11, T15
  //
  // We need to deliver quantization scale and offset elements to the corresponding threads,
  // so we can perform dequantization efficiently. With a column major layout, each thread
  // needs two separate loads for a mma instruction, due to the tile fragment layout shown
  // above. To reduce the number of loads, we rearrange each column as below, so we can use
  // a single load to load fragments for two tiles:
  // T0        T0
  // T1        T0
  // T2        T1
  // T3   =>   T1
  // T0        T2
  // T1        T2
  // T2        T3
  // T3        T3
  if (tensor_offset_prepacked.good()) {
    for (int col = 0; col < tensor_offset_prepacked.shape()[1]; ++col) {
      for (int row_blk = 0; row_blk < tensor_offset_prepacked.shape()[0]; row_blk += 16) {
        for (int thread_id = 0; thread_id < 4; thread_id++) {
          const int dst_idx = row_blk + thread_id * 4;
          const int src_idx = row_blk + thread_id * 2;
          // [a, b, c, d] => [a, c, b, d] so that adjacent weights are in their own
          // 16b element: [a, x, b, x] and [x, c, x, d], which makes it easier to
          // convert to fp16x2 format in a b32 register
          uint8_t pair01 = tensor_offset.at(src_idx / 2, col);
          uint8_t pair89 = tensor_offset.at((src_idx + 8) / 2, col);
          tensor_offset_prepacked.at(dst_idx + 0, col) = pair01 & 0xf;
          tensor_offset_prepacked.at(dst_idx + 1, col) = pair89 & 0xf;
          tensor_offset_prepacked.at(dst_idx + 2, col) = pair01 >> 4;
          tensor_offset_prepacked.at(dst_idx + 3, col) = pair89 >> 4;
        }
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
