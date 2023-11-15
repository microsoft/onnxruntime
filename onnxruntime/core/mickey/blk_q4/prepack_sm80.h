// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// #include <cuda.h>
// #include <vector_types.h>
// #include "cutlass/cutlass.h"
// #include "cutlass/matrix_shape.h"
// #include "cutlass/util/host_tensor.h"
// #include "cutlass/util/reference/host/tensor_compare.h"
// #include "cutlass/util/reference/host/tensor_copy.h"
// #include "cutlass/util/reference/host/tensor_fill.h"

#include "core/common/common.h"
#include "core/util/matrix_layout.h"

namespace onnxruntime {
namespace cuda {


/**
 * @brief Blockwise quantization methods
 * @tparam ElementT       source data type, fp16
 * @tparam block_size     number of elemenets quantized together
 * @tparam qbits          number of bits in each quantized element
 * @tparam Columnwise     true:  elements in a block come from one single column
 *                        false: elements in a block come from one single row
 */
template <
    typename ElementT,
    int block_size,
    int qbits,
    bool Columnwise,
    bool ExtraBoundsCheck = false>
struct BlockwiseQuantization {
  static_assert(qbits == 4, "Only 4b block quantization is supported!");
  static_assert(sizeof(ElementT) == 2, "Only 16b floating point types are supported!");

  using QuantBlocking =
      std::conditional_t<Columnwise,
          MatrixShape<block_size, 1>,
          MatrixShape<1, block_size>>;

  using ElementW = uint8_t;                 // <- Weight is int4, uint8 for two of them
  // We pack 4 weights into one 16b element, so we can leverage cutlass tile iterators
  // for async share memory loading, and minimizing bank conflict during matrix loading
  using ElementWPack = ElementT;
  using LayoutWPack = ColumnMajorLayout;    // <- layout of packed weight, must be column major

  // Current Ampere kernel use 8b zero point, need to shrink it to 4b in the future
  using ElementQOffset = uint8_t;

  // Layout of the quantization parameters (scales and zero points)
  // Major on the dimension that has the most parameters per squarish weight block.
  // E.g. for column-wise quantization, a [64, 64] block has [2, 64] parameters,
  // where each row has more data, so we use row major layout so that warp threads
  // can use less load instructions to load more parameters.
  using LayoutQmeta =
      typename std::conditional<Columnwise,
          RowMajorLayout, ColumnMajorLayout>::type;

  /**
   * @brief  Get quantized weight tensor dimensions.
   * Actual weight type is int4, we use ElementW = uint8 to avoid possible compilation
   * troubles. Since the layout is column major, we are packing 2 weights in a column
   * into one int8
   */
  static inline auto get_quant_weights_shape(int rows, int columns) {
    return make_Position((rows + 1) / 2, columns);
  }

  static inline auto get_quant_meta_shape(int rows, int columns) {
    return make_Position(rows / QuantBlocking::kRow, columns / QuantBlocking::kColumn);
  }

  /**
   * @brief Prepack weight matrix to facilitate matrix loading, depending on MMA
   * instruction layout.
   *
   * The weight matrix is int4, yet we want to leverage existing fp16/bf16
   * tile loading and MMA layout code in CUTLASS. So we group 4 int4 into 2
   * bytes, pretending it's fp16. This grouping must be done in a way to be
   * easily unpacked into tiles that match the MMA instruction layout.
   * For MMA instruction <16, 8, 16>, each instruction processes 2 8x8 tiles,
   * vertically stacked on the K dimension. And MmaTensorOpMultiplicandTileIterator
   * loads a <InstructionShape::kK, WarpShape::kN> tile.
   *
   * So we stack 2x2 tiles on a 3rd dimeansion, and reshape them in a HWC fashion:
   * T0, T2
   * T1, T3
   * ==>
   * T0[0, 0], T1[0, 0], T2[0, 0], T3[0, 0]
   * T0[1, 0], T1[1, 0], T2[1, 0], T3[1, 0]
   * T0[2, 0], T1[2, 0], T2[2, 0], T3[2, 0]
   * T0[3, 0], T1[3, 0], T2[3, 0], T3[3, 0]
   * ...
   * T0[0, 7], T1[0, 7], T2[0, 7], T3[0, 7]
   * T0[1, 7], T1[1, 7], T2[1, 7], T3[1, 7]
   * T0[2, 7], T1[2, 7], T2[2, 7], T3[2, 7]
   * T0[3, 7], T1[3, 7], T2[3, 7], T3[3, 7]
   *
   * This pack a 8x16 int8 tile into a 16x8 int8 tile, i.e. a 8x8 16b tile
  */
  static void prepack_weights(
    int rows,
    int columns,
    const gsl::span<uint8_t const>& weights,     // <- int4 weights, column major
    const gsl::span<uint8_t>& weights_prepacked  // <- int4 prepacked weights tensor, same size buffer
    ) {

    ORT_ENFORCE((rows % 16) == 0 && (columns % 16) == 0,
                "Does not support odd number of rows or columns!");
    ORT_ENFORCE(weights.size() == size_t(rows * columns / 2),
                "Weight tensor shape mismatch!");
    ORT_ENFORCE(weights_prepacked.size() == weights.size(),
                "Prepacked Weight tensor buffer should be the same size!");

    const MatrixRef<uint8_t const, ColumnMajorLayout, ExtraBoundsCheck> tensor_weight
        (weights, make_Position(rows/2, columns));
    const MatrixRef<uint8_t, LayoutWPack, ExtraBoundsCheck> tensor_weight_prepacked
        (weights_prepacked, make_Position(rows, columns / 2));

    // TODO!! parallized this.
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
            auto packed_cord = packed_tile_base + make_Position(row * 4, col); // packed tile is 16x8
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

  /**
   * @brief We rearrange the values of the quantization scale and offset tensors
   * to facilitate faster loading to tensor core, only 16b gemm, and (1,n)
   * block quantization.
   *
   * TODO!! also only in sm80 the mma tile is 16x8x16.
  */
  static constexpr bool ShouldRearrangeMeta = sizeof(ElementT) == 2 && QuantBlocking::kRow == 1;

  static void prepack_quant_scales(
      size_t rows,
      size_t columns,
      const gsl::span<ElementT const>& scales,     // <- quant scales, column major layout
      const gsl::span<ElementT>& scales_prepacked  // <- quant scales prepacked, same size buffer
      ) {

    auto meta_shape = get_quant_meta_shape(rows, columns);
    ORT_ENFORCE(scales.size() == size_t(meta_shape.product()),
                "Quantization scale tensor shape mismatch!");
    ORT_ENFORCE(scales_prepacked.size() == size_t(meta_shape.product()),
                "Prepacked quantization scale tensor buffer should be the same size!");

    MatrixRef<ElementT const, ColumnMajorLayout, ExtraBoundsCheck> tensor_scale(scales, meta_shape);
    MatrixRef<ElementT, LayoutQmeta, ExtraBoundsCheck> tensor_scale_prepacked(scales_prepacked, meta_shape);

    // Only prepacking scale and offset tensors for a often used special case:
    //    16b gemm (2 elements per 32b register, operand tile shape 8x8)
    //    2 B operand tiles per mma instruction stacked on k dimension
    //    (1,n) quantization blocking
    if constexpr(sizeof(ElementT) == 2 &&  QuantBlocking::kRow == 1){
        // In Ampere tensor op, each operand B tile is 8 x 8, in a warp of 32 threads, each thread
        // holds a fragement of the tile containing 2 elements in the k dimension. Most often we use
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
        // needs two seperate loads for a mma instruction, due to the tile fragement layout shown
        // above. To reduce the number of loads, we rearrange each column as below, so we can use
        // a single load to load fragements for two tiles:
        // T0        T0
        // T1        T0
        // T2        T1
        // T3   =>   T1
        // T0        T2
        // T1        T2
        // T2        T3
        // T3        T3

        for (int col = 0; col < tensor_scale.shape()[1]; ++col){
          for (int row_blk = 0; row_blk < tensor_scale.shape()[0]; row_blk += 16){
            for (int thread_id = 0; thread_id < 4; thread_id++){
              const int dst_idx = row_blk + thread_id * 4;
              const int src_idx = row_blk + thread_id * 2;
              tensor_scale_prepacked.at(dst_idx + 0, col) = tensor_scale.at(src_idx + 0, col);
              tensor_scale_prepacked.at(dst_idx + 1, col) = tensor_scale.at(src_idx + 1, col);
              tensor_scale_prepacked.at(dst_idx + 2, col) = tensor_scale.at(src_idx + 8, col);
              tensor_scale_prepacked.at(dst_idx + 3, col) = tensor_scale.at(src_idx + 9, col);
            }
          }
        }
    } else {
      // In all other cases, we don't prepack scale or offset
      // Potential transpose if the prepacked layout is different from the original layout
      for (int col = 0; col < tensor_scale.shape()[1]; ++col){
        for (int row = 0; row < tensor_scale.shape()[0]; ++row){
          tensor_scale_prepacked.at(row, col) = tensor_scale.at(row, col);
        }
      }
    }
  }

  static void prepack_quant_offsets(
      size_t rows,
      size_t columns,
      const gsl::span<uint8_t const>& offsets,     // <- quant offsets, int4, column major layout
      const gsl::span<uint8_t>& offsets_prepacked  // <- quant offsets prepacked, double size buffer
      ) {
    auto meta_shape = get_quant_meta_shape(rows, columns);

    ORT_ENFORCE((rows % 16) == 0 && (columns % 16) == 0,
                "Does not support odd number of rows or columns!");
    ORT_ENFORCE(offsets_prepacked.size() == size_t(meta_shape.product()),
                "Wrong buffer size for prepacked quantization offsets!");
    ORT_ENFORCE(offsets.size() == size_t(((meta_shape[0] + 1) / 2) * meta_shape[1]),
                "Quantization offset tensor shape mismatch!");

    MatrixRef<uint8_t const, ColumnMajorLayout, ExtraBoundsCheck> tensor_offset
        (offsets, make_Position((meta_shape[0] + 1) / 2, meta_shape[1]));
    MatrixRef<uint8_t, LayoutQmeta, ExtraBoundsCheck> tensor_offset_prepacked
        (offsets_prepacked, meta_shape);

    // Only prepacking scale and offset tensors for a often used special case:
    //    16b gemm (2 elements per 32b register, operand tile shape 8x8)
    //    2 B operand tiles per mma instruction stacked on k dimension
    //    (1,n) quantization blocking
    if constexpr(sizeof(ElementT) == 2 && QuantBlocking::kRow == 1){
      // In Ampere tensor op, each operand B tile is 8 x 8, in a warp of 32 threads, each thread
      // holds a fragement of the tile containing 2 elements in the k dimension. Most often we use
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
      // needs two seperate loads for a mma instruction, due to the tile fragement layout shown
      // above. To reduce the number of loads, we rearrange each column as below, so we can use
      // a single load to load fragements for two tiles:
      // T0        T0
      // T1        T0
      // T2        T1
      // T3   =>   T1
      // T0        T2
      // T1        T2
      // T2        T3
      // T3        T3
      for (int col = 0; col < meta_shape[1]; ++col){
        for (int row_blk = 0; row_blk < meta_shape[0]; row_blk += 16){
          for (int thread_id = 0; thread_id < 4; thread_id++){
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
    } else {
      // In all other cases, we don't prepack scale or offset
      // Potential transpose if the prepacked layout is different from the original layout
      for (int col = 0; col < meta_shape[1]; ++col){
        for (int row = 0; row < meta_shape[0]; row += 2){
          uint8_t pair01 = tensor_offset.at(row / 2, col);
          tensor_offset_prepacked.at(row + 0, col) = pair01 & 0xf;
          if (row + 1 < meta_shape[0]) {
            tensor_offset_prepacked.at(row + 1, col) = pair01 >> 4;
          }
        }
      }
    }
  }

};


}  // namespace cuda
}  // namespace onnxruntime
