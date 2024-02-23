// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>

#include "core/framework/float16.h"
#include "core/mickey/blk_q4/prepack_sm80.h"
#include "core/mlas/inc/mlas_q4.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

void prepack_weights_ref(
    int rows,
    int columns,
    const MatrixRef<uint8_t const, ColumnMajorLayout, true>& tensor_weight,
    const MatrixRef<uint8_t, ColumnMajorLayout, true>& tensor_weight_prepacked) {
  EXPECT_TRUE(tensor_weight.shape()[0] == rows / 2 && tensor_weight.shape()[1] == columns);
  EXPECT_TRUE(tensor_weight_prepacked.shape()[0] == rows && tensor_weight_prepacked.shape()[1] == columns / 2);

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
void prepack_quant_scales_ref(
    int rows,
    int columns,
    const MatrixRef<ScaleElementT const, Layout, true>& tensor_scale,
    const MatrixRef<ScaleElementT, Layout, true>& tensor_scale_prepacked) {
  EXPECT_TRUE(tensor_scale.shape()[0] == (rows / QuantBlocking::kRow) && tensor_scale.shape()[1] == (columns / QuantBlocking::kColumn));
  EXPECT_TRUE(tensor_scale_prepacked.shape() == tensor_scale.shape());

  // Only prepacking scale and offset tensors for a often used special case:
  //    16b gemm (2 elements per 32b register, operand tile shape 8x8)
  //    2 B operand tiles per mma instruction stacked on k dimension
  //    (1,n) quantization blocking
  if constexpr (sizeof(ScaleElementT) == 2 && QuantBlocking::kRow == 1) {
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
  } else {
    // In all other cases, we don't prepack scale or offset
    FAIL() << "Scale prepack only supported for 16b gemm with (1,n) quantization blocking";
  }
}

template <typename Layout, typename QuantBlocking>
void prepack_quant_offsets_ref(
    size_t rows,
    size_t columns,
    MatrixRef<uint8_t const, Layout, true> tensor_offset,
    MatrixRef<uint8_t, Layout, true> tensor_offset_prepacked) {
  // EXPECT_TRUE(tensor_offset.shape()[0] == (rows / QuantBlocking::kRow) && tensor_offset.shape()[1] == (columns / QuantBlocking::kColumn));
  EXPECT_TRUE(tensor_offset_prepacked.shape() == tensor_offset.shape());

  // Only prepacking scale and offset tensors for a often used special case:
  //    16b gemm (2 elements per 32b register, operand tile shape 8x8)
  //    2 B operand tiles per mma instruction stacked on k dimension
  //    (1,n) quantization blocking
  if constexpr (QuantBlocking::kRow != 1) {
    FAIL() << "Offsets prepack only supported for 16b gemm with (1,n) quantization blocking";
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
    for (int col = 0; col < tensor_offset.shape()[1]; ++col) {
      for (int row_blk = 0; row_blk < tensor_offset.shape()[0]; row_blk += 16) {
        for (int thread_id = 0; thread_id < 4; thread_id++) {
          const int dst_idx = row_blk + thread_id * 4;
          const int src_idx = row_blk + thread_id * 2;
          // [a, b, c, d] => [a, c, b, d] so that adjacent weights are in their own
          // 16b element: [a, x, b, x] and [x, c, x, d], which makes it easier to
          // convert to fp16x2 format in a b32 register
          tensor_offset_prepacked.at(dst_idx + 0, col) = tensor_offset.at(src_idx + 0, col);
          tensor_offset_prepacked.at(dst_idx + 1, col) = tensor_offset.at(src_idx + 8, col);
          tensor_offset_prepacked.at(dst_idx + 2, col) = tensor_offset.at(src_idx + 1, col);
          tensor_offset_prepacked.at(dst_idx + 3, col) = tensor_offset.at(src_idx + 9, col);
        }
      }
    }
  }
}

template <bool ColumnMajorQuantBlocking>
void testPrepack(int rows, int columns, bool has_offset = true) {
  using ElementT = MLFloat16;
  constexpr int block_size = 32;
  using Base = onnxruntime::cuda::BlockwiseQuantization<
      ElementT,
      block_size,
      4,
      ColumnMajorQuantBlocking>;

  using QuantBlocking = typename Base::QuantBlocking;
  using ElementW = typename Base::ElementW;
  using LayoutWPack = typename Base::LayoutWPack;
  using ElementQOffset = typename Base::ElementQOffset;
  using LayoutQmeta = typename Base::LayoutQmeta;

  unsigned int seed = 28571;  // Replace with desired seed value
  std::seed_seq seq{seed};
  std::mt19937 gen(seq);
  std::uniform_int_distribution<> dis(0, 8192);

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

  std::vector<ElementW> q_weights(q_weight_shape.product());
  MatrixRef<ElementW, LayoutWPack, true> tensor_q_weight(
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

  std::vector<ElementT> q_scales(meta_shape.product());
  for (size_t i = 0; i < q_scales.size(); i++) {
    q_scales[i] = ElementT(((dis(gen) % 127) + 1) / 32.0f);
  }
  MatrixRef<ElementT, LayoutQmeta, true> tensor_scale(
      q_scales, meta_shape);

  std::vector<ElementQOffset> q_zp(meta_shape.product());
  for (size_t i = 0; i < q_zp.size(); i++) {
    q_zp[i] = dis(gen) % 16;
  }
  MatrixRef<ElementQOffset, LayoutQmeta, true> tensor_offset(
      q_zp, meta_shape);

#if 0  // debug
  // Fill tensor_q_weight with the patterned data, easier to debug with print
  int loop_val = 0;
  int offset = 3;
  for (int col_tile = 0; col_tile < tensor_q_weight.extent().column()/8; ++col_tile) {
    for (int row_tile = 0; row_tile < tensor_q_weight.extent().row()/4; ++row_tile) {
      for (int col = 0; col < 8; ++col) {
        for (int row = 0; row < 4; ++row) {
          auto weight_cord = cutlass::make_Coord(row_tile * 4 + row, col_tile * 8 + col);
          auto val = (loop_val + offset) % 256;
          tensor_q_weight.at(weight_cord) = ElementW(val);
          loop_val++;
          if (loop_val == 256) {
            loop_val = 0;
            offset += 11;
          }
        }
      }
    }
  }
  for (int col = 0; col < tensor_scale.extent().column(); ++col){
    int c =  col * QuantBlocking::kColumn;
    for (int row = 0; row < tensor_scale.extent().row(); ++row){
      int r = row * QuantBlocking::kRow;
      auto weight_cord = cutlass::make_Coord(r/2, c);
      int w = 0;
      if (r % 2 == 0) {
        w = int(tensor_q_weight.at(weight_cord) & 0x0f);
      } else {
        w = int(tensor_q_weight.at(weight_cord) >> 4);
      }
      tensor_scale.at({row, col}) = w;
      tensor_offset.at({row, col}) = ElementQOffset(w);
    }
  }

  int fill_val = -512;
  int factor = 1;
  for (int col = 0; col < tensor_scale.extent().column(); ++col){
    for (int row = 0; row < tensor_scale.extent().row(); ++row){
      tensor_scale.at({row, col}) = ElementQScale((float)fill_val * float(factor));
      fill_val++;
      if (fill_val == 512) {
        fill_val = -512;
        factor += 1;
      }
    }
  }

#endif  // debug

  std::vector<ElementT> dequants(rows * columns);
  MatrixRef<ElementT, RowMajorLayout> tensor_dequant(dequants, make_Position(rows, columns));

  // Dequantize weights and save into matrix B for reference
  for (int col = 0; col < tensor_dequant.shape()[1]; ++col) {
    for (int row = 0; row < tensor_dequant.shape()[0]; ++row) {
      auto weight_cord = make_Position(row / 2, col);
      auto scale_cord = make_Position(row / QuantBlocking::kRow, col / QuantBlocking::kColumn);
      const uint8_t offset = has_offset ? tensor_offset.at(scale_cord) : 8;
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

  int q_rows, q_cols;
  MlasBlockwiseQuantizedShape<ElementT, 4>(
      block_size, ColumnMajorQuantBlocking, rows, columns, q_rows, q_cols);
  // to be exact, q_rows are padded to multiple of block_size, deal with it when we care about strange shapes
  EXPECT_EQ(q_rows, q_weight_shape[0]);
  EXPECT_EQ(q_cols, q_weight_shape[1]);

  //
  // Quantization tool outputs:
  //
  std::vector<ElementW> o_elements(q_rows * q_cols);
  MatrixRef<ElementW, ColumnMajorLayout, true> tensor_o_elements(o_elements, q_weight_shape);

  std::vector<ElementT> o_scales(meta_shape.product());
  MatrixRef<ElementT, ColumnMajorLayout, true> tensor_o_scales(o_scales, meta_shape);

  std::vector<uint8_t> o_zp(((meta_shape[0] + 1) / 2) * meta_shape[1], true);
  MatrixRef<uint8_t, ColumnMajorLayout, true> tensor_o_zp(
      o_zp, make_Position((meta_shape[0] + 1) / 2, meta_shape[1]));

  MlasQuantizeBlockwise<MLFloat16, 4>(o_elements.data(), o_scales.data(), has_offset ? o_zp.data() : nullptr,
                                      tensor_dequant.data().data(), block_size,
                                      ColumnMajorQuantBlocking, rows, columns, columns, nullptr);
  for (int col = 0; col < tensor_q_weight.shape()[1]; ++col) {
    for (int row = 0; row < tensor_q_weight.shape()[0]; ++row) {
      EXPECT_EQ(tensor_o_elements.at(row, col), tensor_q_weight.at(row, col))
          << "quantized value mismatch at [" << row << "," << col << "]"
          << " shape[" << rows << "," << columns << "]"
          << (ColumnMajorQuantBlocking ? "Column-wise-block" : "Row-wise-block")
          << std::endl;
    }
  }

  for (int col = 0; col < meta_shape[1]; ++col) {
    for (int row = 0; row < meta_shape[0]; row += 2) {
      if (has_offset) {
        uint8_t pair01 = tensor_o_zp.at(row / 2, col);
        EXPECT_EQ(tensor_offset.at(row + 0, col), pair01 & 0xf)
            << "quantized offset mismatch at [" << row << "," << col << "]"
            << " shape[" << rows << "," << columns << "]"
            << (ColumnMajorQuantBlocking ? "Column-wise-block" : "Row-wise-block")
            << std::endl;
        if (row + 1 < meta_shape[0]) {
          EXPECT_EQ(tensor_offset.at(row + 1, col), pair01 >> 4)
              << "quantized offset mismatch at [" << row + 1 << "," << col << "]"
              << " shape[" << rows << "," << columns << "]"
              << (ColumnMajorQuantBlocking ? "Column-wise-block" : "Row-wise-block")
              << std::endl;
        }
      }

      EXPECT_EQ(tensor_scale.at(row + 0, col), tensor_o_scales.at(row + 0, col))
          << "quantized scale mismatch at [" << row << "," << col << "]"
          << " shape[" << rows << "," << columns << "]"
          << (ColumnMajorQuantBlocking ? "Column-wise-block" : "Row-wise-block")
          << std::endl;
      if (row + 1 < meta_shape[0]) {
        EXPECT_EQ(tensor_scale.at(row + 1, col), tensor_o_scales.at(row + 1, col))
            << "quantized scale mismatch at [" << row + 1 << "," << col << "]"
            << " shape[" << rows << "," << columns << "]"
            << (ColumnMajorQuantBlocking ? "Column-wise-block" : "Row-wise-block")
            << std::endl;
      }
    }
  }

  //
  // Now we just setup fp16 weights tensor_dequant, quantized weights tensor_q_weight,
  // quantization scale tensor_scale and quantization offset tensor_offset. The above
  // testing just make sure our test setup is consistent with quantization tool output.
  //
  // Next we test the prepack code
  //

  std::vector<ElementW> packed_w_ref(q_weight_shape.product());
  MatrixRef<ElementW, LayoutWPack, true> tensor_packed_w_ref(
      packed_w_ref, make_Position(rows, columns / 2));
  prepack_weights_ref(rows, columns, tensor_q_weight, tensor_packed_w_ref);

  std::vector<ElementW> packed_w(q_weight_shape.product());
  MatrixRef<ElementW, LayoutWPack, true> tensor_packed_w(
      packed_w, make_Position(rows, columns / 2));
  Base::prepack_weights(rows, columns, o_elements, packed_w);

  for (int col = 0; col < tensor_packed_w.shape()[1]; ++col) {
    for (int row = 0; row < tensor_packed_w.shape()[0]; ++row) {
      EXPECT_EQ(tensor_packed_w_ref.at(row, col), tensor_packed_w.at(row, col))
          << "prepacked weights mismatch at [" << row << "," << col << "]"
          << " shape[" << rows << "," << columns << "]"
          << (ColumnMajorQuantBlocking ? "Column-wise-block" : "Row-wise-block")
          << std::endl;
    }
  }

  std::vector<ElementT> packed_scales_ref(meta_shape.product());
  MatrixRef<ElementT, LayoutQmeta, true> tensor_packed_s_ref =
      Base::ShouldRearrangeMeta ? make_MatrixRef<ElementT, LayoutQmeta, true>(packed_scales_ref, meta_shape)
                                : tensor_scale;
  if (Base::ShouldRearrangeMeta) {
    prepack_quant_scales_ref<ElementT, LayoutQmeta, QuantBlocking>(
        rows, columns, tensor_scale.const_ref(), tensor_packed_s_ref);
  }

  std::vector<ElementT> packed_scales(meta_shape.product());
  MatrixRef<ElementT, LayoutQmeta, true> tensor_packed_s(
      packed_scales, meta_shape);
  Base::prepack_quant_scales(rows, columns, o_scales, packed_scales);

  for (int col = 0; col < tensor_packed_s.shape()[1]; ++col) {
    for (int row = 0; row < tensor_packed_s.shape()[0]; ++row) {
      EXPECT_EQ(tensor_packed_s_ref.at(row, col), tensor_packed_s.at(row, col))
          << "prepacked scales mismatch at [" << row << "," << col << "]"
          << " shape[" << rows << "," << columns << "]"
          << (ColumnMajorQuantBlocking ? "Column-wise-block" : "Row-wise-block")
          << std::endl;
    }
  }

  if (has_offset) {
    std::vector<ElementQOffset> packed_zp_ref(meta_shape.product());
    MatrixRef<ElementQOffset, LayoutQmeta, true> tensor_packed_zp_ref =
        Base::ShouldRearrangeMeta ? make_MatrixRef<ElementQOffset, LayoutQmeta, true>(packed_zp_ref, meta_shape)
                                  : tensor_offset;
    if (Base::ShouldRearrangeMeta) {
      prepack_quant_offsets_ref<LayoutQmeta, QuantBlocking>(
          rows, columns, tensor_offset.const_ref(), tensor_packed_zp_ref);
    }

    std::vector<ElementQOffset> packed_zp(meta_shape.product());
    MatrixRef<ElementQOffset, LayoutQmeta, true> tensor_packed_zp(
        packed_zp, meta_shape);
    Base::prepack_quant_offsets(rows, columns, o_zp, packed_zp);

    for (int col = 0; col < tensor_packed_zp.shape()[1]; ++col) {
      for (int row = 0; row < tensor_packed_zp.shape()[0]; ++row) {
        EXPECT_EQ(tensor_packed_zp_ref.at(row, col), tensor_packed_zp.at(row, col))
            << "prepacked offsets mismatch at [" << row << "," << col << "]"
            << " shape[" << rows << "," << columns << "]"
            << (ColumnMajorQuantBlocking ? "Column-wise-block" : "Row-wise-block")
            << std::endl;
      }
    }
  }
}

// TODO: code runs on CPU, but this is for sm80 only, maybe enable only when test on sm80
TEST(BlkQ4_GEMM, PrepackSm80Test) {
  testPrepack<false>(32, 32);
  testPrepack<false>(32, 32, false);
  testPrepack<true>(32, 32);
  testPrepack<true>(32, 32, false);
  testPrepack<false>(32, 64);
  testPrepack<false>(32, 128);
  testPrepack<false>(32, 256);
  testPrepack<false>(64, 32);
  testPrepack<false>(128, 32);
  testPrepack<false>(256, 32);
  testPrepack<false>(256, 256);
  testPrepack<false>(32, 128, false);
  testPrepack<false>(128, 32, false);
  testPrepack<false>(256, 256, false);
  testPrepack<true>(32, 64);
  testPrepack<true>(32, 128);
  testPrepack<true>(32, 256);
  testPrepack<true>(64, 32);
  testPrepack<true>(128, 32);
  testPrepack<true>(256, 32);
  testPrepack<true>(256, 256);
  testPrepack<true>(32, 128, false);
  testPrepack<true>(128, 32, false);
  testPrepack<true>(256, 256, false);
}

}  // namespace test
}  // namespace onnxruntime
