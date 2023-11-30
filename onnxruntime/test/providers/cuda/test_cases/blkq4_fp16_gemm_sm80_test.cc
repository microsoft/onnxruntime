/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    blkq4_fp16_gemm_sm80_test.cc
 *
 * Abstract:
 *   Test code for block-wise quantized 4b GEMM kernels.
 *   This part requires gtest header files, which do not play
 *   well with CUTLASS headers.
 */

#include <random>

#include "core/framework/float16.h"
#include "core/mickey/blk_q4/f16_prepack_sm80.h"
#include "core/mlas/inc/mlas_q4.h"

#include "blkq4_fp16_gemm_sm80.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

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
  onnxruntime::cuda::test::prepack_weights_ref(rows, columns, tensor_q_weight, tensor_packed_w_ref);

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
    onnxruntime::cuda::test::prepack_quant_scales_ref<ElementT, LayoutQmeta, QuantBlocking>(
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
      onnxruntime::cuda::test::prepack_quant_offsets_ref<LayoutQmeta, QuantBlocking>(
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
  Status status = onnxruntime::cuda::test::sm80_supported();
  if (!status.IsOK()) {
    // skip the test if sm80 is not supported
    return;
  }

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

TEST(BlkQ4_GEMM, Sm80Test) {
  Status status = onnxruntime::cuda::test::sm80_supported();
  if (!status.IsOK()) {
    // skip the test if sm80 is not supported
    return;
  }

  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(32, 32, 64);
  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, true>(32, 32, 64);

  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(32, 96, 64);
  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, true>(32, 96, 64);

  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(32, 96, 192);
  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, true>(32, 96, 192);

  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(256, 672, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, true>(256, 672, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(512, 2048 + 32, 960);
  onnxruntime::cuda::test::run_blkq4_gemm<32, false, false, false>(512, 2048 + 32, 960);

  onnxruntime::cuda::test::run_blkq4_gemm<16, false, false, false>(256, 672, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<16, false, false, true>(256, 672, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<64, false, false, false>(256, 1024, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<64, false, false, true>(256, 1024, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<16, true, false, false>(256, 672, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<16, true, false, true>(256, 672, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<64, true, false, false>(256, 1024, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<64, true, false, true>(256, 1024, 576);

  // small m
  onnxruntime::cuda::test::run_blkq4_gemm<16, false, true, false>(16, 704, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<16, false, true, true>(16, 704, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<64, false, true, false>(16, 1024, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<64, false, true, true>(16, 1024, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<16, true, true, false>(16, 672, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<16, true, true, true>(16, 672, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<64, true, true, false>(16, 1024, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<64, true, true, true>(16, 1024, 576);
}

}  // namespace test
}  // namespace onnxruntime
