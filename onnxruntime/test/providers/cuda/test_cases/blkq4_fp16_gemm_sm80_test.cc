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
#include "core/mlas/inc/mlas_q4.h"

#include "blkq4_fp16_gemm_sm80.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

template <bool col_blocking, bool has_offset = true>
void testPrepack(int rows, int columns) {
  using ElementT = MLFloat16;
  constexpr int block_size = 32;
  using Base = onnxruntime::cuda::BlockwiseQuantization<
      ElementT,
      block_size,
      4,
      col_blocking>;

  using QuantBlocking = typename Base::QuantBlocking;
  using ElementW = typename Base::ElementW;
  using LayoutWPack = typename Base::LayoutWPack;
  using ElementQOffset = typename Base::ElementQOffset;
  using LayoutQmeta = typename Base::LayoutQmeta;

  const auto q_weight_shape = Base::get_quant_weights_shape(rows, columns);
  const auto meta_shape = Base::get_quant_meta_shape(rows, columns);
  const auto zp_shape = make_Position((meta_shape[0] + 1) / 2, meta_shape[1]);

  std::vector<ElementW> q_weights;
  std::vector<ElementT> q_scales;
  std::vector<ElementQOffset> q_zp;
  std::vector<ElementT> dequants;
  onnxruntime::cuda::test::blkq4_weights_gen<ElementT, block_size, col_blocking, has_offset>(
      rows, columns, dequants, q_weights, q_scales, q_zp);

  // for quantization tool, the input is row major, all outputs are column major
  MatrixRef<ElementW, ColumnMajorLayout, true> tensor_q_weight(
      q_weights, make_Position(rows / 2, columns));
  MatrixRef<ElementT, ColumnMajorLayout, true> tensor_scale(
      q_scales, meta_shape);
  MatrixRef<ElementQOffset, ColumnMajorLayout, true> tensor_offset;
  if constexpr (has_offset) {
    tensor_offset = MatrixRef<ElementQOffset, ColumnMajorLayout, true>(q_zp, zp_shape);
  }

  // for quantization tool, the input is row major, test weight gen output is column major
  std::vector<ElementT> dequants_transposed(dequants.size());
  MatrixRef<ElementT, ColumnMajorLayout> tensor_dequant(dequants, make_Position(rows, columns));
  MatrixRef<ElementT, RowMajorLayout> tensor_dequant_transposed(dequants_transposed, make_Position(rows, columns));
  for (int col = 0; col < tensor_dequant.shape()[1]; ++col) {
    for (int row = 0; row < tensor_dequant.shape()[0]; ++row) {
      tensor_dequant_transposed.at(row, col) = tensor_dequant.at(row, col);
    }
  }

  int q_rows, q_cols;
  MlasBlockwiseQuantizedShape<ElementT, 4>(
      block_size, col_blocking, rows, columns, q_rows, q_cols);
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

  std::vector<uint8_t> o_zp(zp_shape.product());
  MatrixRef<uint8_t, ColumnMajorLayout, true> tensor_o_zp(o_zp, zp_shape);

  MlasQuantizeBlockwise<MLFloat16, 4>(o_elements.data(), o_scales.data(), has_offset ? o_zp.data() : nullptr,
                                      dequants_transposed.data(), block_size,
                                      col_blocking, rows, columns, columns, nullptr);
  for (int col = 0; col < tensor_q_weight.shape()[1]; ++col) {
    for (int row = 0; row < tensor_q_weight.shape()[0]; ++row) {
      EXPECT_EQ(tensor_o_elements.at(row, col), tensor_q_weight.at(row, col))
          << "quantized value mismatch at [" << row << "," << col << "]"
          << " shape[" << rows << "," << columns << "]"
          << (col_blocking ? "Column-wise-block" : "Row-wise-block")
          << std::endl;
    }
  }

  for (int col = 0; col < meta_shape[1]; ++col) {
    for (int row = 0; row < meta_shape[0]; row += 2) {
      if (has_offset) {
        uint8_t pair01 = tensor_o_zp.at(row / 2, col);
        uint8_t expected_pair01 = tensor_offset.at(row / 2, col);
        EXPECT_EQ(expected_pair01 & 0xf, pair01 & 0xf)
            << "quantized offset mismatch at [" << row << "," << col << "]"
            << " shape[" << rows << "," << columns << "]"
            << (col_blocking ? "Column-wise-block" : "Row-wise-block")
            << std::endl;
        if (row + 1 < meta_shape[0]) {
          EXPECT_EQ(expected_pair01 >> 4, pair01 >> 4)
              << "quantized offset mismatch at [" << row + 1 << "," << col << "]"
              << " shape[" << rows << "," << columns << "]"
              << (col_blocking ? "Column-wise-block" : "Row-wise-block")
              << std::endl;
        }
      }

      EXPECT_EQ(tensor_scale.at(row + 0, col), tensor_o_scales.at(row + 0, col))
          << "quantized scale mismatch at [" << row << "," << col << "]"
          << " shape[" << rows << "," << columns << "]"
          << (col_blocking ? "Column-wise-block" : "Row-wise-block")
          << std::endl;
      if (row + 1 < meta_shape[0]) {
        EXPECT_EQ(tensor_scale.at(row + 1, col), tensor_o_scales.at(row + 1, col))
            << "quantized scale mismatch at [" << row + 1 << "," << col << "]"
            << " shape[" << rows << "," << columns << "]"
            << (col_blocking ? "Column-wise-block" : "Row-wise-block")
            << std::endl;
      }
    }
  }

  //
  // Now we just setup quantized weights tensor_q_weight, quantization scale tensor_scale
  // and quantization offset tensor_offset. The above tests just make sure our setup is
  // consistent with quantization tool output.
  //
  // Next we test the prepack code
  //

  std::vector<ElementW> packed_w_ref(q_weight_shape.product());
  MatrixRef<ElementW, LayoutWPack, true> tensor_packed_w_ref(
      packed_w_ref, make_Position(rows, columns / 2));
  onnxruntime::test::sm80_prepack_weights_ref(rows, columns, tensor_q_weight, tensor_packed_w_ref);

  std::vector<ElementW> packed_w(q_weight_shape.product());
  MatrixRef<ElementW, LayoutWPack, true> tensor_packed_w(
      packed_w, make_Position(rows, columns / 2));
  Base::prepack_weights(rows, columns, o_elements, packed_w);

  for (int col = 0; col < tensor_packed_w.shape()[1]; ++col) {
    for (int row = 0; row < tensor_packed_w.shape()[0]; ++row) {
      EXPECT_EQ(tensor_packed_w_ref.at(row, col), tensor_packed_w.at(row, col))
          << "prepacked weights mismatch at [" << row << "," << col << "]"
          << " shape[" << rows << "," << columns << "]"
          << (col_blocking ? "Column-wise-block" : "Row-wise-block")
          << std::endl;
    }
  }

  std::vector<ElementT> packed_scales_ref(meta_shape.product());
  MatrixRef<ElementT, LayoutQmeta, true> tensor_packed_s_ref =
      make_MatrixRef<ElementT, LayoutQmeta, true>(packed_scales_ref, meta_shape);
  if constexpr (Base::ShouldRearrangeMeta) {
    onnxruntime::test::sm80_prepack_quant_scales_ref<ElementT, LayoutQmeta, QuantBlocking>(
        rows, columns, tensor_scale.const_ref(), tensor_packed_s_ref);
  } else {
    for (int col = 0; col < tensor_packed_s_ref.shape()[1]; ++col) {
      for (int row = 0; row < tensor_packed_s_ref.shape()[0]; ++row) {
        tensor_packed_s_ref.at(row, col) = tensor_scale.at(row, col);
      }
    }
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
          << (col_blocking ? "Column-wise-block" : "Row-wise-block")
          << std::endl;
    }
  }

  if (has_offset) {
    std::vector<ElementQOffset> packed_zp_ref(meta_shape.product());
    MatrixRef<ElementQOffset, LayoutQmeta, true> tensor_packed_zp_ref =
        make_MatrixRef<ElementQOffset, LayoutQmeta, true>(packed_zp_ref, meta_shape);
    if constexpr (Base::ShouldRearrangeMeta) {
      onnxruntime::test::sm80_prepack_quant_offsets_ref<LayoutQmeta, QuantBlocking>(
          rows, columns, tensor_offset.const_ref(), tensor_packed_zp_ref);
    } else {
      for (int col = 0; col < meta_shape[1]; ++col) {
        for (int row = 0; row < meta_shape[0]; row += 2) {
          uint8_t pair01 = tensor_offset.at(row / 2, col);
          tensor_packed_zp_ref.at(row, col) = pair01 & 0xf;
          if (row + 1 < meta_shape[0]) {
            tensor_packed_zp_ref.at(row + 1, col) = pair01 >> 4;
          }
        }
      }
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
            << (col_blocking ? "Column-wise-block" : "Row-wise-block")
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
  testPrepack<false, false>(32, 32);
  testPrepack<true>(32, 32);
  testPrepack<true, false>(32, 32);
  testPrepack<false>(32, 64);
  testPrepack<false>(32, 128);
  testPrepack<false>(32, 256);
  testPrepack<false>(64, 32);
  testPrepack<false>(128, 32);
  testPrepack<false>(256, 32);
  testPrepack<false>(256, 256);
  testPrepack<false, false>(32, 128);
  testPrepack<false, false>(128, 32);
  testPrepack<false, false>(256, 256);
  testPrepack<true>(32, 64);
  testPrepack<true>(32, 128);
  testPrepack<true>(32, 256);
  testPrepack<true>(64, 32);
  testPrepack<true>(128, 32);
  testPrepack<true>(256, 32);
  testPrepack<true>(256, 256);
  testPrepack<true, false>(32, 128);
  testPrepack<true, false>(128, 32);
  testPrepack<true, false>(256, 256);
}

TEST(BlkQ4_GEMM, Sm80RowBlockingTest) {
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
}

TEST(BlkQ4_GEMM, Sm80ColBlockingTest) {
  Status status = onnxruntime::cuda::test::sm80_supported();
  if (!status.IsOK()) {
    // skip the test if sm80 is not supported
    return;
  }
  onnxruntime::cuda::test::run_blkq4_gemm<16, true, false, false>(64, 672, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<16, true, false, true>(64, 672, 576);

  onnxruntime::cuda::test::run_blkq4_gemm<64, true, false, false>(256, 1024, 576);
  onnxruntime::cuda::test::run_blkq4_gemm<64, true, false, true>(256, 1024, 576);
}

TEST(BlkQ4_GEMM, Sm80SmallMTest) {
  Status status = onnxruntime::cuda::test::sm80_supported();
  if (!status.IsOK()) {
    // skip the test if sm80 is not supported
    return;
  }

  // // small m
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
