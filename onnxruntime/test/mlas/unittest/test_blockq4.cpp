/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_blockq4.cpp

Abstract:

    Tests for MLAS blockwise int4 quantization and dequantization code.

--*/

#ifndef ORT_MINIMAL_BUILD

#include "test_util.h"
#include "mlas_q4.h"

constexpr int Q2Bits = 2;
constexpr int Q4Bits = 4;

class MlasBlockwiseQdqTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> FpBuf;
  MatrixGuardBuffer<float> FpBuf2;
  MatrixGuardBuffer<uint8_t> InputElements;
  MatrixGuardBuffer<float> InputScales;
  MatrixGuardBuffer<uint8_t> InputOffsets;
  MatrixGuardBuffer<uint8_t> OutputElements;
  MatrixGuardBuffer<float> OutputScales;
  MatrixGuardBuffer<uint8_t> OutputOffsets;
  MatrixGuardBuffer<uint8_t> QDQOutputElements;
  MatrixGuardBuffer<float> QDQOutputScales;
  MatrixGuardBuffer<uint8_t> QDQOutputOffsets;
  MatrixGuardBuffer<uint8_t> QDQTransposedOutputElements;
  MatrixGuardBuffer<float> QDQTransposedOutputScales;
  MatrixGuardBuffer<uint8_t> QDQTransposedOutputOffsets;

  template<int qbits>
  void Test(int rows, int columns, int block_size, bool columnwise, bool symmetric) {
    float* dequant_buf = FpBuf.GetBuffer(rows * columns, true);
    float* transposed = FpBuf2.GetBuffer(rows * columns, true);
    size_t scale_size = (rows + block_size - 1) / block_size * columns;
    size_t zp_size = (scale_size + 1) / 2;

    MLAS_THREADPOOL* threadpool_ptr = GetMlasThreadPool();

    int meta_rows;
    int meta_cols;
    MlasBlockwiseQuantMetaShape<float, qbits>(block_size, columnwise, rows, columns, meta_rows, meta_cols);

    int q_rows;
    int q_cols;
    MlasBlockwiseQuantizedShape<float, qbits>(block_size, columnwise, rows, columns, q_rows, q_cols);

    size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
    MlasBlockwiseQuantizedBufferSizes<qbits>(block_size, columnwise, rows, columns,
                                      q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

    uint8_t* elements = InputElements.GetBuffer(q_data_size_in_bytes, true);
    uint8_t* qdq_weights = QDQOutputElements.GetBuffer((rows * columns + 1) / 2, true);
    uint8_t* qdq_weights_T = QDQTransposedOutputElements.GetBuffer(q_data_size_in_bytes, true);

    int pack_size = 8 / qbits;
    int v;
    if constexpr (qbits == 2) {
      v = 1;
      for (int c = 0; c < columns; c++) {
        for (int r = 0; r < rows; r += pack_size) {
          int idx = c * q_rows + r / pack_size;
          uint8_t v0 = static_cast<uint8_t>(v);
          v = (v + 1) % 4;
          uint8_t v1 = 0;
          if (r + 1 < rows) {
            v1 = static_cast<uint8_t>(v);
            v = (v + 1) % 4;
            if (v == 3) {
              v = (v + 1) % 4;
            }
          }
          uint8_t v2 = 0;
          if (r + 2 < rows) {
            v2 = static_cast<uint8_t>(v);
            v = (v + 1) % 4;
            if (v == 3) {
              v = (v + 1) % 4;
            }
          }
          uint8_t v3 = 0;
          if (r + 3 < rows) {
            v3 = static_cast<uint8_t>(v);
            v = (v + 1) % 4;
            if (v == 3) {
              v = (v + 1) % 4;
            }
          }
          elements[idx] = (v3 << 6) | (v2 << 4) | (v1 << 2) | v0;
        }
      }
    } else if constexpr(qbits == 4) {
      v = 7;
      for (int c = 0; c < columns; c++) {
        for (int r = 0; r < rows; r += 2) {
          int idx = c * q_rows + r / 2;
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

          elements[idx] = (v1 << 4) | v0;
        }
      }
    }

    float* scales = InputScales.GetBuffer(q_scale_size);
    float* qdq_scales = QDQOutputScales.GetBuffer(scale_size);
    float* qdq_scales_T = QDQTransposedOutputScales.GetBuffer(q_scale_size);
    uint8_t* zp = symmetric ? nullptr : InputOffsets.GetBuffer(q_zp_size_in_bytes, true);
    uint8_t* qdq_zp = symmetric ? nullptr : QDQOutputOffsets.GetBuffer(zp_size, true);
    uint8_t* qdq_zp_T = symmetric ? nullptr : QDQTransposedOutputOffsets.GetBuffer(q_zp_size_in_bytes, true);
    if (zp) {
      if constexpr (qbits == 2) {
        for (int c = 0; c < meta_cols; c++) {
          for (int r = 0; r < meta_rows; r += pack_size) {
            int idx = c * ((meta_rows + 3) / pack_size) + r / pack_size;
            uint8_t v0 = static_cast<uint8_t>(v);
            v = (v + 1) % 4;
            uint8_t v1 = 0;
            if (r + 1 < meta_rows) {
              v1 = static_cast<uint8_t>(v);
              v = (v + 1) % 4;
            }
            uint8_t v2 = 0;
            if (r + 2 < meta_rows) {
              v2 = static_cast<uint8_t>(v);
              v = (v + 1) % 4;
            }
            uint8_t v3 = 0;
            if (r + 3 < meta_rows) {
              v3 = static_cast<uint8_t>(v);
              v = (v + 1) % 4;
            }
            zp[idx] = (v3 << 6) | (v2 << 4) | (v1 << 2) | v0;
          }
        }
      }
      else if constexpr (qbits == 4) {
        for (int c = 0; c < meta_cols; c++) {
          for (int r = 0; r < meta_rows; r += 2) {
            int idx = c * ((meta_rows + 1) / 2) + r / 2;
            uint8_t v0 = static_cast<uint8_t>(v);
            v = (v + 5) % 16;
            if (v == 11 || v == 7 || v == 3) {
              // making the cycle 13 instead of 16, avoiding same values in a row
              v = (v + 5) % 16;
            }
            uint8_t v1 = 0;
            if (r + 1 < meta_rows) {
              v1 = static_cast<uint8_t>(v);
              v = (v + 5) % 16;
              if (v == 11 || v == 7 || v == 3) {
                // making the cycle 13 instead of 16, avoiding same values in a row
                v = (v + 5) % 16;
              }
            }
            zp[idx] = (v1 << 4) | v0;
          }
        }
      }
    }

    MlasDequantizeBlockwise<float, qbits>(dequant_buf, elements, scales, zp, block_size,
                                      columnwise, rows, columns, threadpool_ptr);

    MlasTranspose(dequant_buf, transposed, columns, rows);

    uint8_t* o_elements = OutputElements.GetBuffer(q_rows * q_cols, true);
    float* o_scales = OutputScales.GetBuffer(meta_rows * meta_cols);
    uint8_t* o_zp = symmetric ? nullptr : OutputOffsets.GetBuffer(((meta_rows + 1) / 2) * meta_cols, true);

    MlasQuantizeBlockwise<float, qbits>(o_elements, o_scales, o_zp, transposed, block_size,
                                    columnwise, rows, columns, columns, threadpool_ptr);

    if constexpr (qbits == 4) {
      if (columnwise) {
        bool signed_quant = MlasQDQQuantizeBlockwise<float, qbits>(
            transposed, qdq_scales, qdq_zp, qdq_weights,
            true, rows, columns, block_size, threadpool_ptr);

        ASSERT_EQ(symmetric, signed_quant) << "symmetric quantization should be signed";

        if (symmetric) {
          MlasQDQTransposeBlockwiseQuantized<float, qbits, true>(
              qdq_weights, qdq_scales, qdq_zp, qdq_weights_T, qdq_scales_T, qdq_zp_T,
              true, rows, columns, block_size, threadpool_ptr);

        } else {
          MlasQDQTransposeBlockwiseQuantized<float, qbits, false>(
              qdq_weights, qdq_scales, qdq_zp, qdq_weights_T, qdq_scales_T, qdq_zp_T,
              true, rows, columns, block_size, threadpool_ptr);
        }
      }
    }

    if constexpr (qbits == 2) {
      for (int c = 0; c < columns; c++) {
        for (int r = 0; r < rows; r += pack_size) {
          int idx = c * q_rows + r / pack_size;
          ASSERT_EQ(o_elements[idx] & 0x3, elements[idx] & 0x3)
              << ", index=[" << r << "x" << c << "], shape=[" << rows << "x" << columns
              << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          if (r + 1 < rows) {
            ASSERT_EQ((o_elements[idx] >> 2) & 0x3, (elements[idx] >> 2) & 0x3)
                << ", index=[" << r + 1 << "x" << c << "], shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          }
          if (r + 2 < rows) {
            ASSERT_EQ((o_elements[idx] >> 4) & 0x3, (elements[idx] >> 4) & 0x3)
                << ", index=[" << r + 2 << "x" << c << "], shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          }
          if (r + 3 < rows) {
            ASSERT_EQ((o_elements[idx] >> 6) & 0x3, (elements[idx] >> 6) & 0x3)
                << ", index=[" << r + 3 << "x" << c << "], shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          }
        }
      }
    } else if constexpr (qbits == 4) {
      for (int c = 0; c < columns; c++) {
        for (int r = 0; r < rows; r += 2) {
          int idx = c * q_rows + r / 2;
          ASSERT_EQ(o_elements[idx] & 0xf, elements[idx] & 0xf)
              << ", index=[" << r << "x" << c << "], shape=[" << rows << "x" << columns
              << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          if constexpr (qbits == 4) {
            if (columnwise) {
              ASSERT_EQ(qdq_weights_T[idx] & 0xf, elements[idx] & 0xf)
                  << ", index=[" << r << "x" << c << "], shape=[" << rows << "x" << columns
                  << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
            }
          }
          if (r + 1 < rows) {
            ASSERT_EQ(o_elements[idx] >> 4, elements[idx] >> 4)
                << ", index=[" << r + 1 << "x" << c << "], shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
            if constexpr (qbits == 4) {
              if (columnwise) {
                ASSERT_EQ(qdq_weights_T[idx] >> 4, elements[idx] >> 4)
                    << ", index=[" << r + 1 << "x" << c << "], shape=[" << rows << "x" << columns
                    << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
              }
            }
          }
        }
      }
    }

    for (int c = 0; c < meta_cols; c++) {
      for (int r = 0; r < meta_rows; r++) {
        int idx = c * meta_rows + r;
        ASSERT_EQ(o_scales[idx], scales[idx])
            << ", index=" << r << "x" << c << ", shape=[" << rows << "x" << columns
            << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;

        if constexpr (qbits == 4) {
          if (columnwise) {
            ASSERT_EQ(qdq_scales_T[idx], scales[idx])
                << ", index=" << r << "x" << c << ", shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          }
        }
      }
    }

    if (symmetric) return;

    if constexpr (qbits == 2) {
      for (int c = 0; c < meta_cols; c++) {
        for (int r = 0; r < meta_rows; r += pack_size) {
          int idx = c * ((meta_rows + 3) / pack_size) + r / pack_size;
          ASSERT_EQ(o_zp[idx] & 0x3, zp[idx] & 0x3)
              << ", index=" << r << "x" << c << ", shape=[" << rows << "x" << columns
              << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          if (r + 1 < meta_rows) {
            ASSERT_EQ((o_zp[idx] >> 2) & 0x3, (zp[idx] >> 2) & 0x3)
                << ", index=" << r + 1 << "x" << c << ", shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          }
          if (r + 2 < meta_rows) {
            ASSERT_EQ((o_zp[idx] >> 4) & 0x3, (zp[idx] >> 4) & 0x3)
                << ", index=" << r + 2 << "x" << c << ", shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          }
          if (r + 3 < meta_rows) {
            ASSERT_EQ((o_zp[idx] >> 6) & 0x3, (zp[idx] >> 6) & 0x3)
                << ", index=" << r + 3 << "x" << c << ", shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          }
        }
      }
    } else if constexpr (qbits == 4) {
      for (int c = 0; c < meta_cols; c++) {
        for (int r = 0; r < meta_rows; r += 2) {
          int idx = c * ((meta_rows + 1) / 2) + r / 2;
          ASSERT_EQ(o_zp[idx] & 0xf, zp[idx] & 0xf)
              << ", index=" << r << "x" << c << ", shape=[" << rows << "x" << columns
              << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          if (columnwise) {
            ASSERT_EQ(qdq_zp_T[idx] & 0xf, zp[idx] & 0xf)
                << ", index=" << r << "x" << c << ", shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          }
          if (r + 1 < meta_rows) {
            ASSERT_EQ(o_zp[idx] >> 4, zp[idx] >> 4)
                << ", index=" << r + 1 << "x" << c << ", shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
            if (columnwise) {
              ASSERT_EQ(qdq_zp_T[idx] >> 4, zp[idx] >> 4)
                  << ", index=" << r + 1 << "x" << c << ", shape=[" << rows << "x" << columns
                  << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
            }
          }
        }
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("BlockQ4");
    return suite_name.c_str();
  }

  void ExecuteShort(void) {
    // only support columnwise = true with qbits=2
    Test<Q2Bits>(20, 1, 32, true, false);
    Test<Q2Bits>(20, 1, 32, true, true);
    //Test<Q2Bits>(1, 20, 32, false, false);
    //Test<Q2Bits>(1, 20, 32, false, true);
    Test<Q2Bits>(52, 1, 32, true, false);
    Test<Q2Bits>(52, 1, 32, true, true);
    //Test<Q2Bits>(1, 52, 32, false, false);
    //Test<Q2Bits>(1, 52, 32, false, true);
    Test<Q2Bits>(20, 3, 32, true, false);
    Test<Q2Bits>(20, 3, 32, true, true);
    //Test<Q2Bits>(3, 20, 32, false, false);
    //Test<Q2Bits>(3, 20, 32, false, true);
    Test<Q2Bits>(52, 3, 32, true, false);
    Test<Q2Bits>(52, 3, 32, true, true);
    //Test<Q2Bits>(3, 52, 32, false, false);
    //Test<Q2Bits>(3, 52, 32, false, true);
    Test<Q2Bits>(52, 3, 64, true, false);
    Test<Q2Bits>(52, 3, 64, true, true);
    //Test<Q2Bits>(3, 52, 64, false, false);
    //Test<Q2Bits>(3, 52, 64, false, true);
    Test<Q2Bits>(32 * 9 + 17, 41, 32, true, false);
    Test<Q2Bits>(32 * 9 + 17, 41, 32, true, true);
    //Test<Q2Bits>(41, 32 * 9 + 17, 32, false, false);
    //Test<Q2Bits>(41, 32 * 9 + 17, 32, false, true);
    Test<Q2Bits>(32 * 9 + 17, 41, 64, true, false);
    Test<Q2Bits>(32 * 9 + 17, 41, 64, true, true);
    //Test<Q2Bits>(41, 32 * 9 + 17, 64, false, false);
    //Test<Q2Bits>(41, 32 * 9 + 17, 64, false, true);
    Test<Q2Bits>(32 * 15 + 17, 63, 128, true, false);
    Test<Q2Bits>(32 * 15 + 17, 63, 128, true, true);
    //Test<Q2Bits>(63, 32 * 15 + 17, 128, false, false);
    //Test<Q2Bits>(63, 32 * 15 + 17, 128, false, true);

    Test<Q4Bits>(20, 1, 32, true, false);
    Test<Q4Bits>(20, 1, 32, true, true);
    Test<Q4Bits>(1, 20, 32, false, false);
    Test<Q4Bits>(1, 20, 32, false, true);
    Test<Q4Bits>(52, 1, 32, true, false);
    Test<Q4Bits>(52, 1, 32, true, true);
    Test<Q4Bits>(1, 52, 32, false, false);
    Test<Q4Bits>(1, 52, 32, false, true);
    Test<Q4Bits>(20, 3, 32, true, false);
    Test<Q4Bits>(20, 3, 32, true, true);
    Test<Q4Bits>(3, 20, 32, false, false);
    Test<Q4Bits>(3, 20, 32, false, true);
    Test<Q4Bits>(52, 3, 32, true, false);
    Test<Q4Bits>(52, 3, 32, true, true);
    Test<Q4Bits>(3, 52, 32, false, false);
    Test<Q4Bits>(3, 52, 32, false, true);
    Test<Q4Bits>(52, 3, 64, true, false);
    Test<Q4Bits>(52, 3, 64, true, true);
    Test<Q4Bits>(3, 52, 64, false, false);
    Test<Q4Bits>(3, 52, 64, false, true);
    Test<Q4Bits>(32 * 9 + 17, 41, 32, true, false);
    Test<Q4Bits>(32 * 9 + 17, 41, 32, true, true);
    Test<Q4Bits>(41, 32 * 9 + 17, 32, false, false);
    Test<Q4Bits>(41, 32 * 9 + 17, 32, false, true);
    Test<Q4Bits>(32 * 9 + 17, 41, 64, true, false);
    Test<Q4Bits>(32 * 9 + 17, 41, 64, true, true);
    Test<Q4Bits>(41, 32 * 9 + 17, 64, false, false);
    Test<Q4Bits>(41, 32 * 9 + 17, 64, false, true);
    Test<Q4Bits>(32 * 15 + 17, 63, 128, true, false);
    Test<Q4Bits>(32 * 15 + 17, 63, 128, true, true);
    Test<Q4Bits>(63, 32 * 15 + 17, 128, false, false);
    Test<Q4Bits>(63, 32 * 15 + 17, 128, false, true);

    Test<Q4Bits>(256, 256, 32, true, false);
    Test<Q4Bits>(256, 256, 32, true, true);
    Test<Q4Bits>(256, 256, 32, false, false);
    Test<Q4Bits>(256, 256, 32, false, true);
  }

  MlasBlockwiseQdqTest() = default;
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasBlockwiseQdqTest>::RegisterShortExecute();
  }
  return count;
});

#endif  // ORT_MINIMAL_BUILD
