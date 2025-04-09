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
#include "core/mlas/lib/mlasi.h"

template <int qbits>
int GetElem(int v, int idx) {
  return (v >> (qbits * idx)) & ((1 << qbits) - 1);
}

template <int qbits>
int SetElem(int v, int idx, int value) {
  v &= ~(((1 << qbits) - 1) << (qbits * idx));
  v |= (value & ((1 << qbits) - 1)) << (qbits * idx);
  return v;
}

template <typename T, int qbits>
class MlasBlockwiseQdqTest : public MlasTestBase {
 private:
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<int> dist_int{0, (1 << qbits) - 1};
  MatrixGuardBuffer<T> FpBuf;
  MatrixGuardBuffer<T> FpBuf2;
  MatrixGuardBuffer<uint8_t> InputElements;
  MatrixGuardBuffer<T> InputScales;
  MatrixGuardBuffer<uint8_t> InputOffsets;
  MatrixGuardBuffer<uint8_t> OutputElements;
  MatrixGuardBuffer<T> OutputScales;
  MatrixGuardBuffer<uint8_t> OutputOffsets;
  MatrixGuardBuffer<uint8_t> QDQOutputElements;
  MatrixGuardBuffer<T> QDQOutputScales;
  MatrixGuardBuffer<uint8_t> QDQOutputOffsets;
  MatrixGuardBuffer<uint8_t> QDQTransposedOutputElements;
  MatrixGuardBuffer<T> QDQTransposedOutputScales;
  MatrixGuardBuffer<uint8_t> QDQTransposedOutputOffsets;

  void SetValue(uint8_t *arr, int rows, int cols, int q_rows, int block_size, bool columnwise) {
    constexpr int packSize = 8 / qbits;
    for (int c = 0; c < cols; c++) {
      for (int r = 0; r < rows; r += packSize) {
        int idx = c * q_rows + r / packSize;
        for (int l = 0; l < packSize && r + l < rows; ++l) {
          int v;
          if (columnwise) {
            if (r % block_size == 0) {
              v = 0;
            } else if (r % block_size == 1) {
              v = (1 << qbits) - 1;
            } else {
              v = dist_int(gen);
            }
          } else {
            if (c % block_size == 0) {
              v = 0;
            } else if (c % block_size == 1) {
              v = (1 << qbits) - 1;
            } else {
              v = dist_int(gen);
            }
          }
          arr[idx] = (uint8_t)SetElem<qbits>(arr[idx], l, v);
        }
      }
    }
  }

  void Test(int rows, int columns, int block_size, bool columnwise, bool symmetric) {
    constexpr int packSize = 8 / qbits;
    T* dequant_buf = FpBuf.GetBuffer(rows * columns, true);
    T* transposed = FpBuf2.GetBuffer(rows * columns, true);
    size_t scale_size = (rows + block_size - 1) / block_size * columns;
    size_t zp_size = (scale_size + packSize - 1) / packSize;

    MLAS_THREADPOOL* threadpool_ptr = GetMlasThreadPool();

    int meta_rows;
    int meta_cols;
    MlasBlockwiseQuantMetaShape<T, qbits>(block_size, columnwise, rows, columns, meta_rows, meta_cols);

    int q_rows;
    int q_cols;
    MlasBlockwiseQuantizedShape<T, qbits>(block_size, columnwise, rows, columns, q_rows, q_cols);

    size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
    MlasBlockwiseQuantizedBufferSizes<qbits>(block_size, columnwise, rows, columns,
                                      q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

    uint8_t* elements = InputElements.GetBuffer(q_data_size_in_bytes, true);
    uint8_t* qdq_weights;
    uint8_t* qdq_weights_T;
    if constexpr (qbits == 4) {
      qdq_weights = QDQOutputElements.GetBuffer((rows * columns + packSize - 1) / packSize, true);
      qdq_weights_T = QDQTransposedOutputElements.GetBuffer(q_data_size_in_bytes, true);
    }

    SetValue(elements, rows, columns, q_rows, block_size, columnwise);

    T* scales = InputScales.GetBuffer(q_scale_size);
    uint8_t* zp = symmetric ? nullptr : InputOffsets.GetBuffer(q_zp_size_in_bytes, true);
    T* qdq_scales;
    T* qdq_scales_T;
    uint8_t* qdq_zp;
    uint8_t* qdq_zp_T;
    if constexpr (qbits == 4) {
      qdq_scales = QDQOutputScales.GetBuffer(scale_size);
      qdq_scales_T = QDQTransposedOutputScales.GetBuffer(q_scale_size);
      qdq_zp = symmetric ? nullptr : QDQOutputOffsets.GetBuffer(zp_size, true);
      qdq_zp_T = symmetric ? nullptr : QDQTransposedOutputOffsets.GetBuffer(q_zp_size_in_bytes, true);
    }

    if (zp) {
      for (int c = 0; c < meta_cols; c++) {
        for (int r = 0; r < meta_rows; r += packSize) {
          int idx = c * ((meta_rows + packSize - 1) / packSize) + r / packSize;
          for (int l = 0; l < packSize && r + l < meta_rows; ++l) {
           zp[idx] = (uint8_t)SetElem<qbits>(zp[idx], l, dist_int(gen));
          }
        }
      }
    }

    // std::cout << "before dequant: " << (elements[0] & ((1 << qbits) - 1)) << " " << scales[0] << " " << (zp ? (zp[0] & ((1 << qbits) - 1)) : 0) << std::endl;
    MlasDequantizeBlockwise<T, qbits>(dequant_buf, elements, scales, zp, block_size,
                                      columnwise, rows, columns, threadpool_ptr);

    // std::cout << "after dequant: " << dequant_buf[0] << " " << scales[0] << " " << (zp ? (zp[0] & ((1 << qbits) - 1)) : 0) << std::endl;

    MlasTranspose(dequant_buf, transposed, columns, rows, threadpool_ptr);

    uint8_t* o_elements = OutputElements.GetBuffer(q_rows * q_cols, true);
    T* o_scales = OutputScales.GetBuffer(meta_rows * meta_cols);
    uint8_t* o_zp = symmetric
      ? nullptr
      : OutputOffsets.GetBuffer(((meta_rows + packSize - 1) / packSize) * meta_cols, true);

    MlasQuantizeBlockwise<T, qbits>(o_elements, o_scales, o_zp, transposed, block_size,
                                    columnwise, rows, columns, columns, threadpool_ptr);

    // std::cout << "after quant: " << (o_elements[0] & ((1 << qbits) - 1)) << " " << o_scales[0] << " " << (o_zp ? (o_zp[0] & ((1 << qbits) - 1)) : 0) << std::endl;

    if constexpr (qbits == 4) {
      if (columnwise) {
        bool signed_quant = MlasQDQQuantizeBlockwise<T, qbits>(
            transposed, qdq_scales, qdq_zp, qdq_weights,
            true, rows, columns, block_size, threadpool_ptr);

        ASSERT_EQ(symmetric, signed_quant) << "symmetric quantization should be signed";

        if (symmetric) {
          MlasQDQTransposeBlockwiseQuantized<T, qbits, true>(
              qdq_weights, qdq_scales, qdq_zp, qdq_weights_T, qdq_scales_T, qdq_zp_T,
              true, rows, columns, block_size, threadpool_ptr);

        } else {
          MlasQDQTransposeBlockwiseQuantized<T, qbits, false>(
              qdq_weights, qdq_scales, qdq_zp, qdq_weights_T, qdq_scales_T, qdq_zp_T,
              true, rows, columns, block_size, threadpool_ptr);
        }
      }
    }

    for (int c = 0; c < columns; c++) {
      for (int r = 0; r < rows; r += packSize) {
        int idx = c * q_rows + r / packSize;
        for (int l = 0; l < packSize && l + r < rows; ++l) {
          ASSERT_EQ(GetElem<qbits>(o_elements[idx], l), GetElem<qbits>(elements[idx], l))
              << ", index=[" << r+l << "x" << c << "], shape=[" << rows << "x" << columns
              << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          if (columnwise && qbits == 4) {
            ASSERT_EQ(GetElem<qbits>(qdq_weights_T[idx], l), GetElem<qbits>(elements[idx], l))
                << ", index=[" << r+l << "x" << c << "], shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
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

        if (columnwise && qbits == 4) {
          ASSERT_EQ(qdq_scales_T[idx], scales[idx])
              << ", index=" << r << "x" << c << ", shape=[" << rows << "x" << columns
              << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
        }
      }
    }

    if (symmetric) return;
    for (int c = 0; c < meta_cols; c++) {
      for (int r = 0; r < meta_rows; r += packSize) {
        int idx = c * ((meta_rows + packSize - 1) / packSize) + r / packSize;
        for (int l = 0; l < packSize && r+l < meta_rows; ++l) {
          ASSERT_EQ(GetElem<qbits>(o_zp[idx], l), GetElem<qbits>(zp[idx], l))
              << ", index=" << r+l << "x" << c << ", shape=[" << rows << "x" << columns
              << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          if (columnwise && qbits == 4) {
            ASSERT_EQ(GetElem<qbits>(qdq_zp_T[idx], l), GetElem<qbits>(zp[idx], l))
                << ", index=" << r+l << "x" << c << ", shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          }
        }
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("BlockQ" + std::to_string(qbits));
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    Test(20, 1, 32, true, false);
    Test(20, 1, 32, true, true);
    Test(1, 20, 32, false, false);
    Test(1, 20, 32, false, true);
    Test(52, 1, 32, true, false);
    Test(52, 1, 32, true, true);
    Test(1, 52, 32, false, false);
    Test(1, 52, 32, false, true);
    Test(20, 3, 32, true, false);
    Test(20, 3, 32, true, true);
    Test(3, 20, 32, false, false);
    Test(3, 20, 32, false, true);
    Test(52, 3, 32, true, false);
    Test(52, 3, 32, true, true);
    Test(3, 52, 32, false, false);
    Test(3, 52, 32, false, true);
    Test(52, 3, 64, true, false);
    Test(52, 3, 64, true, true);
    Test(3, 52, 64, false, false);
    Test(3, 52, 64, false, true);
    Test(32 * 9 + 17, 41, 32, true, false);
    Test(32 * 9 + 17, 41, 32, true, true);
    Test(41, 32 * 9 + 17, 32, false, false);
    Test(41, 32 * 9 + 17, 32, false, true);
    Test(32 * 9 + 17, 41, 64, true, false);
    Test(32 * 9 + 17, 41, 64, true, true);
    Test(41, 32 * 9 + 17, 64, false, false);
    Test(41, 32 * 9 + 17, 64, false, true);
    Test(32 * 15 + 17, 63, 128, true, false);
    Test(32 * 15 + 17, 63, 128, true, true);
    Test(63, 32 * 15 + 17, 128, false, false);
    Test(63, 32 * 15 + 17, 128, false, true);

    Test(256, 256, 32, true, false);
    Test(256, 256, 32, true, true);
    Test(256, 256, 32, false, false);
    Test(256, 256, 32, false, true);
  }

  MlasBlockwiseQdqTest() = default;
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasBlockwiseQdqTest<float, 2>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasBlockwiseQdqTest<float, 4>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasBlockwiseQdqTest<float, 8>>::RegisterShortExecute();
  }
  return count;
});

#endif  // ORT_MINIMAL_BUILD
