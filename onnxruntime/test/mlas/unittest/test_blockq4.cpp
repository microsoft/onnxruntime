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
  std::mt19937 gen{192373};
  std::uniform_int_distribution<int> dist_int{0, (1 << qbits) - 1};
  std::uniform_real_distribution<float> dist_float{-1.f, 1.f};
  MatrixGuardBuffer<T> FpBuf;
  MatrixGuardBuffer<T> FpBuf2;
  MatrixGuardBuffer<T> FpBuf3;
  MatrixGuardBuffer<uint8_t> InputElements;
  MatrixGuardBuffer<T> InputScales;
  MatrixGuardBuffer<uint8_t> InputOffsets;
  MatrixGuardBuffer<uint8_t> QDQOutputElements;
  MatrixGuardBuffer<T> QDQOutputScales;
  MatrixGuardBuffer<uint8_t> QDQOutputOffsets;
  MatrixGuardBuffer<uint8_t> QDQTransposedOutputElements;
  MatrixGuardBuffer<T> QDQTransposedOutputScales;
  MatrixGuardBuffer<uint8_t> QDQTransposedOutputOffsets;
  constexpr static float err_ = qbits == 8 ? 1e-2f : qbits == 4 ? 6e-2f
                                                                : 2e-1f;
  constexpr static float rel_ = qbits == 8 ? 5e-2f : qbits == 4 ? 2e-1f
                                                                : 5e-1f;

  bool FloatEqual(T a, T b, float err = err_, float rel = rel_) {
    float va = static_cast<float>(a);
    float vb = static_cast<float>(b);
    return std::abs(va - vb) < err + std::abs(va) * rel;
  }

  void Test(int rows, int columns, int block_size, bool columnwise, bool symmetric) {
    constexpr int packSize = 8 / qbits;
    T* input = FpBuf.GetFilledBuffer(rows * columns, [this](T* start, size_t size) {
      for (size_t i = 0; i < size; i++) {
        start[i] = T(this->dist_float(this->gen));
      }
    });
    T* dequant = FpBuf2.GetBuffer(rows * columns, true);
    T* transposed = FpBuf3.GetBuffer(rows * columns, true);
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

    uint8_t* elements = InputElements.GetBuffer(q_data_size_in_bytes, true);  // after quantize
    uint8_t* qdq_weights;
    uint8_t* qdq_weights_T;
    if constexpr (qbits == 4) {
      qdq_weights = QDQOutputElements.GetBuffer((rows * columns + packSize - 1) / packSize, true);
      qdq_weights_T = QDQTransposedOutputElements.GetBuffer(q_data_size_in_bytes, true);
    }

    T* scales = InputScales.GetBuffer(q_scale_size, true);
    uint8_t* zp = symmetric ? nullptr : InputOffsets.GetBuffer(q_zp_size_in_bytes, true);
    T* qdq_scales;
    T* qdq_scales_T;
    uint8_t* qdq_zp;
    uint8_t* qdq_zp_T;
    if constexpr (qbits == 4) {
      qdq_scales = QDQOutputScales.GetBuffer(scale_size, true);
      qdq_scales_T = QDQTransposedOutputScales.GetBuffer(q_scale_size, true);
      qdq_zp = symmetric ? nullptr : QDQOutputOffsets.GetBuffer(zp_size, true);
      qdq_zp_T = symmetric ? nullptr : QDQTransposedOutputOffsets.GetBuffer(q_zp_size_in_bytes, true);
    }

    MlasQuantizeBlockwise<T, qbits>(elements, scales, zp, input, block_size,
                                    columnwise, rows, columns, columns, threadpool_ptr);

    MlasDequantizeBlockwise<T, qbits>(dequant, elements, scales, zp, block_size,
                                      columnwise, rows, columns, threadpool_ptr);

    MlasTranspose(dequant, transposed, columns, rows, threadpool_ptr);

    if constexpr (qbits == 4) {
      if (columnwise) {
        bool signed_quant = MlasQDQQuantizeBlockwise<T, qbits>(
            input, qdq_scales, qdq_zp, qdq_weights,
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

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < columns; c++) {
        int idx = r * columns + c;
        ASSERT_TRUE(FloatEqual(input[idx], transposed[idx]))
            << " input: " << input[idx] << ", transposed: " << transposed[idx]
            << ", index=[" << r << "x" << c << "], shape=[" << rows << "x" << columns
            << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
      }
    }

    if (columnwise && qbits == 4) {
      for (int c = 0; c < columns; c++) {
        for (int r = 0; r < rows; r += packSize) {
          int idx = c * q_rows + r / packSize;
          for (int l = 0; l < packSize && l + r < rows; ++l) {
            ASSERT_EQ(GetElem<qbits>(qdq_weights_T[idx], l), GetElem<qbits>(elements[idx], l))
                << ", qdq index=[" << r + l << "x" << c << "], shape=[" << rows << "x" << columns
                << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
          }
        }
      }

      for (int c = 0; c < meta_cols; c++) {
        for (int r = 0; r < meta_rows; r++) {
          int idx = c * meta_rows + r;
          ASSERT_TRUE(FloatEqual(qdq_scales_T[idx], scales[idx]))
              << ", qdq index=" << r << "x" << c << ", shape=[" << rows << "x" << columns
              << "] block: " << block_size << ", symmetric: " << symmetric << ", columnwise: " << columnwise;
        }
      }

      if (symmetric) return;
      for (int c = 0; c < meta_cols; c++) {
        for (int r = 0; r < meta_rows; r += packSize) {
          int idx = c * ((meta_rows + packSize - 1) / packSize) + r / packSize;
          for (int l = 0; l < packSize && r + l < meta_rows; ++l) {
            ASSERT_EQ(GetElem<qbits>(qdq_zp_T[idx], l), GetElem<qbits>(zp[idx], l))
                << ", qdq index=" << r + l << "x" << c << ", shape=[" << rows << "x" << columns
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
