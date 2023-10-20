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

  void Test(int rows, int columns, int block_size, bool columnwise, bool symmetric) {
    float* dequant_buf = FpBuf.GetBuffer(rows * columns, true);
    float* transposed = FpBuf2.GetBuffer(rows * columns, true);

    MLAS_THREADPOOL* threadpool_ptr = GetMlasThreadPool();

    int meta_rows;
    int meta_cols;
    MlasBlockwiseQuantMetaShape<float>(block_size, columnwise, rows, columns, meta_rows, meta_cols);

    uint8_t* elements = InputElements.GetBuffer(((rows + 1) / 2) * columns, true);

    int v = 7;
    for (size_t c = 0; c < columns; c++) {
      for (size_t r = 0; r < rows; r += 2) {
		size_t idx = c * ((rows + 1) / 2) + r / 2;
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

    float* scales = InputScales.GetBuffer(meta_rows * meta_cols);
    uint8_t* zp = symmetric ? nullptr : InputOffsets.GetBuffer(((meta_rows + 1) / 2) * meta_cols, true);
    if (zp) {
      for (size_t c = 0; c < meta_cols; c++) {
        for (size_t r = 0; r < meta_rows; r += 2) {
          size_t idx = c * ((meta_rows + 1) / 2) + r / 2;
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


    MlasDequantizeBlockwise(dequant_buf, elements, scales, zp, block_size, columnwise, rows, columns, threadpool_ptr);

    MlasTranspose(dequant_buf, transposed, columns, rows);

    uint8_t* o_elements = OutputElements.GetBuffer(((rows + 1) / 2) * columns);
    float* o_scales = OutputScales.GetBuffer(meta_rows * meta_cols);
    uint8_t* o_zp = symmetric ? nullptr : OutputOffsets.GetBuffer(((meta_rows + 1) / 2) * meta_cols);


    MlasQuantizeBlockwise(o_elements, o_scales, o_zp, transposed, block_size, columnwise, rows, columns, columns, threadpool_ptr);

    for (size_t c = 0; c < columns; c++) {
      for (size_t r = 0; r < rows; r+=2) {
        size_t idx = c * ((rows + 1) / 2) + r / 2;
        ASSERT_EQ(o_elements[idx], elements[idx]) << ", index=" << idx << ", [" << rows << "x"
                                                  << columns << "] block: " << block_size;
      }
	}

    for (size_t c = 0; c < meta_cols; c++) {
      for (size_t r = 0; r < meta_rows; r++) {
        size_t idx = c * meta_rows + r;
        ASSERT_EQ(o_scales[idx], scales[idx]) << ", index=" << idx << ", [" << rows << "x"
                                              << columns << "] block: " << block_size;
      }
    }

    if (symmetric) return;
    for (size_t c = 0; c < meta_cols; c++) {
        for (size_t r = 0; r < meta_rows; r += 2) {
		  size_t idx = c * ((meta_rows + 1) / 2) + r / 2;
          ASSERT_EQ(o_zp[idx], zp[idx]) << ", index=" << idx << ", [" << rows << "x"
                                        << columns << "] block: " << block_size;
	  }
    }

  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("BlockQ4");
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

template <>
MlasBlockwiseQdqTest* MlasTestFixture<MlasBlockwiseQdqTest>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasBlockwiseQdqTest>::RegisterShortExecute();
  }
  return count;
});

#endif  // ORT_MINIMAL_BUILD
