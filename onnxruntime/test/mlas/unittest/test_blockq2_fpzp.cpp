/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_blockq2_fpzp.cpp

Abstract:

    Tests for MLAS blockwise 2-bit dequantization with floating point zero
    points (MlasDequantizeBlockwiseFpZeroPoint).

--*/

#ifndef ORT_MINIMAL_BUILD

#include "test_util.h"
#include "mlas_q4.h"
#include "core/common/float16.h"

#include <algorithm>
#include <random>

class MlasBlockwise2BitsFpZpTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<uint8_t> InputElements;
  MatrixGuardBuffer<float> InputScales;
  MatrixGuardBuffer<float> InputZeroPointsFp32;
  MatrixGuardBuffer<MLAS_FP16> InputZeroPointsFp16;
  MatrixGuardBuffer<float> OutputElements;
  MatrixGuardBuffer<float> ReferenceElements;

  enum class ZeroPointMode { None,
                             Fp32,
                             Fp16 };

  // A value no valid output can produce, to catch elements the kernel never writes.
  static constexpr float UnwrittenSentinel = -888.0f;

  void Test(int rows, int columns, int block_size, ZeroPointMode zp_mode) {
    const int k_blocks = (rows + block_size - 1) / block_size;
    const size_t q_bytes = static_cast<size_t>(columns) * k_blocks * block_size / 4;
    const size_t scale_count = static_cast<size_t>(columns) * k_blocks;
    const size_t out_count = static_cast<size_t>(columns) * rows;

    uint8_t* q_data = InputElements.GetBuffer(q_bytes, true);
    float* scales = InputScales.GetBuffer(scale_count, true);
    float* zp_fp32 = nullptr;
    MLAS_FP16* zp_fp16 = nullptr;
    float* dst = OutputElements.GetBuffer(out_count, true);
    std::fill_n(dst, out_count, UnwrittenSentinel);
    float* ref = ReferenceElements.GetBuffer(out_count, true);

    std::mt19937 gen(1337 + rows * 7 + columns * 3 + block_size);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);
    std::uniform_real_distribution<float> zp_dist(0.0f, 3.0f);

    for (size_t i = 0; i < q_bytes; i++) {
      q_data[i] = static_cast<uint8_t>(byte_dist(gen));
    }
    for (size_t i = 0; i < scale_count; i++) {
      scales[i] = scale_dist(gen);
    }
    if (zp_mode == ZeroPointMode::Fp32) {
      zp_fp32 = InputZeroPointsFp32.GetBuffer(scale_count, true);
      for (size_t i = 0; i < scale_count; i++) {
        zp_fp32[i] = zp_dist(gen);
      }
    } else if (zp_mode == ZeroPointMode::Fp16) {
      zp_fp16 = InputZeroPointsFp16.GetBuffer(scale_count, true);
      for (size_t i = 0; i < scale_count; i++) {
        zp_fp16[i] = MLAS_FP16(zp_dist(gen));
      }
    }

    for (int n = 0; n < columns; n++) {
      for (int kb = 0; kb < k_blocks; kb++) {
        const size_t block_idx = static_cast<size_t>(n) * k_blocks + kb;
        const float scale = scales[block_idx];
        float zp = 0.0f;
        if (zp_mode == ZeroPointMode::Fp32) {
          zp = zp_fp32[block_idx];
        } else if (zp_mode == ZeroPointMode::Fp16) {
          zp = zp_fp16[block_idx].ToFloat();
        }
        const float zp_adjust = -scale * zp;
        const int k_end = std::min(rows, (kb + 1) * block_size);
        for (int k = kb * block_size; k < k_end; k++) {
          const size_t element_offset = block_idx * block_size + (k - kb * block_size);
          const int q = (q_data[element_offset >> 2] >> (2 * (element_offset & 3))) & 0x3;
          ref[static_cast<size_t>(n) * rows + k] = static_cast<float>(q) * scale + zp_adjust;
        }
      }
    }

    if (zp_mode == ZeroPointMode::Fp16) {
      MlasDequantizeBlockwiseFpZeroPoint<float, MLAS_FP16, 2>(
          dst, q_data, scales, zp_fp16, block_size, /*columnwise=*/true, rows, columns, GetMlasThreadPool());
    } else {
      MlasDequantizeBlockwiseFpZeroPoint<float, float, 2>(
          dst, q_data, scales, zp_fp32, block_size, /*columnwise=*/true, rows, columns, GetMlasThreadPool());
    }

    for (size_t i = 0; i < out_count; i++) {
      ASSERT_EQ(dst[i], ref[i]) << " element " << i << ", rows=" << rows << ", columns=" << columns
                                << ", block_size=" << block_size << ", zp_mode=" << static_cast<int>(zp_mode);
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Q2BlockwiseFpZpDequant");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    Test(32, 1, 16, ZeroPointMode::None);
    Test(32, 1, 16, ZeroPointMode::Fp32);
    Test(32, 1, 16, ZeroPointMode::Fp16);
    Test(17, 2, 16, ZeroPointMode::Fp32);
    Test(100, 3, 32, ZeroPointMode::None);
    Test(100, 3, 32, ZeroPointMode::Fp32);
    Test(100, 3, 32, ZeroPointMode::Fp16);
    Test(1000, 5, 128, ZeroPointMode::Fp32);
    Test(1000, 5, 128, ZeroPointMode::Fp16);
    Test(4096, 8, 64, ZeroPointMode::Fp32);
    Test(300, 7, 256, ZeroPointMode::None);
    Test(300, 7, 256, ZeroPointMode::Fp32);
    Test(300, 7, 256, ZeroPointMode::Fp16);
    Test(2048, 31, 32, ZeroPointMode::Fp32);
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasBlockwise2BitsFpZpTest>::RegisterShortExecute();
  }
  return count;
});

#endif  // ORT_MINIMAL_BUILD
