/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_rope_neon_fp16.cpp

Abstract:

    Tests for MLAS fp16 RoPE on NEON.

--*/

#include <vector>
#include <cmath>

#include "core/mlas/inc/mlas.h"

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/rotary_embedding.h"
#include "core/mlas/lib/rotary_embedding_kernel_neon.h"

class MlasNeonFp16RoPETest : public MlasTestBase {
 private:
  const float Pi = 2 * std::acos(0.0f);

  void Test(size_t rotary_emb_dim, bool interleaved) {
    // Per kernel logic (both fallback and optimized), the sin/cos tables
    // are always half the rotary embedding dimension.
    const size_t table_len = rotary_emb_dim / 2;

    std::vector<MLAS_FP16> input(rotary_emb_dim);
    std::vector<MLAS_FP16> sin_data(table_len);
    std::vector<MLAS_FP16> cos_data(table_len);
    std::vector<MLAS_FP16> output_ref(rotary_emb_dim);
    std::vector<MLAS_FP16> output_impl(rotary_emb_dim);

    // Initialize input data
    for (size_t i = 0; i < rotary_emb_dim; ++i) {
      input[i] = MLAS_FP16(static_cast<float>(i + 1));
    }

    // Initialize sin/cos tables
    for (size_t i = 0; i < table_len; ++i) {
      float theta = static_cast<float>(i) / 1000.0f * Pi;
      sin_data[i] = MLAS_FP16(std::sin(theta));
      cos_data[i] = MLAS_FP16(std::cos(theta));
    }

    // Call fallback implementation
    MlasRotaryEmbedOneRow_FallBack<MLAS_FP16>(input.data(), sin_data.data(), cos_data.data(), rotary_emb_dim, interleaved, output_ref.data());

    // Call dispatched implementation (which should pick up the NEON kernel)
    MlasRotaryEmbedOneRow<MLAS_FP16>(input.data(), sin_data.data(), cos_data.data(), rotary_emb_dim, interleaved, output_impl.data());

    // Compare results
    for (size_t i = 0; i < rotary_emb_dim; i++) {
      ASSERT_TRUE(CloseEnough(output_impl[i].ToFloat(), output_ref[i].ToFloat()))
          << "Expected bits: " << output_ref[i].val << " (" << output_ref[i].ToFloat() << ")"
          << " Actual bits: " << output_impl[i].val << " (" << output_impl[i].ToFloat() << ")"
          << " @[" << i << "], "
          << "rotary_emb_dim=" << rotary_emb_dim << ", interleaved=" << interleaved;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    return "NeonFp16RoPE";
  }

  void ExecuteShort(void) override {
    // Test dimensions that cover main loops and various remainders
    Test(6, false);
    Test(6, true);
    Test(16, false);
    Test(16, true);
    Test(24, false);
    Test(24, true);
    Test(32, false);
    Test(32, true);
    Test(42, false);
    Test(42, true);
    Test(64, false);
    Test(64, true);
    Test(70, false);
    Test(70, true);
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasNeonFp16RoPETest>::RegisterShortExecute();
  }
  return count;
});

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
