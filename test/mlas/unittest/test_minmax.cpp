// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

class MlasFindMinMaxElementsTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;

  void Test(size_t N, float MinimumValue, float MaximumValue) {
    float* Input = BufferInput.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t n = 0; n < N; n++) {
      Input[n] = distribution(generator);
    }

    auto min_max_pair = std::minmax_element(Input, Input + N);
    float min_ref = *min_max_pair.first;
    float max_ref = *min_max_pair.second;

    float min, max;
    MlasFindMinMaxElement(Input, &min, &max, N);

    constexpr float epsilon = 1e-6f;

    float diff_min = std::fabs(min - min_ref);
    ASSERT_LE(diff_min, epsilon) << " for minimum with parameter (" << N << "," << MinimumValue << "," << MaximumValue << ")";

    float diff_max = std::fabs(max - max_ref);
    ASSERT_LE(diff_max, epsilon) << " for maximum with parameter (" << N << "," << MinimumValue << "," << MaximumValue << ")";
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("MinMaxElement");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t n = 1; n < 128; n++) {
      Test(n, -10.f, 10.f);
    }
  }
};

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? MlasDirectShortExecuteTests<MlasFindMinMaxElementsTest>::RegisterShortExecute() : 0;
});
