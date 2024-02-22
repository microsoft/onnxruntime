// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

class MlasScaleOutputTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<int32_t> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputRef;
  MatrixGuardBuffer<float> BufferScale;

  void Test(size_t M, size_t N, bool PerColumn, bool AccumulateMode) {
    int32_t* Input = BufferInput.GetBuffer(M * N);
    float* Output = BufferOutput.GetBuffer(M * N);
    float* OutputRef = BufferOutputRef.GetBuffer(M * N);
    float* Scale = BufferScale.GetBuffer(PerColumn ? N : 1);

    std::default_random_engine generator(static_cast<unsigned>(M * N));
    std::uniform_real_distribution<float> real_distribution(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> int_distribution(std::numeric_limits<int16_t>::min(),
                                                            std::numeric_limits<int16_t>::max());

    for (size_t s = 0; s < M * N; s++) {
      Input[s] = int_distribution(generator);
      Output[s] = OutputRef[s] = real_distribution(generator);
    }

    for (size_t s = 0; s < (PerColumn ? N : 1); s++) {
      Scale[s] = real_distribution(generator);
    }

    // Compute Reference Value
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        float current_scale = PerColumn ? Scale[n] : Scale[0];
        if (AccumulateMode) {
          OutputRef[m * N + n] += Input[m * N + n] * current_scale;
        } else {
          OutputRef[m * N + n] = Input[m * N + n] * current_scale;
        }
      }
    }

    // Compute Output with MLAS
    MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR OutputProcessor(
        Output, N, Scale, nullptr,
        AccumulateMode ? MLAS_QGEMM_OUTPUT_MODE::AccumulateMode : MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
        PerColumn ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);
    OutputProcessor.Process(Input, 0, 0, M, N, N);

    constexpr float epsilon = 1e-6f;

    for (size_t n = 0; n < M * N; n++) {
      float diff = std::fabs((Output[n] - OutputRef[n]) / OutputRef[n]);
      ASSERT_LE(diff, epsilon)
          << " @[" << n / N << "," << n % N << "], total:[" << M << "," << N << "], got:"
          << Output[n] << ", expecting:" << OutputRef[n];
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("ScaleOutput");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t m = 1; m < 18; m++) {
      for (size_t n = 1; n < 18; n++) {
        Test(m, n, true, true);
        Test(m, n, true, false);
        Test(m, n, false, true);
        Test(m, n, false, false);
      }
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? MlasDirectShortExecuteTests<MlasScaleOutputTest>::RegisterShortExecute() : 0;
});
