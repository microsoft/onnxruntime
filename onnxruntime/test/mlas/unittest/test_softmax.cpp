// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

template <bool Threaded>
class MlasSoftmaxTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;
  MLAS_THREADPOOL* threadpool_;

  void Test(size_t N, size_t D, float MinimumValue, float MaximumValue) {
    float* Input = BufferInput.GetBuffer(N * D);
    float* Output = BufferOutput.GetBuffer(N * D);
    float* OutputReference = BufferOutputReference.GetBuffer(N * D);

    std::default_random_engine generator(static_cast<unsigned>(N * D));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t nd = 0; nd < N * D; nd++) {
      Input[nd] = distribution(generator);
    }

    Test(Input, Output, OutputReference, N, D, false);
    Test(Input, Output, OutputReference, N, D, true);
  }

  void Test(const float* Input, float* Output, float* OutputReference, size_t N, size_t D, bool LogSoftmax) {
    MlasComputeSoftmax(Input, Output, N, D, LogSoftmax, threadpool_);
    ReferenceSoftmax(Input, OutputReference, N, D, LogSoftmax);

    constexpr float AbsoluteTolerance = 1e-6f;
    constexpr float RelativeTolerance = 1e-6f;

    for (size_t nd = 0; nd < N * D; nd++) {
      float diff = std::fabs(Output[nd] - OutputReference[nd]);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(OutputReference[nd]) * RelativeTolerance)
          << "LogSoftmax:" << (int)LogSoftmax << " difference " << N << "/" << D
          << ", got: " << Output[nd] << ", expecting: " << OutputReference[nd];
    }
  }

  void ReferenceSoftmax(const float* Input, float* Output, size_t N, size_t D, bool LogSoftmax) {
    for (size_t n = 0; n < N; n++) {
      float MaximumValue = std::numeric_limits<float>::lowest();

      for (size_t d = 0; d < D; d++) {
        MaximumValue = (std::max)(MaximumValue, Input[d]);
      }

      double Sum = 0.0;

      for (size_t d = 0; d < D; d++) {
        double e = std::exp(double(Input[d]) - double(MaximumValue));
        Sum += e;
        Output[d] = float(e);
      }

      if (LogSoftmax) {
        float Scale = float(std::log(Sum));

        for (size_t d = 0; d < D; d++) {
          Output[d] = Input[d] - MaximumValue - Scale;
        }

      } else {
        float Scale = float(Sum);

        for (size_t d = 0; d < D; d++) {
          Output[d] /= Scale;
        }
      }

      Input += D;
      Output += D;
    }
  }

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name(Threaded ? "Softmax_Threaded" : "Softmax_SingleThread");
    return suite_name.c_str();
  }

  MlasSoftmaxTest() : threadpool_(Threaded ? GetMlasThreadPool() : nullptr) {}

  void ExecuteShort(void) override {
    for (size_t d = 1; d < 128; d++) {
      Test(1, d, -10.f, 10.f);
    }

    Test(3, 128, 20.f, 30.f);
    Test(63, 95, -150.f, 190.f);
    Test(16, 211, 20.f, 30.f);
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasSoftmaxTest<false>>::RegisterShortExecute();
    if (GetMlasThreadPool() != nullptr) {
      count += MlasDirectShortExecuteTests<MlasSoftmaxTest<true>>::RegisterShortExecute();
    }
  }
  return count;
});
