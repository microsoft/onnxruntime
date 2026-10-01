// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"
#include "core/mlas/lib/softmax.h"

class MlasComputeExpTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;
  MatrixGuardBuffer<MLAS_FP16> BufferInputFp16;
  MatrixGuardBuffer<MLAS_FP16> BufferOutputFp16;

  void Test(size_t N, float MinimumValue, float MaximumValue) {
    float* Input = BufferInput.GetBuffer(N);
    float* Output = BufferOutput.GetBuffer(N);
    float* OutputReference = BufferOutputReference.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t n = 0; n < N; n++) {
      Input[n] = distribution(generator);
    }

    for (size_t n = 0; n < N; n++) {
      OutputReference[n] = std::exp(Input[n]);
    }

    MlasComputeExp(Input, Output, N);

    constexpr float AbsoluteTolerance = 1e-6f;
    constexpr float RelativeTolerance = 1e-6f;

    for (size_t n = 0; n < N; n++) {
      float diff = std::fabs(Output[n] - OutputReference[n]);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(OutputReference[n]) * RelativeTolerance)
          << " @" << n << " of " << N << ", got: " << Output[n] << ", expecting: " << OutputReference[n];
    }
  }

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

  void TestFp16(size_t N, float MinimumValue, float MaximumValue) {
    MLAS_FP16* Input = BufferInputFp16.GetBuffer(N);
    MLAS_FP16* Output = BufferOutputFp16.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t n = 0; n < N; n++) {
      Input[n] = MLAS_FP16(distribution(generator));
    }

    MlasComputeExp(Input, Output, N);

    constexpr float AbsoluteTolerance = 5e-4f;
    constexpr float RelativeTolerance = 1e-3f;

    for (size_t n = 0; n < N; n++) {
      float in = Input[n].ToFloat();
      float ref = std::exp(in);
      float out = Output[n].ToFloat();
      float diff = std::fabs(out - ref);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(ref) * RelativeTolerance)
          << " @ " << in << ", got: " << out << ", expecting: " << ref << ", r-diff: " << diff / std::fabs(ref);
    }
  }

  // Non-finite and boundary inputs, placed at every offset within an 8-wide fp16
  // vector and in the scalar tail. The random cases above never produce NaN, Inf
  // or subnormals, and a lane-position-dependent fault cannot show up in an
  // otherwise benign buffer -- that is how the Gelu/Erf kernel bugs fixed in
  // #32631 survived their own random coverage.
  void TestExpFp16Extreme(size_t N, size_t pos, float value) {
    MLAS_FP16* Input = BufferInputFp16.GetBuffer(N);
    MLAS_FP16* Output = BufferOutputFp16.GetBuffer(N);

    for (size_t n = 0; n < N; n++) {
      Input[n] = MLAS_FP16(0.5f);
    }
    Input[pos] = MLAS_FP16(value);

    MlasComputeExp(Input, Output, N);

    constexpr float AbsoluteTolerance = 5e-4f;
    constexpr float RelativeTolerance = 1e-3f;

    for (size_t n = 0; n < N; n++) {
      float in = Input[n].ToFloat();
      float out = Output[n].ToFloat();
      float ref = std::exp(in);
      if (std::isnan(ref)) {
        ASSERT_TRUE(std::isnan(out)) << " @ " << in << ", got: " << out << ", expecting NaN";
        continue;
      }
      // exp() of a large fp16 input overflows the format; both the kernel and the
      // reference saturate, so compare in fp16 rather than in float.
      float ref16 = MLAS_FP16(ref).ToFloat();
      if (std::isinf(ref16)) {
        ASSERT_TRUE(std::isinf(out) && (out > 0) == (ref16 > 0))
            << " @ " << in << ", got: " << out << ", expecting: " << ref16;
        continue;
      }
      float diff = std::fabs(out - ref16);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(ref16) * RelativeTolerance)
          << " @ " << in << " (lane " << pos << " of " << N << "), got: " << out
          << ", expecting: " << ref16;
    }
  }

  void TestSumFp16(size_t N, float MinimumValue, float MaximumValue) {
    MLAS_FP16* Input = BufferInputFp16.GetBuffer(N);
    MLAS_FP16* Output = BufferOutputFp16.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    float max_val = std::numeric_limits<float>::lowest();
    for (size_t n = 0; n < N; n++) {
      Input[n] = MLAS_FP16(distribution(generator));
      max_val = std::fmax(max_val, Input[n].ToFloat());
    }

    const auto* dispatch = GetMlasPlatform().SoftmaxDispatch;
    auto sum = dispatch->SumExp_Fp16(Input, Output, N, MLAS_FP16(-max_val));

    constexpr float AbsoluteTolerance = 5e-4f;
    constexpr float RelativeTolerance = 1e-3f;

    float sum_ref = 0.0f;
    for (size_t n = 0; n < N; n++) {
      float in = Input[n].ToFloat();
      float ref = std::exp(in - max_val);
      sum_ref += ref;
      float out = Output[n].ToFloat();
      float diff = std::fabs(out - ref);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(ref) * RelativeTolerance)
          << " @ " << in << ", got: " << out << ", expecting: " << ref << ", r-diff: " << diff / std::fabs(ref);
    }

    float diff = std::fabs(sum.ToFloat() - sum_ref);
    ASSERT_TRUE(diff <= 1e-3f || diff <= std::fabs(sum_ref) * 5e-3f)
        << " sum: " << sum.ToFloat() << ", expecting: " << sum_ref << ", r-diff: " << diff / std::fabs(sum_ref);
  }

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Exp");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    for (size_t n = 1; n < 128; n++) {
      Test(n, -10.f, 10.f);
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
      TestFp16(n, -17.f, 11.f);
      TestSumFp16(n, -10.f, 10.f);
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    }

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    // fp16 min-normal, min-subnormal, the exp() overflow boundary (exp(11.09) is
    // the largest representable result) and the non-finite inputs, each placed at
    // every lane offset of an 8-wide vector, at the first scalar-tail lane, and at
    // the last lane of the buffer.
    constexpr float kExtremes[] = {
        0.0f, -0.0f, 5.9604645e-8f, -5.9604645e-8f, 6.103515625e-5f, -6.103515625e-5f,
        11.0f, 11.09f, 11.2f, -17.0f, -24.0f, 65504.0f, -65504.0f,
        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    for (size_t N : {size_t{1}, size_t{7}, size_t{8}, size_t{9}, size_t{16}, size_t{17}}) {
      for (float value : kExtremes) {
        for (size_t pos : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{4},
                           size_t{5}, size_t{6}, size_t{7}, size_t{8}}) {
          if (pos < N) {
            TestExpFp16Extreme(N, pos, value);
          }
        }
        TestExpFp16Extreme(N, N - 1, value);
      }
    }
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
  }
};

template <bool Threaded>
class MlasSoftmaxTest : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;
  MatrixGuardBuffer<MLAS_FP16> BufferInputFp16;
  MatrixGuardBuffer<MLAS_FP16> BufferOutputFp16;
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

    Test(Input, Output, OutputReference, N, D, false, true);
    Test(Input, Output, OutputReference, N, D, true, true);
    Test(Input, Output, OutputReference, N, D, false, false);
    Test(Input, Output, OutputReference, N, D, true, false);
  }

  void Test(const float* Input, float* Output, float* OutputReference, size_t N, size_t D, bool LogSoftmax, bool SmoothSoftmax) {
    MlasComputeSoftmax(Input, Output, N, D, LogSoftmax, SmoothSoftmax, 0.0f, threadpool_);
    ReferenceSoftmax(Input, OutputReference, N, D, LogSoftmax, SmoothSoftmax);

    constexpr float AbsoluteTolerance = 1e-6f;
    constexpr float RelativeTolerance = 1e-6f;

    for (size_t nd = 0; nd < N * D; nd++) {
      float diff = std::fabs(Output[nd] - OutputReference[nd]);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(OutputReference[nd]) * RelativeTolerance)
          << "LogSoftmax:" << (int)LogSoftmax << " difference " << N << "/" << D
          << ", got: " << Output[nd] << ", expecting: " << OutputReference[nd];
    }
  }

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
  void TestReduceMaxFp16(size_t N, float MinimumValue, float MaximumValue) {
    MLAS_FP16* Input = BufferInputFp16.GetBuffer(N);

    std::default_random_engine generator(static_cast<unsigned>(N));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    float ref = std::numeric_limits<float>::lowest();

    for (size_t nd = 0; nd < N; nd++) {
      Input[nd] = MLAS_FP16(distribution(generator));
      ref = std::fmax(ref, Input[nd].ToFloat());
    }

    const auto* dispatch = GetMlasPlatform().SoftmaxDispatch;
    auto out = dispatch->ReduceMax_Fp16(Input, N).ToFloat();

    constexpr float AbsoluteTolerance = 1e-3f;
    constexpr float RelativeTolerance = 1e-3f;

    float diff = std::fabs(out - ref);
    ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(ref) * RelativeTolerance)
        << "ReduceMaxFp16: " << N << ", got: " << out << ", expecting: " << ref
        << ", diff: " << diff << ", r-diff: " << diff / std::fabs(ref);
  }

  void TestFp16(size_t N, size_t D, float MinimumValue, float MaximumValue, bool LogSoftmax, bool SmoothSoftmax) {
    MLAS_FP16* Input = BufferInputFp16.GetBuffer(N * D);
    MLAS_FP16* Output = BufferOutputFp16.GetBuffer(N * D);
    float* InputReference = BufferInput.GetBuffer(N * D);
    float* OutputReference = BufferOutputReference.GetBuffer(N * D);

    std::default_random_engine generator(static_cast<unsigned>(N * D));
    std::uniform_real_distribution<float> distribution(MinimumValue, MaximumValue);

    for (size_t nd = 0; nd < N * D; nd++) {
      Input[nd] = MLAS_FP16(distribution(generator));
      InputReference[nd] = Input[nd].ToFloat();
    }

    MlasComputeSoftmax(Input, Output, N, D, LogSoftmax, SmoothSoftmax, 0.0f, threadpool_);
    ReferenceSoftmax(InputReference, OutputReference, N, D, LogSoftmax, SmoothSoftmax);

    constexpr float AbsoluteTolerance = 5e-3f;
    constexpr float RelativeTolerance = 5e-3f;

    for (size_t nd = 0; nd < N * D; nd++) {
      float in = Input[nd].ToFloat();
      float ref = OutputReference[nd];
      float out = Output[nd].ToFloat();
      float diff = std::fabs(out - ref);
      ASSERT_TRUE(diff <= AbsoluteTolerance || diff <= std::fabs(ref) * RelativeTolerance)
          << "LogSoftmax:" << LogSoftmax << ", SmoothSoftmax: " << SmoothSoftmax << ", input " << in
          << ", got: " << out << ", expecting: " << ref << ", diff: " << diff << ", r-diff: " << diff / std::fabs(ref);
    }
  }
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

  void ReferenceSoftmax(const float* Input, float* Output, size_t N, size_t D, bool LogSoftmax, bool SmoothSoftmax) {
    for (size_t n = 0; n < N; n++) {
      float MaximumValue = std::numeric_limits<float>::lowest();

      for (size_t d = 0; d < D; d++) {
        MaximumValue = (std::max)(MaximumValue, Input[d]);
      }

      if (SmoothSoftmax && MaximumValue < 0.0f) {
        MaximumValue = 0.0f;
      }

      double Sum = 0.0;

      for (size_t d = 0; d < D; d++) {
        double e = std::exp(double(Input[d]) - double(MaximumValue));
        Sum += e;
        Output[d] = float(e);
      }

      if (SmoothSoftmax) {
        Sum += expf(-MaximumValue);
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
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
      TestReduceMaxFp16(d, -10.f, 10.f);
      TestFp16(1, d, -10.f, 10.f, false, true);
      TestFp16(1, d, -10.f, 10.f, true, true);
      TestFp16(1, d, -10.f, 10.f, false, false);
      TestFp16(1, d, -10.f, 10.f, true, false);
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    }

    Test(3, 128, 20.f, 30.f);
    Test(63, 95, -150.f, 190.f);
    Test(16, 211, 20.f, 30.f);
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
    TestFp16(3, 128, 3.f, 7.f, false, true);
    TestFp16(3, 128, 3.f, 7.f, true, true);
    TestFp16(3, 128, 3.f, 7.f, false, false);
    TestFp16(3, 128, 3.f, 7.f, true, false);
    TestFp16(63, 95, -15.f, 19.f, false, true);
    TestFp16(63, 95, -15.f, 19.f, true, true);
    TestFp16(63, 95, -15.f, 19.f, false, false);
    TestFp16(63, 95, -15.f, 19.f, true, false);
    TestFp16(16, 211, -7.f, -3.f, false, true);
    TestFp16(16, 211, -7.f, -3.f, true, true);
    TestFp16(16, 211, -7.f, -3.f, false, false);
    TestFp16(16, 211, -7.f, -3.f, true, false);
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasSoftmaxTest<false>>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasComputeExpTest>::RegisterShortExecute();
    if (GetMlasThreadPool() != nullptr) {
      count += MlasDirectShortExecuteTests<MlasSoftmaxTest<true>>::RegisterShortExecute();
    }
  }
  return count;
});
