// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_fp16.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

class MlasFp16ActivationTest : public MlasTestBase {
 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Fp16Activation");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    union AliasedValue {
      unsigned u;
      float f;
    };

    // N.B. The test data includes values at the edge of Tanh/Logistic boundaries.
    static const AliasedValue TestData[] = {
        {0x00000001},  // positive denormal
        {0x80000001},  // negative denormal
        {0x7fc00000},  // positive NaN
        {0xffc00000},  // negative NaN
        {0x00000000},  // 0.0f
        {0x80000000},  // -0.0f
        {0x3e800000},  // 0.25f
        {0xbe800000},  // -0.25f
        {0x40800000},  // 4.0f
        {0xc0800000},  // -4.0f
        {0x41200000},  // 10.0f
        {0xc1200000},  // -10.0f
        {0xc18866eb},  // -17.0502529144f
        {0xc18869bb},  // -17.0516262054f
        {0xc18852a8},  // -17.0403594971f
        {0xc18844aa},  // -17.0335273743f
        {0x418866eb},  // +17.0502529144f
        {0x418869bb},  // +17.0516262054f
        {0x418852a8},  // +17.0403594971f
        {0x418844aa}   // +17.0335273743f
    };

    constexpr size_t M = 5;
    constexpr size_t N = 23;

    MatrixGuardBuffer<MLFp16> HalfBuffer1;
    auto* testData1 = HalfBuffer1.GetBuffer(M * N, true);
    MatrixGuardBuffer<MLFp16> HalfBuffer2;
    auto* testData2 = HalfBuffer2.GetBuffer(M * N, true);
    MatrixGuardBuffer<float> FloatBuffer;
    auto* fpBuffer = FloatBuffer.GetBuffer(M * N, true);

    MLAS_ACTIVATION_KIND acts[] = {
        MlasIdentityActivation,
        MlasReluActivation,
        MlasLeakyReluActivation,
        MlasLogisticActivation,
        MlasClipActivation,
        MlasHardSigmoidActivation};

    MLAS_ACTIVATION Activation;
    MLAS_HALF_GEMM_ACTIVATION_PROCESSOR proc(Activation);
    MLAS_HALF_GEMM_2FLOAT_PROCESSOR converter(Activation, fpBuffer, N);
    for (auto kind : acts) {
      Activation.ActivationKind = MLAS_ACTIVATION_KIND(kind);

      if (Activation.ActivationKind == MlasLeakyReluActivation) {
        Activation.Parameters.LeakyRelu.alpha = 0.2f;
      } else if (Activation.ActivationKind == MlasClipActivation) {
        Activation.Parameters.Clip.minimum = 0.0f;
        Activation.Parameters.Clip.maximum = 6.0f;
      } else if (Activation.ActivationKind == MlasHardSigmoidActivation){
        Activation.Parameters.HardSigmoid.alpha = 0.2f;
        Activation.Parameters.HardSigmoid.beta = 0.12f;
      }

      //
      // Test the vectorized activations.
      //

      for (size_t i = 0; i < _countof(TestData); i++) {
        testData1[i] = TestData[i].f;
        testData2[i] = TestData[i].f;
      }
      constexpr float MinimumFillValue = -11.0f;
      size_t offset = 7;
      for (size_t i = _countof(TestData); i < M * N; i++) {
        offset = (offset + 19) % 23;
        testData1[i] = (MinimumFillValue + offset) / 16.0f;
        testData2[i] = testData1[i];
      }

      proc.Process(reinterpret_cast<MLAS_FP16*>(testData1), 0, 0, M, N, N);
      converter.Process(reinterpret_cast<MLAS_FP16*>(testData2), 0, 0, M, N, N);

      for (size_t i = 0; i < M*N; i++) {
        float actual = testData1[i].ToFloat();
        if (std::isnan(actual)) {
          EXPECT_TRUE(std::isnan(fpBuffer[i]))
              << ", Vector Activation Kind:" << (int)kind << ", i=" << i << ", value:"
              << std::setw(8) << std::setfill('0') << std::hex << actual << ", expecting:"
              << std::setw(8) << std::setfill('0') << std::hex << fpBuffer[i];

        } else {
          float diff = std::abs(actual - fpBuffer[i]);
          float top = std::max(std::abs(actual), std::abs(fpBuffer[i]));
          float ratio = 0;
          if (top > 0.0001) {
            ratio = diff / top;
          }
          EXPECT_TRUE(ratio < 0.005)
              << ", Vector Activation Kind:" << (int)kind << ", i=" << i << ", value:"
              << actual << ", expecting:" << fpBuffer[i];
        }
      }
    }
  }
};

template<> MlasFp16ActivationTest* MlasTestFixture<MlasFp16ActivationTest>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? MlasDirectShortExecuteTests<MlasFp16ActivationTest>::RegisterShortExecute() : 0;
});

#endif // fp16 vector intrinsic supported