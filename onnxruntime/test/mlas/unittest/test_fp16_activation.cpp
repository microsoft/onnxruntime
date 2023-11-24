// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_fp16.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

bool check_equal(float actual, float expected) {
  if (std::isnan(actual)) {
    return std::isnan(expected);
  } else {
    float diff = std::abs(actual - expected);
    float top = std::max(std::abs(actual), std::abs(expected));
    float ratio = 0;
    if (top > 0.0001) {
      ratio = diff / top;
    }
    return ratio < 0.005;
  }
}

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
    constexpr float MinimumFillValue = -11.0f;

    MatrixGuardBuffer<MLFp16> HalfBuffer1;
    auto* testData1 = HalfBuffer1.GetBuffer(M * N, true);
    MatrixGuardBuffer<MLFp16> HalfBuffer2;
    auto* testData2 = HalfBuffer2.GetBuffer(M * N, true);
    MatrixGuardBuffer<MLFp16> HalfBuffer3;
    auto* testData3 = HalfBuffer3.GetBuffer(M * N, true);
    MatrixGuardBuffer<MLFp16> AddonBuffer;
    auto addonData = AddonBuffer.GetBuffer(M * N, true);
    MatrixGuardBuffer<float> FloatBuffer;
    auto* fpBuffer = FloatBuffer.GetBuffer(M * N, true);
    MatrixGuardBuffer<float> FloatBuffer1;
    auto* fpAddBuffer = FloatBuffer1.GetBuffer(M * N, true);

    size_t o = 3;
    for (size_t i = 0; i < M * N; i++) {
      o = (o + 19) % 23;
      addonData[i] = (MinimumFillValue + o) / 16.0f;
    }

    MLAS_ACTIVATION_KIND acts[] = {
        MlasIdentityActivation,
        MlasReluActivation,
        MlasLeakyReluActivation,
        MlasTanhActivation,
        MlasLogisticActivation,
        MlasClipActivation,
        MlasHardSigmoidActivation};

    MLAS_ACTIVATION Activation;
    MLAS_HALF_GEMM_ACTIVATION_PROCESSOR proc(Activation, nullptr);
    MLAS_HALF_GEMM_ACTIVATION_PROCESSOR addon(Activation, reinterpret_cast<const MLAS_FP16*>(addonData));
    for (auto kind : acts) {
      Activation.ActivationKind = MLAS_ACTIVATION_KIND(kind);

      if (Activation.ActivationKind == MlasLeakyReluActivation) {
        Activation.Parameters.LeakyRelu.alpha = 0.2f;
      } else if (Activation.ActivationKind == MlasClipActivation) {
        Activation.Parameters.Clip.minimum = 0.0f;
        Activation.Parameters.Clip.maximum = 6.0f;
      } else if (Activation.ActivationKind == MlasHardSigmoidActivation) {
        Activation.Parameters.HardSigmoid.alpha = 0.2f;
        Activation.Parameters.HardSigmoid.beta = 0.12f;
      }

      //
      // Test the vectorized activations.
      //

      for (size_t i = 0; i < _countof(TestData); i++) {
        testData1[i] = TestData[i].f;
        testData2[i] = TestData[i].f;
        testData3[i] = TestData[i].f;
        fpBuffer[i] = TestData[i].f;
        fpAddBuffer[i] = TestData[i].f + addonData[i].ToFloat();
      }
      size_t offset = 7;
      for (size_t i = _countof(TestData); i < M * N; i++) {
        offset = (offset + 19) % 23;
        float f = (MinimumFillValue + offset) / 16.0f;
        testData1[i] = f;
        testData2[i] = testData1[i];
        testData3[i] = testData1[i];
        fpBuffer[i] = f;
        fpAddBuffer[i] = f + addonData[i].ToFloat();
      }

      proc.Process(reinterpret_cast<MLAS_FP16*>(testData1), 0, 0, M, N, N);
      MlasActivation(&Activation, fpBuffer, nullptr, M, N, N);
      MlasActivation(&Activation, fpAddBuffer, nullptr, M, N, N);
      addon.Process(reinterpret_cast<MLAS_FP16*>(testData3), 0, 0, M, N, N);

      for (size_t i = 0; i < M * N; i++) {
        float actual = testData1[i].ToFloat();
        EXPECT_TRUE(check_equal(actual, fpBuffer[i]))
            << ", Vector Activation Kind:" << (int)kind << ", i=" << i << ", value:"
            << std::setw(8) << std::setfill('0') << std::hex << actual << ", expecting:"
            << std::setw(8) << std::setfill('0') << std::hex << fpBuffer[i];

        float addonActual = testData3[i].ToFloat();
        EXPECT_TRUE(check_equal(addonActual, fpAddBuffer[i]))
            << ", Vector + Activation Kind:" << (int)kind << ", i=" << i << ", value:"
            << std::setw(8) << std::setfill('0') << std::hex << actual << ", expecting:"
            << std::setw(8) << std::setfill('0') << std::hex << fpBuffer[i];
      }
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? MlasDirectShortExecuteTests<MlasFp16ActivationTest>::RegisterShortExecute() : 0;
});

#endif  // fp16 vector intrinsic supported
