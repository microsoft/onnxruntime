// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"

class MlasActivationTest : public MlasTestBase {
 public:
  static const char* GetTestSuiteName() {
    static const std::string suite_name("Activation");
    return suite_name.c_str();
  }

  void ExecuteShort(void) override {
    union AliasedValue {
      unsigned u;
      float f;
    };

    // N.B. The test data includes values at the edge of Tanh/Logistic boundaries.
    //    Identity,     Relu,         LeakyRelu,    Tanh,         Logistic,     Clip,
    static const AliasedValue TestData[20][6] = {
        { {0x00000001}, {0x00000001}, {0x00000001}, {0x00000000}, {0x3f000000}, {0x00000001}, }, // positive denormal
        { {0x80000001}, {0x00000000}, {0x80000000}, {0x80000000}, {0x3f000000}, {0x00000000}, }, // negative denormal
        { {0x7ff00002}, {0x7ff00002}, {0x7ff00002}, {0x7ff00002}, {0x7ff00002}, {0x7ff00002}, }, // positive NaN
        { {0xfff00002}, {0xfff00002}, {0xfff00002}, {0xfff00002}, {0xfff00002}, {0xfff00002}, }, // negative NaN
        { {0x00000000}, {0x00000000}, {0x00000000}, {0x00000000}, {0x3f000000}, {0x00000000}, }, // 0.0f
        { {0x80000000}, {0x80000000}, {0x80000000}, {0x80000000}, {0x3f000000}, {0x80000000}, }, // -0.0f
        { {0x3e800000}, {0x3e800000}, {0x3e800000}, {0x3e7acbf5}, {0x3f0feacc}, {0x3e800000}, }, // 0.25f
        { {0xbe800000}, {0x00000000}, {0xbd4ccccd}, {0xbe7acbf5}, {0x3ee02a67}, {0x00000000}, }, // -0.25f
        { {0x40800000}, {0x40800000}, {0x40800000}, {0x3f7fd40a}, {0x3f7b6541}, {0x40800000}, }, // 4.0f
        { {0xc0800000}, {0x00000000}, {0xbf4ccccd}, {0xbf7fd40a}, {0x3c9357e0}, {0x00000000}, }, // -4.0f
        { {0x41200000}, {0x41200000}, {0x41200000}, {0x3f800000}, {0x3f7ffd06}, {0x40c00000}, }, // 10.0f
        { {0xc1200000}, {0x00000000}, {0xc0000000}, {0xbf800000}, {0x383e6000}, {0x00000000}, }, // -10.0f
        { {0xc18866eb}, {0x00000000}, {0xc05a3e45}, {0xbf800000}, {0x33000000}, {0x00000000}, }, // -17.0502529144f
        { {0xc18869bb}, {0x00000000}, {0xc05a42c5}, {0xbf800000}, {0x33c00000}, {0x00000000}, }, // -17.0516262054f
        { {0xc18852a8}, {0x00000000}, {0xc05a1dda}, {0xbf800000}, {0x00000000}, {0x00000000}, }, // -17.0403594971f
        { {0xc18844aa}, {0x00000000}, {0xc05a0777}, {0xbf800000}, {0x00000000}, {0x00000000}, }, // -17.0335273743f
        { {0x418866eb}, {0x418866eb}, {0x418866eb}, {0x3f800000}, {0x3f800000}, {0x40c00000}, }, // +17.0502529144f
        { {0x418869bb}, {0x418869bb}, {0x418869bb}, {0x3f800000}, {0x3f7ffffe}, {0x40c00000}, }, // +17.0516262054f
        { {0x418852a8}, {0x418852a8}, {0x418852a8}, {0x3f800000}, {0x3f800000}, {0x40c00000}, }, // +17.0403594971f
        { {0x418844aa}, {0x418844aa}, {0x418844aa}, {0x3f800000}, {0x3f800000}, {0x40c00000}, }, // +17.0335273743f
    };

    MLAS_ACTIVATION Activation;
    AliasedValue Buffer[_countof(TestData)];

    for (unsigned kind = 0; kind < unsigned(MlasClipActivation); kind++) {
      Activation.ActivationKind = MLAS_ACTIVATION_KIND(kind);

      if (Activation.ActivationKind == MlasLeakyReluActivation) {
        Activation.Parameters.LeakyRelu.alpha = 0.2f;
      } else if (Activation.ActivationKind == MlasClipActivation) {
        Activation.Parameters.Clip.minimum = 0.0f;
        Activation.Parameters.Clip.maximum = 6.0f;
      }

      //
      // Test the vectorized activations.
      //

      for (unsigned i = 0; i < _countof(TestData); i++) {
        Buffer[i].u = TestData[i][0].u;
      }

      MlasActivation(&Activation, &Buffer[0].f, nullptr, 1, _countof(Buffer), _countof(Buffer));

      for (unsigned i = 0; i < _countof(TestData); i++) {
        // Sensitive to comparing positive/negative zero and NaNs.
        EXPECT_TRUE(Buffer[i].u == TestData[i][kind].u || Buffer[i].f == TestData[i][kind].f)
            << ", Vector Activation Kind:" << (int)kind << ", i=" << i << ", value:"
            << std::setw(8) << std::setfill('0') <<std::hex << Buffer[i].u << ", expecting:"
            << std::setw(8) << std::setfill('0') <<std::hex << TestData[i][kind].u;
      }

      //
      // Test the scalar activations.
      //

      for (unsigned i = 0; i < _countof(TestData); i++) {
        Buffer[i].u = TestData[i][0].u;
        MlasActivation(&Activation, &Buffer[i].f, nullptr, 1, 1, 1);
      }

      for (unsigned i = 0; i < _countof(TestData); i++) {
        // Sensitive to comparing positive/negative zero and NaNs.
        EXPECT_TRUE(Buffer[i].u == TestData[i][kind].u || Buffer[i].f == TestData[i][kind].f)
            << ", Scalar Activation Kind:" << (int)kind << ", i=" << i << ", value:"
            << std::setw(8) << std::setfill('0') <<std::hex << Buffer[i].u << ", expecting:"
            << std::setw(8) << std::setfill('0') <<std::hex << TestData[i][kind].u;
      }
    }
  }
};

template<> MlasActivationTest* MlasTestFixture<MlasActivationTest>::mlas_tester(nullptr);

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? MlasDirectShortExecuteTests<MlasActivationTest>::RegisterShortExecute() : 0;
});
