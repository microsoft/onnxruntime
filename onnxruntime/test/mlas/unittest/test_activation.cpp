// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <iomanip>
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
    //    Identity,     Relu,         LeakyRelu,    Tanh,         Logistic,     Clip,         HardSigmoid,  HardSwish
    static const AliasedValue TestData[20][8] = {
        {
            {0x00000001},
            {0x00000001},
            {0x00000001},
            {0x00000000},
            {0x3f000000},
            {0x00000001},
            {0x3df5c28f},
            {0x00000000},
        },  // positive denormal
        {
            {0x80000001},
            {0x00000000},
            {0x80000000},
            {0x80000000},
            {0x3f000000},
            {0x00000000},
            {0x3df5c28f},
            {0x80000000},
        },  // negative denormal
        {
            {0x7ff00002},
            {0x7ff00002},
            {0x7ff00002},
            {0x7ff00002},
            {0x7ff00002},
            {0x7ff00002},
            {0x7ff00002},
            {0x7ff00002},
        },  // positive NaN
        {
            {0xfff00002},
            {0xfff00002},
            {0xfff00002},
            {0xfff00002},
            {0xfff00002},
            {0xfff00002},
            {0xfff00002},
            {0xfff00002},
        },  // negative NaN
        {
            {0x00000000},
            {0x00000000},
            {0x00000000},
            {0x00000000},
            {0x3f000000},
            {0x00000000},
            {0x3df5c28f},
            {0x00000000},
        },  // 0.0f
        {
            {0x80000000},
            {0x80000000},
            {0x80000000},
            {0x80000000},
            {0x3f000000},
            {0x80000000},
            {0x3df5c28f},
            {0x80000000},
        },  // -0.0f
        {
            {0x3e800000},
            {0x3e800000},
            {0x3e800000},
            {0x3e7acbf5},
            {0x3f0feacc},
            {0x3e800000},
            {0x3e2e147b},
            {0x3e0aaaab},
        },  // 0.25f
        {
            {0xbe800000},
            {0x00000000},
            {0xbd4ccccd},
            {0xbe7acbf5},
            {0x3ee02a67},
            {0x00000000},
            {0x3d8f5c28},
            {0xbdeaaaab},
        },  // -0.25f
        {
            {0x40800000},
            {0x40800000},
            {0x40800000},
            {0x3f7fd40a},
            {0x3f7b6541},
            {0x40800000},
            {0x3f6b851f},
            {0x40800000},
        },  // 4.0f
        {
            {0xc0800000},
            {0x00000000},
            {0xbf4ccccd},
            {0xbf7fd40a},
            {0x3c9357e0},
            {0x00000000},
            {0x00000000},
            {0x80000000},
        },  // -4.0f
        {
            {0x41200000},
            {0x41200000},
            {0x41200000},
            {0x3f800000},
            {0x3f7ffd06},
            {0x40c00000},
            {0x3f800000},
            {0x41200000},
        },  // 10.0f
        {
            {0xc1200000},
            {0x00000000},
            {0xc0000000},
            {0xbf800000},
            {0x383e6000},
            {0x00000000},
            {0x00000000},
            {0x80000000},
        },  // -10.0f
        {
            {0xc18866eb},
            {0x00000000},
            {0xc05a3e45},
            {0xbf800000},
            {0x33000000},
            {0x00000000},
            {0x00000000},
            {0x80000000},
        },  // -17.0502529144f
        {
            {0xc18869bb},
            {0x00000000},
            {0xc05a42c5},
            {0xbf800000},
            {0x33c00000},
            {0x00000000},
            {0x00000000},
            {0x80000000},
        },  // -17.0516262054f
        {
            {0xc18852a8},
            {0x00000000},
            {0xc05a1dda},
            {0xbf800000},
            {0x00000000},
            {0x00000000},
            {0x00000000},
            {0x80000000},
        },  // -17.0403594971f
        {
            {0xc18844aa},
            {0x00000000},
            {0xc05a0777},
            {0xbf800000},
            {0x00000000},
            {0x00000000},
            {0x00000000},
            {0x80000000},
        },  // -17.0335273743f
        {
            {0x418866eb},
            {0x418866eb},
            {0x418866eb},
            {0x3f800000},
            {0x3f800000},
            {0x40c00000},
            {0x3f800000},
            {0x418866eb},
        },  // +17.0502529144f
        {
            {0x418869bb},
            {0x418869bb},
            {0x418869bb},
            {0x3f800000},
            {0x3f7ffffe},
            {0x40c00000},
            {0x3f800000},
            {0x418869bb},
        },  // +17.0516262054f
        {
            {0x418852a8},
            {0x418852a8},
            {0x418852a8},
            {0x3f800000},
            {0x3f800000},
            {0x40c00000},
            {0x3f800000},
            {0x418852a8},
        },  // +17.0403594971f
        {
            {0x418844aa},
            {0x418844aa},
            {0x418844aa},
            {0x3f800000},
            {0x3f800000},
            {0x40c00000},
            {0x3f800000},
            {0x418844aa},
        },  // +17.0335273743f
    };

    MLAS_ACTIVATION Activation;
    AliasedValue Buffer[_countof(TestData)];

    for (unsigned kind = 0; kind < unsigned(MlasActivationKindCount); kind++) {
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

      for (unsigned i = 0; i < _countof(TestData); i++) {
        Buffer[i].u = TestData[i][0].u;
      }

      MlasActivation(&Activation, &Buffer[0].f, nullptr, 1, _countof(Buffer), _countof(Buffer));
      for (unsigned i = 0; i < _countof(TestData); i++) {
        const float actual = Buffer[i].f;
        const float expected = TestData[i][kind].f;
        // Match NaNs by classification and allow rounding differences and either sign of zero.
        const float tolerance = 0.000001f * std::max(1.0f, std::fabs(expected));
        EXPECT_TRUE(actual == expected || (std::isnan(actual) && std::isnan(expected)) ||
                    (std::isfinite(actual) && std::isfinite(expected) && std::fabs(actual - expected) < tolerance))
            << ", Vector Activation Kind:" << kind << ", i=" << i
            << ", value:" << actual << ", expecting:" << expected;
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
        float error = std::min(std::fabs((Buffer[i].f - TestData[i][kind].f) / TestData[i][kind].f), std::fabs(Buffer[i].f - TestData[i][kind].f));
        EXPECT_TRUE(Buffer[i].u == TestData[i][kind].u || Buffer[i].f == TestData[i][kind].f || error < 0.000001f ||
                    (std::isnan(Buffer[i].f) && std::isnan(TestData[i][kind].f)))
            << ", Scalar Activation Kind:" << (int)kind << ", i=" << i << ", value:"
            << std::setw(8) << std::setfill('0') << std::hex << Buffer[i].u << ", expecting:"
            << std::setw(8) << std::setfill('0') << std::hex << TestData[i][kind].u;
      }
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? MlasDirectShortExecuteTests<MlasActivationTest>::RegisterShortExecute() : 0;
});

TEST(ActivationMatrix, BiasAndStride) {
  constexpr MLAS_ACTIVATION_KIND kinds[] = {
      MlasIdentityActivation, MlasReluActivation, MlasLeakyReluActivation,
      MlasClipActivation, MlasHardSigmoidActivation};
  constexpr float values[] = {-10.0f, -4.0f, -0.25f, 0.0f, 0.25f, 4.0f, 10.0f};
  constexpr float expected[][_countof(values)] = {
      {-10.0f, -4.0f, -0.25f, 0.0f, 0.25f, 4.0f, 10.0f},
      {0.0f, 0.0f, 0.0f, 0.0f, 0.25f, 4.0f, 10.0f},
      {-2.0f, -0.8f, -0.05f, 0.0f, 0.25f, 4.0f, 10.0f},
      {0.0f, 0.0f, 0.0f, 0.0f, 0.25f, 4.0f, 6.0f},
      {0.0f, 0.0f, 0.07f, 0.12f, 0.17f, 0.92f, 1.0f},
  };
  constexpr float bias[] = {-1.0f, 0.0f, 1.0f};
  constexpr size_t widths[] = {0, 1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129};
  constexpr size_t paddings[] = {0, 5};
  constexpr float sentinel = -12345.0f;
  MatrixGuardBuffer<float> guarded_buffer;

  for (size_t kind = 0; kind < _countof(kinds); ++kind) {
    MLAS_ACTIVATION activation{};
    activation.ActivationKind = kinds[kind];
    if (activation.ActivationKind == MlasLeakyReluActivation) {
      activation.Parameters.LeakyRelu.alpha = 0.2f;
    } else if (activation.ActivationKind == MlasClipActivation) {
      activation.Parameters.Clip.minimum = 0.0f;
      activation.Parameters.Clip.maximum = 6.0f;
    } else if (activation.ActivationKind == MlasHardSigmoidActivation) {
      activation.Parameters.HardSigmoid.alpha = 0.2f;
      activation.Parameters.HardSigmoid.beta = 0.12f;
    }

    for (bool add_bias : {false, true}) {
      for (size_t n : widths) {
        for (size_t padding : paddings) {
          const size_t ldc = n + padding;
          const size_t count = _countof(bias) * ldc;
          float* buffer = guarded_buffer.GetBuffer(count + 1);
          std::fill_n(buffer, count + 1, sentinel);
          SCOPED_TRACE(::testing::Message() << "kind=" << kinds[kind] << ", bias=" << add_bias
                                            << ", N=" << n << ", ldc=" << ldc);

          for (size_t row = 0; row < _countof(bias); ++row) {
            for (size_t col = 0; col < n; ++col) {
              const size_t index = (row + col) % _countof(values);
              buffer[row * ldc + col] = values[index] - (add_bias ? bias[row] : 0.0f);
            }
          }

          MlasActivation(&activation, buffer, add_bias ? bias : nullptr, _countof(bias), n, ldc);

          for (size_t row = 0; row < _countof(bias); ++row) {
            for (size_t col = 0; col < n; ++col) {
              EXPECT_NEAR(buffer[row * ldc + col], expected[kind][(row + col) % _countof(values)], 0.000001f)
                  << "row=" << row << ", col=" << col;
            }
            for (size_t col = n; col < ldc; ++col) {
              EXPECT_EQ(buffer[row * ldc + col], sentinel) << "row=" << row << ", padding=" << col;
            }
          }
          EXPECT_EQ(buffer[count], sentinel);
        }
      }
    }
  }
}
