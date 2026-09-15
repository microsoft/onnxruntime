// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "mlasi.h"

#include <array>
#include <cstring>

namespace {

constexpr MLAS_ACTIVATION_KIND kKinds[] = {
    MlasIdentityActivation, MlasReluActivation, MlasLeakyReluActivation,
    MlasClipActivation, MlasHardSigmoidActivation, MlasHardSwishActivation};

MLAS_ACTIVATION MakeActivation(MLAS_ACTIVATION_KIND kind) {
  MLAS_ACTIVATION activation{};
  activation.ActivationKind = kind;
  if (kind == MlasLeakyReluActivation) {
    activation.Parameters.LeakyRelu.alpha = 0.2f;
  } else if (kind == MlasClipActivation) {
    activation.Parameters.Clip.minimum = -0.5f;
    activation.Parameters.Clip.maximum = 6.0f;
  } else if (kind == MlasHardSigmoidActivation) {
    activation.Parameters.HardSigmoid.alpha = 0.2f;
    activation.Parameters.HardSigmoid.beta = 0.12f;
  }
  return activation;
}

float Reference(const MLAS_ACTIVATION& activation, float value) {
  switch (activation.ActivationKind) {
    case MlasReluActivation:
      return std::max(value, 0.0f);
    case MlasLeakyReluActivation:
      return value >= 0.0f ? value : value * activation.Parameters.LeakyRelu.alpha;
    case MlasClipActivation:
      return std::min(std::max(value, activation.Parameters.Clip.minimum), activation.Parameters.Clip.maximum);
    case MlasHardSigmoidActivation:
      return std::max(std::min(value * activation.Parameters.HardSigmoid.alpha +
                                   activation.Parameters.HardSigmoid.beta,
                               1.0f),
                      0.0f);
    case MlasHardSwishActivation:
      return value * std::max(std::min(value * (1.0f / 6.0f) + 0.5f, 1.0f), 0.0f);
    default:
      return value;
  }
}

void ExpectSame(float actual, float expected) {
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(actual));
  } else if (expected == 0.0f) {
    EXPECT_EQ(actual, expected);
#if defined(MLAS_TARGET_RISCV64)
    EXPECT_EQ(std::signbit(actual), std::signbit(expected));
#endif
  } else if (std::isinf(expected)) {
    EXPECT_EQ(actual, expected);
  } else {
    EXPECT_NEAR(actual, expected, 1.0e-6f * std::max(1.0f, std::abs(expected)));
  }
}

TEST(FusedActivation, MatrixAndTail) {
  constexpr size_t lengths[] = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129};
  const float values[] = {
      -INFINITY, -10.0f, -3.0f, -1.0f, -0.0f, 0.0f,
      -std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::denorm_min(),
      0.25f, 1.0f, 3.0f, 10.0f, INFINITY, std::numeric_limits<float>::quiet_NaN()};
  const float bias[] = {0.0f, -0.75f, 1.0f};
  std::array<float, 512> output;
  std::array<float, 512> expected;
  for (auto kind : kKinds) {
    auto activation = MakeActivation(kind);
    for (size_t n : lengths) {
      for (size_t m : {size_t(0), size_t(1), size_t(3)}) {
        for (size_t padding : {size_t(0), size_t(5)}) {
          for (bool add_bias : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "kind=" << kind << " M=" << m << " N=" << n
                                              << " padding=" << padding << " bias=" << add_bias);
            const size_t ldc = n + padding;
            output.fill(12345.0f);
            expected.fill(12345.0f);
            for (size_t row = 0; row < m; ++row) {
              for (size_t col = 0; col < n; ++col) {
                const size_t index = 1 + row * ldc + col;
                output[index] = values[(row * n + col) % _countof(values)];
                const float value = add_bias ? output[index] + bias[row] : output[index];
                expected[index] = Reference(activation, value);
              }
            }
            MlasActivation(&activation, output.data() + 1, add_bias ? bias : nullptr, m, n, ldc);
            for (size_t i = 0; i < output.size(); ++i) {
              ExpectSame(output[i], expected[i]);
            }
          }
        }
      }
    }
  }
}

TEST(FusedActivation, GuardedTail) {
  MatrixGuardBuffer<float> guarded;
  for (auto kind : kKinds) {
    auto activation = MakeActivation(kind);
    for (size_t n : {size_t(1), size_t(3), size_t(4), size_t(7), size_t(9), size_t(31), size_t(33), size_t(129)}) {
      for (bool add_bias : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "kind=" << kind << " N=" << n << " bias=" << add_bias);
        float* buffer = guarded.GetBuffer(n);
        const float bias = 0.5f;
        std::array<float, 129> expected;
        for (size_t i = 0; i < n; ++i) {
          expected[i] = Reference(activation, add_bias ? buffer[i] + bias : buffer[i]);
        }
        MlasActivation(&activation, buffer, add_bias ? &bias : nullptr, 1, n, n);
        for (size_t i = 0; i < n; ++i) {
          ExpectSame(buffer[i], expected[i]);
        }
      }
    }
  }
}

#if defined(MLAS_TARGET_RISCV64)
TEST(FusedActivation, ReluAndClipPreserveNaNPayloadsAndZero) {
  constexpr uint32_t bits[] = {0x7fc12345, 0xffc12345, 0x7fa12345, 0xffa12345,
                               0x80000000, 0, 1, 0x3f800000};
  for (auto kind : {MlasReluActivation, MlasClipActivation}) {
    auto activation = MakeActivation(kind);
    std::array<uint32_t, 129> expected;
    std::array<float, 129> buffer;
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = bits[i % _countof(bits)];
    std::memcpy(buffer.data(), expected.data(), sizeof(buffer));
    MlasActivation(&activation, buffer.data(), nullptr, 1, buffer.size(), buffer.size());
    EXPECT_EQ(std::memcmp(buffer.data(), expected.data(), sizeof(buffer)), 0);
  }
}
#endif

TEST(FusedActivation, IdentityWithoutBiasIsNoOp) {
  auto activation = MakeActivation(MlasIdentityActivation);
  std::array<uint32_t, 9> bits = {0x7fc12345, 0xffc12345, 0x80000000, 0, 1, 0x80000001,
                                  0x7f800000, 0xff800000, 0x3f800000};
  std::array<float, 9> buffer;
  std::memcpy(buffer.data(), bits.data(), sizeof(buffer));
  MlasActivation(&activation, buffer.data(), nullptr, 1, buffer.size(), buffer.size());
  EXPECT_EQ(std::memcmp(buffer.data(), bits.data(), sizeof(buffer)), 0);
}

#if defined(MLAS_TARGET_RISCV64) && defined(MLAS_USE_RVV)
TEST(FusedActivationRvv, RuntimeDispatch) {
  const auto& platform = GetMlasPlatform();
  if (platform.GemmFloatKernel == MlasGemmFloatKernelRvv) {
    EXPECT_EQ(platform.MlasActivationOverride, MlasActivationRvv);
  } else {
    EXPECT_EQ(platform.MlasActivationOverride, nullptr);
  }
}

TEST(FusedActivationRvv, MatchesGenericBoundaryAndRandomInputs) {
  auto& platform = GetMlasPlatform();
  const auto activation_override = platform.MlasActivationOverride;
  if (activation_override == nullptr) {
    GTEST_SKIP() << "RVV is unavailable or forced off";
  }
  ASSERT_EQ(activation_override, MlasActivationRvv);
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> random(-10.0f, 10.0f);
  const float exceptional[] = {-0.0f, 0.0f, -INFINITY, INFINITY, std::numeric_limits<float>::quiet_NaN()};
  const float alphas[] = {0.2f, -0.5f, 0.0f, INFINITY, std::numeric_limits<float>::quiet_NaN()};
  const float bias[] = {-0.0f, 0.5f, std::numeric_limits<float>::quiet_NaN()};
  std::array<float, 512> actual;
  std::array<float, 512> expected;
  for (auto kind : kKinds) {
    for (float alpha : alphas) {
      auto activation = MakeActivation(kind);
      if (kind == MlasLeakyReluActivation) activation.Parameters.LeakyRelu.alpha = alpha;
      for (size_t n : {size_t(1), size_t(4), size_t(7), size_t(8), size_t(9), size_t(31), size_t(33), size_t(129)}) {
        for (bool add_bias : {false, true}) {
          SCOPED_TRACE(::testing::Message() << "kind=" << kind << " N=" << n << " alpha=" << alpha
                                            << " bias=" << add_bias);
          const size_t ldc = n + 3;
          for (size_t i = 0; i < actual.size(); ++i) {
            actual[i] = i % 2 == 0 ? exceptional[(i / 2) % _countof(exceptional)] : random(generator);
          }
          expected = actual;
          // Run the existing implementation as a reference; restore before assertions.
          platform.MlasActivationOverride = nullptr;
          MlasActivation(&activation, expected.data(), add_bias ? bias : nullptr, 3, n, ldc);
          platform.MlasActivationOverride = activation_override;
          MlasActivation(&activation, actual.data(), add_bias ? bias : nullptr, 3, n, ldc);
          for (size_t i = 0; i < actual.size(); ++i) {
            ExpectSame(actual[i], expected[i]);
          }
        }
      }
    }
  }
}

TEST(FusedActivationRvv, HardSigmoidNonfiniteParameters) {
  auto& platform = GetMlasPlatform();
  const auto activation_override = platform.MlasActivationOverride;
  if (activation_override == nullptr) {
    GTEST_SKIP() << "RVV is unavailable or forced off";
  }
  const float parameters[][2] = {{std::numeric_limits<float>::max(), -INFINITY},
                                 {INFINITY, -INFINITY},
                                 {0.0f, NAN},
                                 {NAN, 0.0f},
                                 {1.0f, INFINITY}};
  for (const auto& parameter : parameters) {
    auto activation = MakeActivation(MlasHardSigmoidActivation);
    activation.Parameters.HardSigmoid.alpha = parameter[0];
    activation.Parameters.HardSigmoid.beta = parameter[1];
    std::array<float, 33> actual;
    actual.fill(2.0f);
    auto expected = actual;
    platform.MlasActivationOverride = nullptr;
    MlasActivation(&activation, expected.data(), nullptr, 1, expected.size(), expected.size());
    platform.MlasActivationOverride = activation_override;
    MlasActivation(&activation, actual.data(), nullptr, 1, actual.size(), actual.size());
    for (size_t i = 0; i < actual.size(); ++i) ExpectSame(actual[i], expected[i]);
  }
}

TEST(FusedActivationRvv, UnsupportedActivationLeavesBufferUntouched) {
  if (GetMlasPlatform().MlasActivationOverride == nullptr) {
    GTEST_SKIP() << "RVV is unavailable or forced off";
  }
  for (auto kind : {MlasTanhActivation, MlasLogisticActivation}) {
    auto activation = MakeActivation(kind);
    std::array<float, 9> buffer = {-4.0f, -1.0f, 0.0f, 1.0f, 4.0f};
    const auto original = buffer;
    const float bias = 1.0f;
    EXPECT_FALSE(MlasActivationRvv(&activation, buffer.data(), &bias, 1, buffer.size(), buffer.size()));
    EXPECT_EQ(buffer, original);
  }
}
#endif

}  // namespace
