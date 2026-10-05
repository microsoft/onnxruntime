/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_gelu_erf_sve_fp16.cpp

Abstract:

    Tests for MLAS fp16 Gelu/Erf on ARM64 SVE.

    The SVE counterpart of test_gelu_erf_neon_fp16.cpp. No test reached the SVE
    fp16 kernels before this one, which is how the two bugs fixed alongside it
    survived: the shipped frozen kernels returned +1.0 for a NaN erf input and
    overflowed Gelu to +Inf for x >= 32768.

    Two differences from the NEON test:

      * SVE vector length is a runtime property, not a compile-time constant, so
        lane positions are swept far past NEON's fixed 8 lanes. The cases below
        cover every offset within one vector for every vector length up to 512
        bits, plus the final lane of each buffer.

      * The SVE fp16 kernels are portable machine code (aarch64/elementwise_sve_asm.S)
        and are registered on every platform including Windows, so unlike the NEON
        test this one is not excluded on _WIN32. sve/mlasi_sve.h is deliberately
        free of SVE types and intrinsics, so including it cannot break a build
        whose toolchain has no SVE support.

--*/

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "test/mlas/unittest/test_util.h"
#include "core/mlas/lib/mlasi.h"

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64) && defined(MLAS_USE_SVE)

#include "core/mlas/lib/sve/mlasi_sve.h"

namespace {

constexpr float kFp16QuietNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kFp16Inf = std::numeric_limits<float>::infinity();

// Extreme fp16-representable values chosen to probe the erf saturation boundary
// (|x|==4), the fp16 overflow boundary of x*(1+t) for large positive x
// (x>=32768), subnormal/min-normal inputs, and Inf/NaN propagation. Negative
// counterparts of the saturation boundary and subnormal/min-normal values are
// included since erf applies sign separately from the |x| saturation check.
const float kExtremeValues[] = {
    0.0f, -0.0f, 5.9604645e-8f, -5.9604645e-8f, 6.103515625e-5f, -6.103515625e-5f,
    0.5f, 1.0f,
    3.99609375f, -3.99609375f, 4.0f, -4.0f, 4.00390625f, -4.00390625f,
    5.0f, 100.0f, 1000.0f,
    32752.0f, 32768.0f, 53504.0f, 65504.0f, -65504.0f,
    kFp16Inf, -kFp16Inf, kFp16QuietNaN};

const size_t kShapes[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 63, 127, 129, 1000};
// Shapes that straddle a vector boundary at 128, 256 and 512 bit vector lengths
// (8, 16 and 32 fp16 lanes), so the predicated tail is exercised on any machine.
const size_t kLaneShapes[] = {1, 7, 8, 9, 16, 17, 32, 33};
// Every offset within one vector for vector lengths up to 512 bits, plus the
// first lane past a 128-bit and a 256-bit vector. Guarded by pos < N below.
const size_t kLanePositions[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32};

MLAS_FORCEINLINE
bool FloatEqual(MLAS_FP16 v0, MLAS_FP16 v1, float rtol, float atol) {
  float f0 = v0.ToFloat(), f1 = v1.ToFloat();
  if (std::isinf(f0) || std::isinf(f1)) {
    return f0 == f1;
  }
  if (std::isnan(f0) || std::isnan(f1)) {
    return std::isnan(f0) && std::isnan(f1);
  }
  return std::abs(f0 - f1) <= std::abs(f1 * rtol) + atol;
}

bool SveUnavailable() {
  return !MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve();
}

float RefErf(float x) {
  return std::erf(x);
}

float RefGeluErf(float x) {
  return 0.5f * x * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2)));
}

float RefGeluTanh(float x) {
  float inner = x * (0.7978845608028654f + 0.035677408136300125f * x * x);
  inner = std::max(-5.0f, std::min(5.0f, inner));
  // The kernel always round-trips tanh(inner) through fp16 before combining with
  // x, so tanh(+-5) rounds to exactly +-1.0 in fp16 even though it is not exactly
  // +-1.0 in float32. Reproduce that rounding here, or extreme x (e.g. -65504,
  // -Inf) diverges from the real kernel.
  float t = static_cast<float>(MLAS_FP16(std::tanh(inner)));
  return 0.5f * x * (1.0f + t);
}

}  // namespace

class MlasSveErfFP16Test : public MlasTestBase {
 private:
  unsigned int seed_;
  std::mt19937 gen_;
  std::uniform_real_distribution<float> distrib_;
  MatrixGuardBuffer<MLAS_FP16> input_, output_;

  void Check(const MLAS_FP16* output, const std::vector<float>& ref, size_t N, const char* case_desc) {
    for (size_t i = 0; i < N; ++i) {
      ASSERT_TRUE(FloatEqual(output[i], MLAS_FP16(ref[i]), 0.02f, 0.01f))
          << case_desc << " i " << i << " N " << N
          << " value " << output[i] << " ref " << ref[i];
    }
  }

  void TestErfRandom(size_t N) {
    auto* input = input_.GetFilledBuffer(N, [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; ++i) buffer[i] = MLAS_FP16(distrib_(gen_));
    });
    auto* output = output_.GetBuffer(N, true);
    MlasSveErfFP16Kernel(input, output, N);
    std::vector<float> ref(N);
    for (size_t i = 0; i < N; ++i) ref[i] = RefErf(input[i].ToFloat());
    Check(output, ref, N, "TestErfRandom");
  }

  void TestErfFixed(size_t N, float value) {
    auto* input = input_.GetFilledBuffer(N, [value](MLAS_FP16* buffer, size_t count) {
      std::fill_n(buffer, count, MLAS_FP16(value));
    });
    auto* output = output_.GetBuffer(N, true);
    MlasSveErfFP16Kernel(input, output, N);
    std::vector<float> ref(N, RefErf(value));
    Check(output, ref, N, "TestErfFixed");
  }

  void TestErfLane(size_t N, size_t pos, float value) {
    auto* input = input_.GetFilledBuffer(N, [](MLAS_FP16* buffer, size_t count) {
      std::fill_n(buffer, count, MLAS_FP16(0.5f));
    });
    input[pos] = MLAS_FP16(value);
    auto* output = output_.GetBuffer(N, true);
    MlasSveErfFP16Kernel(input, output, N);
    std::vector<float> ref(N, RefErf(0.5f));
    ref[pos] = RefErf(value);
    Check(output, ref, N, "TestErfLane");
  }

 public:
  MlasSveErfFP16Test()
      : seed_(20261001), gen_(seed_), distrib_(-8.f, 8.f) {
  }

  static const char* GetTestSuiteName() {
    return "SveErfFP16";
  }

  void ExecuteShort(void) override {
    if (SveUnavailable()) {
      GTEST_SKIP() << "MlasSveErfFP16Kernel requires ARM64 SVE but it was not detected.";
    }

    for (size_t N : kShapes) {
      TestErfRandom(N);
    }

    for (size_t N : kLaneShapes) {
      for (float value : kExtremeValues) {
        TestErfFixed(N, value);
        for (size_t pos : kLanePositions) {
          if (pos < N) {
            TestErfLane(N, pos, value);
          }
        }
        if (N > 0) {
          TestErfLane(N, N - 1, value);
        }
      }
    }
  }
};

// Note: a NaN input to this class's Fixed/Lane cases always yields a NaN output
// regardless of whether the internal erf/tanh step propagates NaN correctly or
// (as with the bug fixed in elementwise_sve_fp16.cpp) leaks +1.0, since the
// final combine multiplies by the original x, which is itself NaN. The
// erf-NaN-propagation invariant is exercised directly by MlasSveErfFP16Test,
// not by this class. The Gelu overflow invariant, by contrast, is exercised
// here: x >= 32768 with the old associativity returned +Inf.
class MlasSveGeluFP16Test : public MlasTestBase {
 private:
  unsigned int seed_;
  std::mt19937 gen_;
  std::uniform_real_distribution<float> distrib_;
  MatrixGuardBuffer<MLAS_FP16> input_, output_, temp_;

  static float Ref(float x, MLAS_GELU_ALGORITHM algo) {
    return algo == MlasGeluTanh ? RefGeluTanh(x) : RefGeluErf(x);
  }

  void Check(const MLAS_FP16* output, const std::vector<float>& ref, size_t N, const char* case_desc) {
    for (size_t i = 0; i < N; ++i) {
      ASSERT_TRUE(FloatEqual(output[i], MLAS_FP16(ref[i]), 0.02f, 0.01f))
          << case_desc << " i " << i << " N " << N
          << " value " << output[i] << " ref " << ref[i];
    }
  }

  void TestGeluRandom(size_t N, MLAS_GELU_ALGORITHM algo) {
    auto* input = input_.GetFilledBuffer(N, [this](MLAS_FP16* buffer, size_t count) {
      for (size_t i = 0; i < count; ++i) buffer[i] = MLAS_FP16(distrib_(gen_));
    });
    auto* output = output_.GetBuffer(N, true);
    auto* temp = temp_.GetBuffer(N, true);
    MlasSveGeluFP16Kernel(input, output, temp, N, algo);
    std::vector<float> ref(N);
    for (size_t i = 0; i < N; ++i) ref[i] = Ref(input[i].ToFloat(), algo);
    Check(output, ref, N, "TestGeluRandom");
  }

  void TestGeluFixed(size_t N, float value, MLAS_GELU_ALGORITHM algo) {
    auto* input = input_.GetFilledBuffer(N, [value](MLAS_FP16* buffer, size_t count) {
      std::fill_n(buffer, count, MLAS_FP16(value));
    });
    auto* output = output_.GetBuffer(N, true);
    auto* temp = temp_.GetBuffer(N, true);
    MlasSveGeluFP16Kernel(input, output, temp, N, algo);
    std::vector<float> ref(N, Ref(value, algo));
    Check(output, ref, N, "TestGeluFixed");
  }

  void TestGeluLane(size_t N, size_t pos, float value, MLAS_GELU_ALGORITHM algo) {
    auto* input = input_.GetFilledBuffer(N, [](MLAS_FP16* buffer, size_t count) {
      std::fill_n(buffer, count, MLAS_FP16(0.5f));
    });
    input[pos] = MLAS_FP16(value);
    auto* output = output_.GetBuffer(N, true);
    auto* temp = temp_.GetBuffer(N, true);
    MlasSveGeluFP16Kernel(input, output, temp, N, algo);
    std::vector<float> ref(N, Ref(0.5f, algo));
    ref[pos] = Ref(value, algo);
    Check(output, ref, N, "TestGeluLane");
  }

 public:
  MlasSveGeluFP16Test()
      : seed_(20261002), gen_(seed_), distrib_(-8.f, 8.f) {
  }

  static const char* GetTestSuiteName() {
    return "SveGeluFP16";
  }

  void ExecuteShort(void) override {
    if (SveUnavailable()) {
      GTEST_SKIP() << "MlasSveGeluFP16Kernel requires ARM64 SVE but it was not detected.";
    }

    for (MLAS_GELU_ALGORITHM algo : {MlasGeluErf, MlasGeluTanh}) {
      for (size_t N : kShapes) {
        TestGeluRandom(N, algo);
      }

      for (size_t N : kLaneShapes) {
        for (float value : kExtremeValues) {
          TestGeluFixed(N, value, algo);
          for (size_t pos : kLanePositions) {
            if (pos < N) {
              TestGeluLane(N, pos, value, algo);
            }
          }
          if (N > 0) {
            TestGeluLane(N, N - 1, value, algo);
          }
        }
      }
    }
  }
};

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasSveErfFP16Test>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasSveGeluFP16Test>::RegisterShortExecute();
  }
  return count;
});

#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED && MLAS_TARGET_ARM64 && MLAS_USE_SVE
