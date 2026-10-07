/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_gelu_erf_neon_fp16.cpp

Abstract:

    Tests for MLAS fp16 Gelu/Erf on ARM64 NEON.

--*/

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "test/mlas/unittest/test_util.h"
#include "core/mlas/lib/mlasi.h"

// MlasNeonGeluFP16Kernel/MlasNeonErfFP16Kernel are only built for non-Windows
// ARM64 (setup_mlas_source_for_windows() in onnxruntime_mlas.cmake does not
// add erf_neon_fp16.cpp/gelu_neon_fp16.cpp to the Windows ARM64 source list,
// even though MLAS_F16VEC_INTRINSICS_SUPPORTED is defined there), so this
// whole test is excluded on _WIN32 to avoid an unresolved external symbol.
// The two headers below unconditionally include <arm_neon.h> and so must
// stay inside this guard to avoid breaking non-ARM64 builds.
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64) && !defined(_WIN32)

#include "core/mlas/lib/gelu_neon_fp16.h"
#include "core/mlas/lib/erf_neon_fp16.h"

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
const size_t kLaneShapes[] = {1, 7, 8, 9, 16, 17};
// Every offset within an 8-wide vector chunk (0-7), plus the first scalar-tail
// lane (8) and the last lane of the buffer, so a lane-dependent regression
// cannot hide at any position within a chunk or in the tail.
const size_t kLanePositions[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

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

float RefErf(float x) {
  return std::erf(x);
}

float RefGeluErf(float x) {
  return 0.5f * x * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2)));
}

float RefGeluTanh(float x) {
  float inner = x * (0.7978845608028654f + 0.035677408136300125f * x * x);
  inner = std::max(-5.0f, std::min(5.0f, inner));
  // The kernel always round-trips tanh(inner) through fp16 (MlasComputeTanh<MLAS_FP16>
  // on both the vector and scalar paths) before combining with x, so tanh(+-5) rounds
  // to exactly +-1.0 in fp16 even though it is not exactly +-1.0 in float32. Reproduce
  // that rounding here, or extreme x (e.g. -65504, -Inf) diverges from the real kernel.
  float t = static_cast<float>(MLAS_FP16(std::tanh(inner)));
  return 0.5f * x * (1.0f + t);
}

}  // namespace

class MlasNeonErfFP16Test : public MlasTestBase {
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
    MlasNeonErfFP16Kernel(input, output, N);
    std::vector<float> ref(N);
    for (size_t i = 0; i < N; ++i) ref[i] = RefErf(input[i].ToFloat());
    Check(output, ref, N, "TestErfRandom");
  }

  void TestErfFixed(size_t N, float value) {
    auto* input = input_.GetFilledBuffer(N, [value](MLAS_FP16* buffer, size_t count) {
      std::fill_n(buffer, count, MLAS_FP16(value));
    });
    auto* output = output_.GetBuffer(N, true);
    MlasNeonErfFP16Kernel(input, output, N);
    std::vector<float> ref(N, RefErf(value));
    Check(output, ref, N, "TestErfFixed");
  }

  void TestErfLane(size_t N, size_t pos, float value) {
    auto* input = input_.GetFilledBuffer(N, [](MLAS_FP16* buffer, size_t count) {
      std::fill_n(buffer, count, MLAS_FP16(0.5f));
    });
    input[pos] = MLAS_FP16(value);
    auto* output = output_.GetBuffer(N, true);
    MlasNeonErfFP16Kernel(input, output, N);
    std::vector<float> ref(N, RefErf(0.5f));
    ref[pos] = RefErf(value);
    Check(output, ref, N, "TestErfLane");
  }

 public:
  MlasNeonErfFP16Test()
      : seed_(20260916), gen_(seed_), distrib_(-8.f, 8.f) {
  }

  static const char* GetTestSuiteName() {
    return "NeonErfFP16";
  }

  void ExecuteShort(void) override {
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
        // kLanePositions already covers every N-1 <= 8; only the two largest
        // shapes (16, 17) need the last-lane case tested separately.
        if (N - 1 > 8) {
          TestErfLane(N, N - 1, value);
        }
      }
    }
  }
};

// Note: a NaN input to this class's Fixed/Lane cases always yields a NaN
// output regardless of whether the internal erf/tanh step propagates NaN
// correctly or (as with the bug fixed in erf_neon_fp16.cpp) leaks +1.0,
// since the final combine multiplies by the original x, which is itself
// NaN. The erf-NaN-propagation invariant is exercised directly by
// MlasNeonErfFP16Test, not by this class.
class MlasNeonGeluFP16Test : public MlasTestBase {
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
    MlasNeonGeluFP16Kernel(input, output, temp, N, algo);
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
    MlasNeonGeluFP16Kernel(input, output, temp, N, algo);
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
    MlasNeonGeluFP16Kernel(input, output, temp, N, algo);
    std::vector<float> ref(N, Ref(0.5f, algo));
    ref[pos] = Ref(value, algo);
    Check(output, ref, N, "TestGeluLane");
  }

 public:
  MlasNeonGeluFP16Test()
      : seed_(20260917), gen_(seed_), distrib_(-8.f, 8.f) {
  }

  static const char* GetTestSuiteName() {
    return "NeonGeluFP16";
  }

  void ExecuteShort(void) override {
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
          // kLanePositions already covers every N-1 <= 8; only the two largest
          // shapes (16, 17) need the last-lane case tested separately.
          if (N - 1 > 8) {
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
    count += MlasDirectShortExecuteTests<MlasNeonErfFP16Test>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasNeonGeluFP16Test>::RegisterShortExecute();
  }
  return count;
});

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64) && !defined(_WIN32)
