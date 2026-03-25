// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"

#include <array>
#include <vector>

#if defined(MLAS_TARGET_AMD64)

namespace {

constexpr float kGeluMinValue = -10.0f;
constexpr float kGeluMaxValue = 10.0f;
constexpr float kSiluMinValue = -20.0f;
constexpr float kSiluMaxValue = 20.0f;

constexpr float kGeluAbsoluteTolerance = 2e-6f;
constexpr float kGeluRelativeTolerance = 2e-5f;
constexpr float kSiluAbsoluteTolerance = 3e-5f;
constexpr float kSiluRelativeTolerance = 5e-5f;

constexpr std::array<size_t, 20> kShortTestSizes = {
    1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255};

constexpr std::array<size_t, 27> kLongTestSizes = {
    1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63,
    64, 65, 127, 128, 129, 255, 511, 512, 513, 1023, 1024, 1025, 4095};

bool IsGeluErfAvx512Dispatched() {
  return GetMlasPlatform().GeluErfKernelRoutine == MlasGeluErfKernelAvx512F;
}

bool IsSiluAvx512Dispatched() {
  return GetMlasPlatform().SiluKernelRoutine == MlasSiluKernelAvx512F;
}

bool UnaryOutputsMatch(float actual, float expected, float absolute_tolerance, float relative_tolerance,
                       bool check_signed_zero) {
  if (std::isnan(expected)) {
    return std::isnan(actual);
  }

  if (std::isinf(expected)) {
    return std::isinf(actual) && (std::signbit(actual) == std::signbit(expected));
  }

  if (check_signed_zero && actual == 0.0f && expected == 0.0f) {
    return std::signbit(actual) == std::signbit(expected);
  }

  const float diff = std::fabs(actual - expected);
  if (diff <= absolute_tolerance) {
    return true;
  }

  const float scale = std::max(std::fabs(actual), std::fabs(expected));
  return scale > 0.0f && diff <= scale * relative_tolerance;
}

const std::vector<float>& GetGeluSpecialValues() {
  static const std::vector<float> values = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      0.0f,
      -0.0f,
      -10.0f,
      -6.0f,
      -3.0f,
      -1.0f,
      -0.5f,
      0.5f,
      1.0f,
      3.0f,
      6.0f,
      10.0f,
  };

  return values;
}

const std::vector<float>& GetSiluSpecialValues() {
  static const std::vector<float> values = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(),
      1.0e9f,
      -1.0e9f,
      0.0f,
      -0.0f,
      -20.0f,
      -10.0f,
      -6.0f,
      -3.0f,
      -1.0f,
      -0.5f,
      0.5f,
      1.0f,
      3.0f,
      6.0f,
      10.0f,
      20.0f,
  };

  return values;
}

void FillInput(float* input, size_t n, float minimum_value, float maximum_value,
               const std::vector<float>& special_values, uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(minimum_value, maximum_value);

  for (size_t i = 0; i < n; ++i) {
    input[i] = distribution(generator);
  }

  const size_t special_count = std::min(n, special_values.size());
  for (size_t i = 0; i < special_count; ++i) {
    input[i] = special_values[i];
  }
}

class MlasComputeGeluErfAvx512Test : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> input_buffer_;
  MatrixGuardBuffer<float> generic_output_buffer_;
  MatrixGuardBuffer<float> public_output_buffer_;
  MatrixGuardBuffer<float> avx512_output_buffer_;

  void ExecuteCommon(const std::vector<size_t>& sizes, size_t iterations) {
    if (!IsGeluErfAvx512Dispatched()) {
      GTEST_SKIP() << "AVX512F GELU(erf) dispatch is not available on this machine.";
    }

    for (size_t size : sizes) {
      for (size_t iteration = 0; iteration < iterations; ++iteration) {
        float* input = input_buffer_.GetBuffer(size);
        float* generic_output = generic_output_buffer_.GetBuffer(size);
        float* public_output = public_output_buffer_.GetBuffer(size);
        float* avx512_output = avx512_output_buffer_.GetBuffer(size);

        FillInput(input, size, kGeluMinValue, kGeluMaxValue, GetGeluSpecialValues(),
                  static_cast<uint32_t>(size * 131u + iteration * 977u + 17u));

        MlasGeluErfKernel(input, generic_output, size);
        MlasComputeGeluErf(input, public_output, size);
        MlasGeluErfKernelAvx512F(input, avx512_output, size);

        for (size_t i = 0; i < size; ++i) {
          ASSERT_TRUE(UnaryOutputsMatch(public_output[i], generic_output[i],
                                        kGeluAbsoluteTolerance, kGeluRelativeTolerance, true))
              << "Public GELU(erf) mismatch at index " << i << " of " << size
              << ", input=" << input[i]
              << ", public=" << public_output[i]
              << ", generic=" << generic_output[i]
              << ", abs_diff=" << std::fabs(public_output[i] - generic_output[i]);

          ASSERT_TRUE(UnaryOutputsMatch(avx512_output[i], generic_output[i],
                                        kGeluAbsoluteTolerance, kGeluRelativeTolerance, true))
              << "GELU(erf) mismatch at index " << i << " of " << size
              << ", input=" << input[i]
              << ", avx512=" << avx512_output[i]
              << ", generic=" << generic_output[i]
              << ", abs_diff=" << std::fabs(avx512_output[i] - generic_output[i]);

          ASSERT_TRUE(UnaryOutputsMatch(avx512_output[i], public_output[i],
                                        kGeluAbsoluteTolerance, kGeluRelativeTolerance, true))
              << "Public/API GELU(erf) dispatch mismatch at index " << i << " of " << size
              << ", input=" << input[i]
              << ", avx512=" << avx512_output[i]
              << ", public=" << public_output[i]
              << ", abs_diff=" << std::fabs(avx512_output[i] - public_output[i]);
        }
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    return "TranscendentalAvx512Gelu";
  }

  void ExecuteShort() override {
    ExecuteCommon(std::vector<size_t>(kShortTestSizes.begin(), kShortTestSizes.end()), 3);
  }

  void ExecuteLong() override {
    ExecuteCommon(std::vector<size_t>(kLongTestSizes.begin(), kLongTestSizes.end()), 8);
  }
};

class MlasComputeSiluAvx512Test : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> input_buffer_;
  MatrixGuardBuffer<float> generic_output_buffer_;
  MatrixGuardBuffer<float> public_output_buffer_;
  MatrixGuardBuffer<float> avx512_output_buffer_;

  void ExecuteCommon(const std::vector<size_t>& sizes, size_t iterations) {
    if (!IsSiluAvx512Dispatched()) {
      GTEST_SKIP() << "AVX512F SiLU dispatch is not available on this machine.";
    }

    for (size_t size : sizes) {
      for (size_t iteration = 0; iteration < iterations; ++iteration) {
        float* input = input_buffer_.GetBuffer(size);
        float* generic_output = generic_output_buffer_.GetBuffer(size);
        float* public_output = public_output_buffer_.GetBuffer(size);
        float* avx512_output = avx512_output_buffer_.GetBuffer(size);

        FillInput(input, size, kSiluMinValue, kSiluMaxValue, GetSiluSpecialValues(),
                  static_cast<uint32_t>(size * 149u + iteration * 991u + 31u));

        MlasSiluKernel(input, generic_output, size);
        MlasComputeSilu(input, public_output, size);
        MlasSiluKernelAvx512F(input, avx512_output, size);

        for (size_t i = 0; i < size; ++i) {
          ASSERT_TRUE(UnaryOutputsMatch(public_output[i], generic_output[i],
                                        kSiluAbsoluteTolerance, kSiluRelativeTolerance, true))
              << "Public Silu mismatch at index " << i << " of " << size
              << ", input=" << input[i]
              << ", public=" << public_output[i]
              << ", generic=" << generic_output[i]
              << ", abs_diff=" << std::fabs(public_output[i] - generic_output[i]);

          ASSERT_TRUE(UnaryOutputsMatch(avx512_output[i], generic_output[i],
                                        kSiluAbsoluteTolerance, kSiluRelativeTolerance, true))
              << "Silu mismatch at index " << i << " of " << size
              << ", input=" << input[i]
              << ", avx512=" << avx512_output[i]
              << ", generic=" << generic_output[i]
              << ", abs_diff=" << std::fabs(avx512_output[i] - generic_output[i]);

          ASSERT_TRUE(UnaryOutputsMatch(avx512_output[i], public_output[i],
                                        kSiluAbsoluteTolerance, kSiluRelativeTolerance, true))
              << "Public/API Silu dispatch mismatch at index " << i << " of " << size
              << ", input=" << input[i]
              << ", avx512=" << avx512_output[i]
              << ", public=" << public_output[i]
              << ", abs_diff=" << std::fabs(avx512_output[i] - public_output[i]);
        }
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    return "TranscendentalAvx512Silu";
  }

  void ExecuteShort() override {
    ExecuteCommon(std::vector<size_t>(kShortTestSizes.begin(), kShortTestSizes.end()), 3);
  }

  void ExecuteLong() override {
    ExecuteCommon(std::vector<size_t>(kLongTestSizes.begin(), kLongTestSizes.end()), 8);
  }
};

}  // namespace

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasComputeGeluErfAvx512Test>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasComputeSiluAvx512Test>::RegisterShortExecute();
  } else {
    count += MlasLongExecuteTests<MlasComputeGeluErfAvx512Test>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasComputeSiluAvx512Test>::RegisterLongExecute();
  }
  return count;
});

#else

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool) {
  return size_t{0};
});

#endif
