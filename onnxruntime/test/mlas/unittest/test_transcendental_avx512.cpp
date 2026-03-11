// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_util.h"
#include "core/mlas/lib/mlasi.h"

#include <array>
#include <vector>

#if defined(MLAS_TARGET_AMD64)

namespace {

using UnaryKernel = MLAS_COMPUTE_UNARY_FLOAT_KERNEL;

bool IsAvx512Available() {
  return GetMlasPlatform().Avx512Supported_;
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

std::vector<float> GetLogisticSpecialValues() {
  const float lower = -18.0f;
  const float upper = 18.0f;
  return {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      0.0f,
      -0.0f,
      lower,
      std::nextafter(lower, -std::numeric_limits<float>::infinity()),
      std::nextafter(lower, std::numeric_limits<float>::infinity()),
      upper,
      std::nextafter(upper, -std::numeric_limits<float>::infinity()),
      std::nextafter(upper, std::numeric_limits<float>::infinity()),
      -1.0f,
      1.0f,
      -8.0f,
      8.0f,
      -20.0f,
      20.0f,
  };
}

std::vector<float> GetErfSpecialValues() {
  const float split = 0.921875f;
  const float upper_abs = 3.925f;
  return {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      0.0f,
      -0.0f,
      split,
      std::nextafter(split, -std::numeric_limits<float>::infinity()),
      std::nextafter(split, std::numeric_limits<float>::infinity()),
      -split,
      std::nextafter(-split, -std::numeric_limits<float>::infinity()),
      std::nextafter(-split, std::numeric_limits<float>::infinity()),
      upper_abs,
      std::nextafter(upper_abs, -std::numeric_limits<float>::infinity()),
      std::nextafter(upper_abs, std::numeric_limits<float>::infinity()),
      -upper_abs,
      std::nextafter(-upper_abs, -std::numeric_limits<float>::infinity()),
      std::nextafter(-upper_abs, std::numeric_limits<float>::infinity()),
      -1.0f,
      1.0f,
      -6.0f,
      6.0f,
  };
}

std::vector<float> GetGeluSpecialValues() {
  return {
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
}

std::vector<float> GetSiluSpecialValues() {
  return {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
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
}

float ComputeReferenceSilu(float x) {
  if (std::isnan(x)) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  if (x == std::numeric_limits<float>::infinity()) {
    return x;
  }

  if (x == -std::numeric_limits<float>::infinity()) {
    return -0.0f;
  }

  return x / (1.0f + std::exp(-x));
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

void CompareUnaryKernelAgainstGeneric(const char* kernel_name,
                                      UnaryKernel* generic_kernel,
                                      UnaryKernel* fma3_kernel,
                                      UnaryKernel* avx512_kernel,
                                      float minimum_value,
                                      float maximum_value,
                                      float absolute_tolerance,
                                      float relative_tolerance,
                                      bool check_signed_zero,
                                      const std::vector<float>& special_values,
                                      const std::vector<size_t>& sizes,
                                      size_t iterations,
                                      MatrixGuardBuffer<float>& input_buffer,
                                      MatrixGuardBuffer<float>& generic_output_buffer,
                                      MatrixGuardBuffer<float>& fma3_output_buffer,
                                      MatrixGuardBuffer<float>& avx512_output_buffer) {
  for (size_t size : sizes) {
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
      float* input = input_buffer.GetBuffer(size);
      float* generic_output = generic_output_buffer.GetBuffer(size);
      float* fma3_output = fma3_output_buffer.GetBuffer(size);
      float* avx512_output = avx512_output_buffer.GetBuffer(size);

      FillInput(input, size, minimum_value, maximum_value, special_values,
                static_cast<uint32_t>(size * 131u + iteration * 977u));

      generic_kernel(input, generic_output, size);
      fma3_kernel(input, fma3_output, size);
      avx512_kernel(input, avx512_output, size);

      for (size_t i = 0; i < size; ++i) {
        ASSERT_TRUE(UnaryOutputsMatch(avx512_output[i], generic_output[i], absolute_tolerance, relative_tolerance,
                                      check_signed_zero))
            << kernel_name << " mismatch at index " << i << " of " << size
            << ", input=" << input[i]
            << ", avx512=" << avx512_output[i]
            << ", generic=" << generic_output[i]
            << ", abs_diff=" << std::fabs(avx512_output[i] - generic_output[i]);

        ASSERT_TRUE(UnaryOutputsMatch(avx512_output[i], fma3_output[i], absolute_tolerance, relative_tolerance,
                                      check_signed_zero))
            << kernel_name << " FMA3 mismatch at index " << i << " of " << size
            << ", input=" << input[i]
            << ", avx512=" << avx512_output[i]
            << ", fma3=" << fma3_output[i]
            << ", abs_diff=" << std::fabs(avx512_output[i] - fma3_output[i]);
      }
    }
  }
}

class MlasComputeGeluAvx512Test : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> input_buffer_;
  MatrixGuardBuffer<float> generic_output_buffer_;
  MatrixGuardBuffer<float> avx512_output_buffer_;

  void ExecuteCommon(const std::vector<size_t>& sizes, size_t iterations) {
    if (!IsAvx512Available()) {
      GTEST_SKIP() << "AVX512 is not available on this machine.";
    }

    for (size_t size : sizes) {
      for (size_t iteration = 0; iteration < iterations; ++iteration) {
        float* input = input_buffer_.GetBuffer(size);
        float* generic_output = generic_output_buffer_.GetBuffer(size);
        float* avx512_output = avx512_output_buffer_.GetBuffer(size);

        FillInput(input, size, -10.0f, 10.0f, GetGeluSpecialValues(),
                  static_cast<uint32_t>(size * 131u + iteration * 977u + 17u));

        MlasGeluKernel(input, generic_output, size);
        MlasGeluKernelAvx512F(input, avx512_output, size);

        for (size_t i = 0; i < size; ++i) {
          ASSERT_TRUE(UnaryOutputsMatch(avx512_output[i], generic_output[i], 2e-6f, 2e-5f, true))
              << "Gelu mismatch at index " << i << " of " << size
              << ", input=" << input[i]
              << ", avx512=" << avx512_output[i]
              << ", generic=" << generic_output[i]
              << ", abs_diff=" << std::fabs(avx512_output[i] - generic_output[i]);
        }
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    return "TranscendentalAvx512Gelu";
  }

  void ExecuteShort() override {
    ExecuteCommon({1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255}, 3);
  }

  void ExecuteLong() override {
    ExecuteCommon({1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
                   511, 512, 513, 1023, 1024, 1025, 4095},
                  8);
  }
};

class MlasComputeSiluAvx512Test : public MlasTestBase {
 private:
  MatrixGuardBuffer<float> input_buffer_;
  MatrixGuardBuffer<float> avx512_output_buffer_;

  void ExecuteCommon(const std::vector<size_t>& sizes, size_t iterations) {
    if (!IsAvx512Available()) {
      GTEST_SKIP() << "AVX512 is not available on this machine.";
    }

    for (size_t size : sizes) {
      for (size_t iteration = 0; iteration < iterations; ++iteration) {
        float* input = input_buffer_.GetBuffer(size);
        float* avx512_output = avx512_output_buffer_.GetBuffer(size);

        FillInput(input, size, -20.0f, 20.0f, GetSiluSpecialValues(),
                  static_cast<uint32_t>(size * 149u + iteration * 991u + 31u));

        MlasSiluKernelAvx512F(input, avx512_output, size);

        for (size_t i = 0; i < size; ++i) {
          const float expected = ComputeReferenceSilu(input[i]);
          ASSERT_TRUE(UnaryOutputsMatch(avx512_output[i], expected, 3e-5f, 5e-5f, true))
              << "Silu mismatch at index " << i << " of " << size
              << ", input=" << input[i]
              << ", avx512=" << avx512_output[i]
              << ", expected=" << expected
              << ", abs_diff=" << std::fabs(avx512_output[i] - expected);
        }
      }
    }
  }

 public:
  static const char* GetTestSuiteName() {
    return "TranscendentalAvx512Silu";
  }

  void ExecuteShort() override {
    ExecuteCommon({1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255}, 3);
  }

  void ExecuteLong() override {
    ExecuteCommon({1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
                   511, 512, 513, 1023, 1024, 1025, 4095},
                  8);
  }
};

}  // namespace

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  size_t count = 0;
  if (is_short_execute) {
    count += MlasDirectShortExecuteTests<MlasComputeGeluAvx512Test>::RegisterShortExecute();
    count += MlasDirectShortExecuteTests<MlasComputeSiluAvx512Test>::RegisterShortExecute();
  } else {
    count += MlasLongExecuteTests<MlasComputeGeluAvx512Test>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasComputeSiluAvx512Test>::RegisterLongExecute();
  }
  return count;
});

#else

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool) {
  return size_t{0};
});

#endif
