// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cmath>
#include <random>
#include "gtest/gtest.h"
#include "core/framework/op_kernel.h"
#include "core/providers/telum/telum_common.h"
#include "core/providers/telum/utils/endian_utils.h"

namespace onnxruntime {
namespace test {
namespace telum {

/**
 * @brief Test utilities for Telum EP testing
 */

// Tolerance for floating point comparisons
constexpr float kDefaultTolerance = 1e-5f;
constexpr float kRelaxedTolerance = 1e-4f;

/**
 * @brief Check if zDNN is available for testing
 */
inline bool IsZDNNAvailable() {
  return onnxruntime::telum::IsZDNNAvailable();
}

/**
 * @brief Skip test if zDNN is not available
 */
#define SKIP_IF_NO_ZDNN() \
  if (!IsZDNNAvailable()) { \
    GTEST_SKIP() << "zDNN/NNPA not available on this system"; \
  }

/**
 * @brief Verify endianness is correctly detected
 */
inline void VerifyEndianness() {
  ASSERT_TRUE(onnxruntime::telum::VerifyEndianness())
      << "Endianness verification failed";

  // Log endianness for debugging
  std::cout << "Running on " << onnxruntime::telum::GetEndiannessString()
            << " architecture" << std::endl;
}

/**
 * @brief Compare two floating point values with tolerance
 */
inline bool AlmostEqual(float a, float b, float tolerance = kDefaultTolerance) {
  if (std::isnan(a) && std::isnan(b)) return true;
  if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
  return std::abs(a - b) <= tolerance;
}

/**
 * @brief Compare two vectors of floats with tolerance
 */
inline bool VectorsAlmostEqual(const std::vector<float>& a,
                               const std::vector<float>& b,
                               float tolerance = kDefaultTolerance) {
  if (a.size() != b.size()) return false;

  for (size_t i = 0; i < a.size(); ++i) {
    if (!AlmostEqual(a[i], b[i], tolerance)) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i]
                << " (diff: " << std::abs(a[i] - b[i]) << ")" << std::endl;
      return false;
    }
  }
  return true;
}

/**
 * @brief Generate random float vector
 */
inline std::vector<float> GenerateRandomFloats(size_t count,
                                               float min = -1.0f,
                                               float max = 1.0f,
                                               int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(min, max);

  std::vector<float> result(count);
  for (size_t i = 0; i < count; ++i) {
    result[i] = dist(gen);
  }
  return result;
}

/**
 * @brief Generate sequential float vector for deterministic testing
 */
inline std::vector<float> GenerateSequentialFloats(size_t count,
                                                   float start = 0.0f,
                                                   float step = 1.0f) {
  std::vector<float> result(count);
  for (size_t i = 0; i < count; ++i) {
    result[i] = start + i * step;
  }
  return result;
}

/**
 * @brief Create a simple test tensor shape
 */
inline TensorShape MakeShape(const std::vector<int64_t>& dims) {
  return TensorShape(dims);
}

/**
 * @brief Compute expected output for MatMul: C = A × B
 */
inline std::vector<float> ComputeMatMulReference(
    const std::vector<float>& A, const std::vector<float>& B,
    int64_t M, int64_t K, int64_t N) {

  std::vector<float> C(M * N, 0.0f);

  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }

  return C;
}

/**
 * @brief Compute expected output for Gemm: Y = alpha*A*B + beta*C
 */
inline std::vector<float> ComputeGemmReference(
    const std::vector<float>& A, const std::vector<float>& B,
    const std::vector<float>& C,
    int64_t M, int64_t K, int64_t N,
    float alpha = 1.0f, float beta = 1.0f) {

  // First compute A × B
  auto AB = ComputeMatMulReference(A, B, M, K, N);

  // Then apply alpha*AB + beta*C
  std::vector<float> Y(M * N);
  for (size_t i = 0; i < Y.size(); ++i) {
    Y[i] = alpha * AB[i] + beta * C[i];
  }

  return Y;
}

/**
 * @brief Compute expected output for elementwise Add
 */
inline std::vector<float> ComputeAddReference(
    const std::vector<float>& A, const std::vector<float>& B) {

  EXPECT_EQ(A.size(), B.size());
  std::vector<float> C(A.size());
  for (size_t i = 0; i < A.size(); ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

/**
 * @brief Compute expected output for Relu
 */
inline std::vector<float> ComputeReluReference(const std::vector<float>& X) {
  std::vector<float> Y(X.size());
  for (size_t i = 0; i < X.size(); ++i) {
    Y[i] = std::max(0.0f, X[i]);
  }
  return Y;
}

/**
 * @brief Compute expected output for Gelu (approximation)
 */
inline std::vector<float> ComputeGeluReference(const std::vector<float>& X) {
  std::vector<float> Y(X.size());
  constexpr float sqrt_2_over_pi = 0.7978845608f;
  constexpr float coeff = 0.044715f;

  for (size_t i = 0; i < X.size(); ++i) {
    float x = X[i];
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
    Y[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
  }
  return Y;
}

/**
 * @brief Compute expected output for Tanh
 */
inline std::vector<float> ComputeTanhReference(const std::vector<float>& X) {
  std::vector<float> Y(X.size());
  for (size_t i = 0; i < X.size(); ++i) {
    Y[i] = std::tanh(X[i]);
  }
  return Y;
}

/**
 * @brief Test fixture for Telum EP tests
 */
class TelumTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    // Verify endianness
    VerifyEndianness();

    // Skip if zDNN not available
    SKIP_IF_NO_ZDNN();
  }

  void TearDown() override {
    // Cleanup if needed
  }
};

}  // namespace telum
}  // namespace test
}  // namespace onnxruntime

// Made with Bob
