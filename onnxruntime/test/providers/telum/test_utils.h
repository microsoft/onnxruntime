// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include "gtest/gtest.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_options.h"
#include "core/providers/telum/telum_common.h"
#include "core/providers/telum/utils/endian_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/default_providers.h"

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

  // Then apply alpha*AB + beta*C, with basic broadcasting support for C.
  // The tests in this folder use a subset of ONNX Gemm broadcast patterns:
  // - C empty: treat as 0
  // - C size 1: scalar
  // - C size N: bias vector (broadcast across rows)
  // - C size M*N: full bias matrix
  // - C size M with N==1: column bias
  auto get_c = [&](int64_t i, int64_t j) -> float {
    if (C.empty() || beta == 0.0f) return 0.0f;
    if (C.size() == 1) return C[0];
    if (C.size() == static_cast<size_t>(N)) return C[static_cast<size_t>(j)];
    if (N == 1 && C.size() == static_cast<size_t>(M)) return C[static_cast<size_t>(i)];
    if (C.size() == static_cast<size_t>(M * N)) return C[static_cast<size_t>(i * N + j)];

    throw std::runtime_error("ComputeGemmReference: unsupported C broadcast pattern");
  };

  std::vector<float> Y(M * N);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      const size_t idx = static_cast<size_t>(i * N + j);
      Y[idx] = alpha * AB[idx] + beta * get_c(i, j);
    }
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
 * @brief Compute expected output for Softmax over the last dimension.
 *
 * Interprets X as a 2D matrix with shape [outer, inner] in row-major order and applies softmax per row.
 */
inline std::vector<float> ComputeSoftmaxLastDimReference(
    const std::vector<float>& X, int64_t outer, int64_t inner) {
  if (outer < 0 || inner < 0) {
    throw std::runtime_error("ComputeSoftmaxLastDimReference: negative shape");
  }
  const size_t expected_size = static_cast<size_t>(outer) * static_cast<size_t>(inner);
  if (X.size() != expected_size) {
    throw std::runtime_error("ComputeSoftmaxLastDimReference: size mismatch");
  }

  std::vector<float> Y(X.size());
  for (int64_t i = 0; i < outer; ++i) {
    const size_t row = static_cast<size_t>(i) * static_cast<size_t>(inner);
    float maxv = X[row];
    for (int64_t j = 1; j < inner; ++j) {
      maxv = std::max(maxv, X[row + static_cast<size_t>(j)]);
    }
    float sum = 0.0f;
    for (int64_t j = 0; j < inner; ++j) {
      const float e = std::exp(X[row + static_cast<size_t>(j)] - maxv);
      Y[row + static_cast<size_t>(j)] = e;
      sum += e;
    }
    for (int64_t j = 0; j < inner; ++j) {
      Y[row + static_cast<size_t>(j)] /= sum;
    }
  }
  return Y;
}

struct LayerNormReferenceResult {
  std::vector<float> Y;       // size N*C
  std::vector<float> Mean;    // size N
  std::vector<float> InvStd;  // size N
};

/**
 * @brief Compute expected output for LayerNormalization over the last dimension.
 *
 * Interprets X as [N, C] (row-major), Scale as [C], Bias as either empty (no bias) or [C].
 */
inline LayerNormReferenceResult ComputeLayerNormLastDimReference(
    const std::vector<float>& X,
    const std::vector<float>& Scale,
    const std::vector<float>& Bias,
    int64_t N, int64_t C,
    float epsilon) {
  if (N < 0 || C < 0) {
    throw std::runtime_error("ComputeLayerNormLastDimReference: negative shape");
  }
  if (Scale.size() != static_cast<size_t>(C)) {
    throw std::runtime_error("ComputeLayerNormLastDimReference: scale size mismatch");
  }
  if (!Bias.empty() && Bias.size() != static_cast<size_t>(C)) {
    throw std::runtime_error("ComputeLayerNormLastDimReference: bias size mismatch");
  }
  const size_t expected_size = static_cast<size_t>(N) * static_cast<size_t>(C);
  if (X.size() != expected_size) {
    throw std::runtime_error("ComputeLayerNormLastDimReference: X size mismatch");
  }

  LayerNormReferenceResult r;
  r.Y.resize(expected_size);
  r.Mean.resize(static_cast<size_t>(N));
  r.InvStd.resize(static_cast<size_t>(N));

  for (int64_t i = 0; i < N; ++i) {
    const size_t base = static_cast<size_t>(i) * static_cast<size_t>(C);

    float mean = 0.0f;
    for (int64_t j = 0; j < C; ++j) {
      mean += X[base + static_cast<size_t>(j)];
    }
    mean /= static_cast<float>(C);

    float var = 0.0f;
    for (int64_t j = 0; j < C; ++j) {
      const float d = X[base + static_cast<size_t>(j)] - mean;
      var += d * d;
    }
    var /= static_cast<float>(C);

    const float inv_std = 1.0f / std::sqrt(var + epsilon);

    r.Mean[static_cast<size_t>(i)] = mean;
    r.InvStd[static_cast<size_t>(i)] = inv_std;

    for (int64_t j = 0; j < C; ++j) {
      const float x = X[base + static_cast<size_t>(j)];
      float y = (x - mean) * inv_std;
      y *= Scale[static_cast<size_t>(j)];
      if (!Bias.empty()) y += Bias[static_cast<size_t>(j)];
      r.Y[base + static_cast<size_t>(j)] = y;
    }
  }

  return r;
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

  // Helper to run an OpTester on Telum EP. By default, disables fallback to CPU EP so we actually
  // execute the Telum kernels (and fail if partitioning didn't assign nodes to Telum).
  void RunOnTelum(OpTester& test, bool disable_cpu_ep_fallback = true) {
    onnxruntime::SessionOptions so;
    if (disable_cpu_ep_fallback) {
      ASSERT_TRUE(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1").IsOK());
    }

    test.Config(so);
    test.ConfigEp(onnxruntime::test::DefaultTelumExecutionProvider());
    test.RunWithConfig();
  }
};

}  // namespace telum
}  // namespace test
}  // namespace onnxruntime

// Made with Bob
