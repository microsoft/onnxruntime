// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "gtest/gtest.h"
#include <vector>
#include <unordered_map>
#include <functional>
#include <random>
#include "core/mlas/inc/mlas.h"
#include "core/graph/constants.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
inline void TestActivationOp(const char* szOp, const std::vector<std::vector<T>>& input_vals_vec,
                             std::function<T(T)> expected_func,
                             const std::unordered_map<std::string, float> float_attribs = {},
                             const std::unordered_map<std::string, std::string> string_attribs = {},
                             bool is_tensorrt_supported = true, int opset_version = 7,
                             const char* domain = kOnnxDomain) {
  for (const std::vector<T>& input_vals : input_vals_vec) {
    OpTester test(szOp, opset_version, domain);

    for (auto attr : float_attribs) test.AddAttribute<float>(attr.first, attr.second);
    for (auto attr : string_attribs) test.AddAttribute(attr.first, attr.second);

    std::vector<int64_t> dims{(int64_t)input_vals.size()};

    std::vector<T> expected_vals;
    for (const auto& iv : input_vals) expected_vals.push_back(expected_func(iv));

    test.AddInput<T>("X", dims, input_vals);
    test.AddOutput<T>("Y", dims, expected_vals);

    // Disable TensorRT on unsupported tests
    std::unordered_set<std::string> excluded_providers;
    if (!is_tensorrt_supported) {
      excluded_providers.insert(kTensorrtExecutionProvider);
    }

// Disabled because of accuracy issues for GPU
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
    int leaky = strcmp(szOp, "LeakyRelu");
    if (leaky == 0) {
      excluded_providers.insert(kOpenVINOExecutionProvider);
    }
#endif

// Disabled because NNAPI and QNN EP (SDK 2.17) treat float::inf as float::max
#if defined(USE_NNAPI) || defined(USE_QNN)
    int relu = strcmp(szOp, "Relu");
    if (relu == 0) {
      excluded_providers.insert(kNnapiExecutionProvider);
      excluded_providers.insert(kQnnExecutionProvider);
    }
#endif
// Use relative error because of computation error for float::max
#if defined(USE_DNNL)
    int gelu = strcmp(szOp, "Gelu");
    if (gelu == 0) {
      // OneDNN has a computation difference when computing FLT_MAX
      // Expected: 3.4028234663852886e+38
      // Actual:   3.4028232635611926e+38
      // Since the numbers are large relative error is used instead of
      // the default threshold which is a small value.
      test.SetOutputRelErr("Y", .000001f);
    }
#endif

    if (strcmp(szOp, "QuickGelu") == 0) {
      test.SetOutputTolerance(0.0001f);
    }

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
  }
}

class ActivationOpTest : public ::testing::Test {
 protected:
  std::vector<std::vector<float>> input_values{{-1.0f, 0, 1.0f,                                                                  // normal input values for activation
                                                100.0f, -100.0f, 1000.0f, -1000.0f,                                              // input values that leads to exp() overflow
                                                FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,                                            // min, denorm, -denorm
                                                FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity()}};                     // max, -max, inf
  std::vector<std::vector<double>> input_values_double{{-1.0, 0, 1.0,                                                            // normal input values for activation
                                                        100.0, -100.0, 1000.0, -1000.0,                                          // input values that leads to exp() overflow
                                                        DBL_MIN, DBL_MIN / 10, -DBL_MIN / 10,                                    // min, denorm, -denorm
                                                        DBL_MAX, -DBL_MAX, std::numeric_limits<double>::infinity()}};            // max, -max, inf
  std::vector<std::vector<int8_t>> input_values_int8{{-1, -5, 0, 1, 5, 100, -100,                                                // normal input values for activation
                                                      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()}};  // min, max
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
  std::vector<std::vector<MLFloat16>> input_values_fp16{{MLFloat16(-1.0f),
                                                         MLFloat16(-5.f),
                                                         MLFloat16(),
                                                         MLFloat16(1.f),
                                                         MLFloat16(5.f),
                                                         MLFloat16(100.f),
                                                         MLFloat16(-100.f),
                                                         MLFloat16(65504.f),
                                                         MLFloat16(-65504.f)}};
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED

  void SetUp() override {
    float low = -1.0f, high = 1.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<std::size_t> batch_size_list = {1, 2, 4, 9, 100000};
    for (auto batch_size : batch_size_list) {
      std::vector<float> vec(batch_size);
      for (size_t i = 0; i != batch_size; ++i) {
        vec[i] = dist(gen);
      }
      input_values.emplace_back(vec);
    }
  }
};

class ActivationOpNoInfTest : public ::testing::Test {
 protected:
  std::vector<std::vector<float>> input_values{{-1.0f, 0, 1.0f,                        // normal input values for activation
                                                FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,  // min, denorm, -denorm
                                                FLT_MAX, -FLT_MAX}};                   // max, -max, inf

  void SetUp() override {
    float low = -1.0f, high = 1.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<std::size_t> batch_size_list = {1, 2, 4, 9, 100000};
    for (auto batch_size : batch_size_list) {
      std::vector<float> vec(batch_size);
      for (size_t i = 0; i != batch_size; ++i) {
        vec[i] = dist(gen);
      }
      input_values.emplace_back(vec);
    }
  }
};

}  // namespace test
}  // namespace onnxruntime
