// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "gtest/gtest.h"
#include <vector>
#include <unordered_map>
#include <functional>
#include <random>
#include "core/graph/constants.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
inline void TestActivationOp(const char* szOp, const std::vector<std::vector<T>>& input_vals_vec,
                             std::function<T(T)> expected_func,
                             const std::unordered_map<std::string, float> attribs = {},
                             bool is_tensorrt_supported = true, int opset_version = 7,
                             const char* domain = kOnnxDomain) {
  for (const std::vector<T>& input_vals : input_vals_vec) {
    OpTester test(szOp, opset_version, domain);

    for (auto attr : attribs) test.AddAttribute<float>(attr.first, attr.second);
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

//Disabled because of accuracy issues for MYRIAD FP16 and VAD_M
#if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M)
    int relu = strcmp(szOp, "Relu");
    int leaky = strcmp(szOp, "LeakyRelu");
    int elu = strcmp(szOp, "Elu");
    int sigmoid = strcmp(szOp, "Sigmoid");
    int tanh = strcmp(szOp, "Tanh");
    if (relu == 0 || leaky == 0) {
      excluded_providers.insert(kOpenVINOExecutionProvider);
    }
    if (elu == 0)
      excluded_providers.insert(kOpenVINOExecutionProvider);
    if (sigmoid == 0 || tanh == 0)
      excluded_providers.insert(kOpenVINOExecutionProvider);
#endif

//Disabled because of accuracy issues for GPU
#if defined(OPENVINO_CONFIG_GPU_FP16) || defined(OPENVINO_CONFIG_GPU_FP32)
    int leaky = strcmp(szOp, "LeakyRelu");
    if (leaky == 0) {
      excluded_providers.insert(kOpenVINOExecutionProvider);
    }
#endif

//Disabled because of NNAPI treat float::inf as float::max
#if defined(USE_NNAPI)
    int relu = strcmp(szOp, "Relu");
    if (relu == 0) {
      excluded_providers.insert(kNnapiExecutionProvider);
    }
#endif

//Disable TensorRT EP because TensorRT 8.0 doesn't fully support Double data type
#if defined(USE_TENSORRT)
    int relu = strcmp(szOp, "Relu");
    if (relu == 0) {
      excluded_providers.insert(kTensorrtExecutionProvider);
    }
#endif
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
  }
}

class ActivationOpTest : public ::testing::Test {
 protected:
  std::vector<std::vector<float>> input_values{{-1.0f, 0, 1.0f,                                                        // normal input values for activation
                                                100.0f, -100.0f, 1000.0f, -1000.0f,                                    // input values that leads to exp() overflow
                                                FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,                                  // min, denorm, -denorm
                                                FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity()}};           // max, -max, inf
  std::vector<std::vector<double>> input_values_double{{-1.0, 0, 1.0,                                                  // normal input values for activation
                                                        100.0, -100.0, 1000.0, -1000.0,                                // input values that leads to exp() overflow
                                                        DBL_MIN, DBL_MIN / 10, -DBL_MIN / 10,                          // min, denorm, -denorm
                                                        DBL_MAX, -DBL_MAX, std::numeric_limits<double>::infinity()}};  // max, -max, inf

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
