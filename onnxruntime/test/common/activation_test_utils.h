#pragma once
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

inline void TestUnaryElementwiseOp(const char* szOp, std::vector<float>& input_vals,
                            std::function<float(float)> expected_func,
                            const std::unordered_map<std::string, float> attribs = {},
                            bool is_tensorrt_supported = true,
                            int opset_version = 7) {
  OpTester test(szOp, opset_version);

  for (auto attr : attribs)
    test.AddAttribute(attr.first, attr.second);

  std::vector<int64_t> dims{(int64_t)input_vals.size()};

  std::vector<float> expected_vals;
  for (const auto& iv : input_vals)
    expected_vals.push_back(expected_func(iv));

  test.AddInput<float>("X", dims, input_vals);
  test.AddOutput<float>("Y", dims, expected_vals);

  // Disable TensorRT on unsupported tests
  std::unordered_set<std::string> excluded_providers;
  if (!is_tensorrt_supported) {
    excluded_providers.insert(kTensorrtExecutionProvider);
  }
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
}

}  // namespace test
}  // namespace onnxruntime