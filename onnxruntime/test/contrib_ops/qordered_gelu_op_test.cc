// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/contrib_ops/qordered_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA)

static void
RunQOrdered_Gelu_Test(std::vector<int64_t> const& shape, float scale_x,
                      float scale_y, OrderCublasLt order) {
  static const float sqrt_of_2 = std::sqrt(2.0f);

  int64_t N = std::accumulate(shape.begin(), shape.end(), int64_t{1LL}, std::multiplies<int64_t>());
  std::vector<int8_t> vec_x = GenData<int8_t>(shape, 1.0f);
  std::vector<int8_t> vec_y(N);
  for (int64_t i = 0; i < N; i++) {
    float x = scale_x * static_cast<float>(vec_x[i]);
    float r = (x * (0.5f * (1.0f + std::erff(x / sqrt_of_2)))) / scale_y;
    int8_t q = static_cast<int8_t>(std::nearbyintf(std::min(127.0f, std::max(-128.0f, r))));
    vec_y[i] = q;
  }

  OpTester test_qorder("QOrderedGelu", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("order_X", static_cast<int64_t>(order));
  test_qorder.AddAttribute("order_Y", static_cast<int64_t>(order));
  test_qorder.AddInput<int8_t>("X", shape, vec_x);
  test_qorder.AddInput<float>("scale_X", {}, {scale_x});
  test_qorder.AddInput<float>("scale_Y", {}, {scale_y});
  test_qorder.AddOutput<int8_t>("Y", shape, vec_y, false, 0.0f, 0.0f /* abs error */);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QOrderedTest, Gelu_3x11x12) {
  RunQOrdered_Gelu_Test({3, 11, 12}, 1.0f / 32.0f, 1.0f / 128.0f, ORDER_ROW);
}

#endif

}  // namespace test
}  // namespace onnxruntime
