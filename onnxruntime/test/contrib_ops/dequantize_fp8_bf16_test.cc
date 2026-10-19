// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/common/float8.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) && !defined(DISABLE_FLOAT8_TYPES)
TEST(DequantizeLinearTest, Fp8ToBf16) {
  auto provider = DefaultCudaExecutionProvider();
  if (provider == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));

  OpTester test("DequantizeLinear", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("output_dtype", ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
  test.AddInput<Float8E4M3FN>("weight", {1, 8}, std::vector<Float8E4M3FN>(8, Float8E4M3FN(1.0f)));
  test.AddInput<float>("scale", {8}, std::vector<float>(8, 2.0f));
  test.AddOptionalInputEdge<Float8E4M3FN>();
  test.AddOutput<BFloat16>("weight_bf16", {1, 8}, ToBFloat16(std::vector<float>(8, 2.0f)));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
#endif

}  // namespace test
}  // namespace onnxruntime
