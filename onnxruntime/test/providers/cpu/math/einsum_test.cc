// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/data_types.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

TEST(Einsum, EinsumAsIdentityImplicit) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i");
  test.AddInput<float>("x", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.AddOutput<float>("y", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.Run();
}

TEST(Einsum, EinsumAsIdentityExplicit) {
  OpTester test("Einsum", 12, onnxruntime::kOnnxDomain);
  test.AddAttribute<std::string>("equation", "i->i");
  test.AddInput<float>("x", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.AddOutput<float>("y", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime