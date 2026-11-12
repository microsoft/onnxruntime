// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime::test {

TEST(ConvOptionalInputTest, MissingRequiredWeightsWithBias) {
  for (const char* op : {"Conv", "ConvTranspose"}) {
    SCOPED_TRACE(op);
    OpTester test(op, 11);
    test.AddInput<float>("X", {1, 1, 1, 1}, {2});
    test.AddOptionalInputEdge<float>();
    test.AddInput<float>("B", {1}, {5});
    test.AddOutput<float>("Y", {1, 1, 1, 1}, {0});
    // W is required by both schemas, even when the optional bias is supplied.
    test.Run(OpTester::ExpectResult::kExpectFailure, "input 1 is marked single but has an empty string");
  }
}

}  // namespace onnxruntime::test
