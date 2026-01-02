// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
TEST(BinaryCrossEntropyOpTest, BinaryCrossEntropyFloatLabelWithZeros) {
    OpTester test("BinaryCrossEntropy", 1, kMSDomain);
    test.AddInput<float>("logits", {3, 1}, {0.1f, 0.5f, 0.9f});
    test.AddInput<float>("label", {3, 1}, {0.0f, 0.0f, 0.0f});
    test.AddOutput<float>("Y", {}, {1.033697596403939f});
    test.AddOutput<float>("logit", {3, 1}, {0.1, 0.5, 0.9});
    test.Run();
}

TEST(BinaryCrossEntropyOpTest, BinaryCrossEntropyFloatNaN) {
    OpTester test("BinaryCrossEntropy", 1, kMSDomain);
    test.AddInput<float>("logits", {3, 1}, {0.0f, 1.0f, 0.0f});
    test.AddInput<float>("label", {3, 1}, {0.0f, 0.0f, 0.0f});
    test.AddOutput<float>("Y", {}, {6.14022691465078f});
    test.AddOutput<float>("logit", {3, 1}, {0.0, 1.0f, 0.0f});
    test.Run();
}

TEST(BinaryCrossEntropyOpTest, BinaryCrossEntropyFloatGrad) {
    OpTester test("BinaryCrossEntropyGrad", 1, kMSDomain);
    test.AddInput<float>("dY", {1}, {1.0f});
    test.AddInput<float>("logit", {4, 1}, {0.5f, 0.2f, 0.6f, 0.005f});
    test.AddInput<float>("label", {4, 1}, {0.3f, 0.7f, 0.11f, 0.98f});
    test.AddOutput<float>("d_logits", {4, 1}, {0.2f, -0.78125f, 0.5104f, -48.995f});
    test.Run();
}


} // namespace test
} // namespace onnxruntime

