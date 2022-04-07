// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#ifdef USE_CUDA
TEST(MaskFillTest, MaskFillNegativeAxis)
{
    OpTester test("MaskFill", 1, onnxruntime::kMSDomain);
    const int N = 1, C = 2, H = 3, W = 4;
    std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,
                            2.0f, 3.0f, 4.0f, 5.0f,
                            3.0f, 4.0f, 5.0f, 6.0f,

                            4.0f, 5.0f, 6.0f, 7.0f,
                            5.0f, 6.0f, 7.0f, 8.0f,
                            6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<int> mask = {1, 0, 1, 0};
    test.AddInput<float>("x", {N, C, H, W}, X);
    test.AddInput<int>("mask", {W}, mask);

    test.AddAttribute("axis", static_cast<int64_t>(-1));

    std::vector<float> Y = {1.0f, 0.0f, 3.0f, 0.0f,
                            2.0f, 0.0f, 4.0f, 0.0f,
                            3.0f, 0.0f, 5.0f, 0.0f,

                            4.0f, 0.0f, 6.0f, 0.0f,
                            5.0f, 0.0f, 7.0f, 0.0f,
                            6.0f, 0.0f, 8.0f, 0.0f};
    test.AddOutput<float>("y", {N, C, H, W}, Y);
    std::vector<std::unique_ptr<IExecutionProvider>> ep;
    ep.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider}, nullptr, &ep);
}

TEST(MaskFillTest, MaskFillWithAxisAttr)
{
    OpTester test("MaskFill", 1, onnxruntime::kMSDomain);
    const int N = 1, C = 2, H = 3, W = 4;
    std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,
                            2.0f, 3.0f, 4.0f, 5.0f,
                            3.0f, 4.0f, 5.0f, 6.0f,

                            4.0f, 5.0f, 6.0f, 7.0f,
                            5.0f, 6.0f, 7.0f, 8.0f,
                            6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<int> mask = {0, 1, 1};
    test.AddInput<float>("x", {N, C, H, W}, X);
    test.AddInput<int>("mask", {H}, mask);

    test.AddAttribute("axis", static_cast<int64_t>(2));

    std::vector<float> Y = {0.0f, 0.0f, 0.0f, 0.0f,
                            2.0f, 3.0f, 4.0f, 5.0f,
                            3.0f, 4.0f, 5.0f, 6.0f,

                            0.0f, 0.0f, 0.0f, 0.0f,
                            5.0f, 6.0f, 7.0f, 8.0f,
                            6.0f, 7.0f, 8.0f, 9.0f};
    test.AddOutput<float>("y", {N, C, H, W}, Y);
    std::vector<std::unique_ptr<IExecutionProvider>> ep;
    ep.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider}, nullptr, &ep);
}
#endif
}  // namespace test
}  // namespace onnxruntime
