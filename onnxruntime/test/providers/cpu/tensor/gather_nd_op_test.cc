// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
static void RunTest(
    const std::vector<int64_t>& input_dims,
    const std::vector<T>& input,
    const std::vector<int64_t>& indices_dims,
    const std::vector<int64_t>& indices,
    const std::vector<int64_t>& output_dims,
    const std::vector<T>& output) {

  // ONNX domain opset-11
  OpTester test1("GatherND", 11);
  test1.AddInput<T>("data", input_dims, input);
  test1.AddInput<int64_t>("indices", indices_dims, indices);
  test1.AddOutput<T>("output", output_dims, output);
  test1.Run();

#ifndef DISABLE_CONTRIB_OPS

  // MSFT domain opset-1 (contrib op)
  OpTester test2("GatherND", 1, kMSDomain);
  test2.AddInput<T>("data", input_dims, input);
  test2.AddInput<int64_t>("indices", indices_dims, indices);
  test2.AddOutput<T>("output", output_dims, output);
  test2.Run();

#endif

}

TEST(GatherNDOpTest, string) {
  RunTest<std::string>({2, 2},
                       {"h", "k", "o", "z"},
                       {2},
                       {0, 1},
                       {},
                       {"k"});

  RunTest<std::string>({6},
                       {"h", "k", "o", "z", "l", "t"},
                       {1},
                       {3},
                       {},
                       {"z"});

  RunTest<std::string>({3, 2},
                       {"h", "k", "o", "z", "l", "t"},
                       {2},
                       {2, 1},
                       {},
                       {"t"});

  RunTest<std::string>({2, 2},
                       {"a", "b", "c", "d"},
                       {2, 2},
                       {0LL, 0LL, 1LL, 1LL},
                       {2},
                       {"a", "d"});

    RunTest<std::string>({2, 2, 2},
                       {"egg", "dance", "air", "bob", "terry", "smart", "laugh", "kite"},
                       {2, 1, 2},
                       {0LL, 1LL, 1LL, 0LL},
                       {2, 1, 2},
                       {"air", "bob", "terry", "smart"});

    RunTest<std::string>({3, 3},
                         {"egg", "dance", "air", "bob", "terry", "smart", "laugh", "kite", "hop"},
                         {3, 2},
                         {2, 1, 1, 0, 0, 1},
                         {3},
                         {"kite", "bob", "dance"});
}

TEST(GatherNDOpTest, int64_t) {
  RunTest<int64_t>({2, 2},
                   {0LL, 1LL, 2LL, 3LL},
                   {2, 2},
                   {0LL, 0LL, 1LL, 1LL},
                   {2},
                   {0LL, 3LL});

  RunTest<int64_t>({2, 2, 2},
                   {0LL, 1LL, 2LL, 3LL, 4LL, 5LL, 6LL, 7LL},
                   {2, 2},
                   {0LL, 1LL, 1LL, 0LL},
                   {2, 2},
                   {2LL, 3LL, 4LL, 5LL});

  RunTest<int64_t>({2, 2},
                   {0LL, 1LL, 2LL, 3LL},
                   {2, 1, 2},
                   {0LL, 0LL, 0LL, 1LL},
                   {2, 1},
                   {0LL, 1LL});

    RunTest<int64_t>({2, 2},
                   {0LL, 1LL, 2LL, 3LL},
                   {2, 1, 1},
                   {1LL, 0LL},
                   {2, 1, 2},
                   {2LL, 3LL, 0LL, 1LL});
}

TEST(GatherNDOpTest, float) {
  RunTest<float>({2, 2},
                 {0.0f, 0.1f, 0.2f, 0.3f},
                 {2, 1},
                 {1LL, 0LL},
                 {2, 2},
                 {0.2f, 0.3f, 0.0f, 0.1f});
}

TEST(GatherNDOpTest, double) {
  RunTest<double>({2, 2},
                 {0.0f, 0.1f, 0.2f, 0.3f},
                 {2, 1},
                 {1LL, 0LL},
                 {2, 2},
                 {0.2f, 0.3f, 0.0f, 0.1f});
}

}  // namespace test
}  // namespace onnxruntime
