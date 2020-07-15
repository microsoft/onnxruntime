// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

TEST(TriuContribOpTest, two_by_two_float) {
  OpTester test("Triu", 1, kMSDomain);
  test.AddInput<float>("X", {2, 2}, {4.f, 7.f, 2.f, 6.f});
  test.AddOutput<float>("Y", {2, 2}, {4.f, 7.f, 0.f, 6.f});
  test.Run();
}

TEST(TriuContribOpTest, two_by_two_double) {
  OpTester test("Triu", 1, kMSDomain);
  test.AddInput<double>("X", {2, 2}, {4, 7, 2, 6});
  test.AddInput<int64_t>("k", {1}, {1});
  test.AddOutput<double>("Y", {2, 2}, {0, 7, 0, 0});
  test.Run();
}

TEST(TriuContribOpTest, three_dim_float) {
  OpTester test("Triu", 1, kMSDomain);
  test.AddInput<float>("X", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.AddInput<int64_t>("k", {1}, {1});
  test.AddOutput<float>("Y", {2, 3, 4},
    {0.f, 1.f, 5.f, 8.f,
     0.f, 0.f, 2.f, 4.f,
     0.f, 0.f, 0.f, 3.f,
     0.f, 6.f, 2.f, 1.f,
     0.f, 0.f, 5.f, 8.f,
     0.f, 0.f, 0.f, 4.f,
    });
  test.Run();
}

TEST(TriuContribOpTest, neg_k_float) {
  OpTester test("Triu", 1, kMSDomain);
  test.AddInput<float>("X", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.AddInput<int64_t>("k", {1}, {-1});
  test.AddOutput<float>("Y", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     0.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     0.f, 3.f, 2.f, 4.f,
    });
  test.Run();
}

TEST(TriuContribOpTest, small_k_float) {
  OpTester test("Triu", 1, kMSDomain);
  test.AddInput<float>("X", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.AddInput<int64_t>("k", {1}, {-5});
  test.AddOutput<float>("Y", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime