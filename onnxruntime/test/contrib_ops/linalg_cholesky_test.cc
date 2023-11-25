// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

template <typename T>
class LinalgCholeskyContribOpTest : public ::testing::Test {
};

using LinalgTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LinalgCholeskyContribOpTest, LinalgTypes);

// DO NOT EDIT following test cases.they are generated with:
// in test_linalg_ops_with_pytorch.py, set generate_testcases to True to print C++ test cases
// python onnxruntime/test/python/test_linalg_ops_with_pytorch.py -k TestLinalgOps.test_linalg_cholesky

TYPED_TEST(LinalgCholeskyContribOpTest, no_batch_lower) {
  OpTester test("LinalgCholesky", 1, kMSDomain);
  test.AddAttribute("upper", (int64_t)0);
  test.AddInput<TypeParam>("A", {4, 4}, {
      3.846490f, -0.756021f, 0.871596f, -1.733702f,
      -0.756021f, 6.773203f, -1.363091f, 1.128313f,
      0.871596f, -1.363091f, 2.917311f, -1.210296f,
      -1.733702f, 1.128313f, -1.210296f, 3.854497f
    });
  test.AddOutput<TypeParam>("L", {4, 4}, {
      1.961247f, 0.000000f, 0.000000f, 0.000000f,
      -0.385480f, 2.573832f, 0.000000f, 0.000000f,
      0.444409f, -0.463038f, 1.582848f, 0.000000f,
      -0.883979f, 0.305986f, -0.426929f, 1.672477f
    });
  test.Run();
}

TYPED_TEST(LinalgCholeskyContribOpTest, no_batch_upper) {
  OpTester test("LinalgCholesky", 1, kMSDomain);
  test.AddAttribute("upper", (int64_t)1);
  test.AddInput<TypeParam>("A", {4, 4}, {
      3.846490f, -0.756021f, 0.871596f, -1.733702f,
      -0.756021f, 6.773203f, -1.363091f, 1.128313f,
      0.871596f, -1.363091f, 2.917311f, -1.210296f,
      -1.733702f, 1.128313f, -1.210296f, 3.854497f
    });
  test.AddOutput<TypeParam>("L", {4, 4}, {
      1.961247f, -0.385480f, 0.444409f, -0.883979f,
      0.000000f, 2.573832f, -0.463038f, 0.305986f,
      0.000000f, 0.000000f, 1.582848f, -0.426929f,
      0.000000f, 0.000000f, 0.000000f, 1.672477f
    });
  test.Run();
}

TYPED_TEST(LinalgCholeskyContribOpTest, batch_lower) {
  OpTester test("LinalgCholesky", 1, kMSDomain);
  test.AddAttribute("upper", (int64_t)0);
  test.AddInput<TypeParam>("A", {2, 4, 4}, {
      3.846490f, -0.756021f, 0.871596f, -1.733702f,
      -0.756021f, 6.773203f, -1.363091f, 1.128313f,
      0.871596f, -1.363091f, 2.917311f, -1.210296f,
      -1.733702f, 1.128313f, -1.210296f, 3.854497f,
      6.656603f, 3.104256f, 0.025553f, -4.702889f,
      3.104256f, 7.327088f, 1.758924f, -3.521260f,
      0.025553f, 1.758924f, 1.969322f, -0.205388f,
      -4.702889f, -3.521260f, -0.205388f, 7.054066f
    });
  test.AddOutput<TypeParam>("L", {2, 4, 4}, {
      1.961247f, 0.000000f, 0.000000f, 0.000000f,
      -0.385480f, 2.573832f, 0.000000f, 0.000000f,
      0.444409f, -0.463038f, 1.582848f, 0.000000f,
      -0.883979f, 0.305986f, -0.426930f, 1.672477f,
      2.580039f, 0.000000f, 0.000000f, 0.000000f,
      1.203182f, 2.424756f, 0.000000f, 0.000000f,
      0.009904f, 0.720488f, 1.204210f, 0.000000f,
      -1.822797f, -0.547727f, 0.172142f, 1.844407f
    });
  test.Run();
}

TYPED_TEST(LinalgCholeskyContribOpTest, batch_upper) {
  OpTester test("LinalgCholesky", 1, kMSDomain);
  test.AddAttribute("upper", (int64_t)1);
  test.AddInput<TypeParam>("A", {2, 4, 4}, {
      3.846490f, -0.756021f, 0.871596f, -1.733702f,
      -0.756021f, 6.773203f, -1.363091f, 1.128313f,
      0.871596f, -1.363091f, 2.917311f, -1.210296f,
      -1.733702f, 1.128313f, -1.210296f, 3.854497f,
      6.656603f, 3.104256f, 0.025553f, -4.702889f,
      3.104256f, 7.327088f, 1.758924f, -3.521260f,
      0.025553f, 1.758924f, 1.969322f, -0.205388f,
      -4.702889f, -3.521260f, -0.205388f, 7.054066f
    });
  test.AddOutput<TypeParam>("L", {2, 4, 4}, {
      1.961247f, -0.385480f, 0.444409f, -0.883979f,
      0.000000f, 2.573832f, -0.463038f, 0.305986f,
      0.000000f, 0.000000f, 1.582848f, -0.426930f,
      0.000000f, 0.000000f, 0.000000f, 1.672477f,
      2.580039f, 1.203182f, 0.009904f, -1.822797f,
      0.000000f, 2.424756f, 0.720488f, -0.547727f,
      0.000000f, 0.000000f, 1.204210f, 0.172142f,
      0.000000f, 0.000000f, 0.000000f, 1.844407f
    });
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
