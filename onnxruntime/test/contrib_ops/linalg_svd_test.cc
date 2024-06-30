// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

template <typename T>
class LinalgSVDContribOpTest : public ::testing::Test {
};

using LinalgTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LinalgSVDContribOpTest, LinalgTypes);

// DO NOT EDIT following test cases.they are generated with:
// in test_linalg_ops_with_pytorch.py, set generate_testcases to True to print C++ test cases
// python onnxruntime/test/python/test_linalg_ops_with_pytorch.py -k TestLinalgOps.test_linalg_svd
TYPED_TEST(LinalgSVDContribOpTest, batch_full_matrices) {
  OpTester test("LinalgSVD", 1, kMSDomain);
  test.AddAttribute("full_matrices", (int64_t)1);
  test.AddInput<TypeParam>("A", {2, 3, 4}, {
      -1.125840f, -1.152360f, -0.250579f, -0.433879f,
      0.848710f, 0.692009f, -0.316013f, -2.115219f,
      0.468096f, -0.157712f, 1.443660f, 0.266049f,
      0.166455f, 0.874382f, -0.143474f, -0.111609f,
      0.931827f, 1.259009f, 2.004981f, 0.053737f,
      0.618057f, -0.412802f, -0.841065f, -2.316042f
    });
  test.AddOutput<TypeParam>("U", {2, 3, 3}, {
      0.190744f, 0.773181f, -0.604820f,
      -0.969842f, 0.053195f, -0.237860f,
      0.151736f, -0.631950f, -0.760010f,
      0.078401f, -0.181647f, 0.980233f,
      0.702239f, -0.687852f, -0.183633f,
      -0.707612f, -0.702755f, -0.073631f
    });
  test.AddOutput<TypeParam>("S", {2, 3}, {{
      2.456875f, 1.861905f, 1.231135f,
      2.889926f, 2.222110f, 0.797447f
    }});
  test.AddOutput<TypeParam>("Vh", {2, 4, 4}, {
      -0.393522f, -0.372374f, 0.194451f, 0.817720f,
      -0.602149f, -0.405233f, -0.603078f, -0.330906f,
      0.100150f, 0.529781f, -0.707051f, 0.457582f,
      0.687406f, -0.645334f, -0.313951f, 0.111593f,
      0.079611f, 0.430731f, 0.689247f, 0.577123f,
      -0.497517f, -0.330651f, -0.342920f, 0.724951f,
      -0.067035f, 0.822999f, -0.560399f, 0.064283f,
      0.861188f, -0.166776f, -0.305446f, 0.370463f
    });
  test.Run();
}

TYPED_TEST(LinalgSVDContribOpTest, batch_no_full_matrices) {
  OpTester test("LinalgSVD", 1, kMSDomain);
  test.AddAttribute("full_matrices", (int64_t)0);
  test.AddInput<TypeParam>("A", {2, 3, 4}, {
      -1.125840f, -1.152360f, -0.250579f, -0.433879f,
      0.848710f, 0.692009f, -0.316013f, -2.115219f,
      0.468096f, -0.157712f, 1.443660f, 0.266049f,
      0.166455f, 0.874382f, -0.143474f, -0.111609f,
      0.931827f, 1.259009f, 2.004981f, 0.053737f,
      0.618057f, -0.412802f, -0.841065f, -2.316042f
    });
  test.AddOutput<TypeParam>("U", {2, 3, 3}, {
      0.190744f, 0.773181f, -0.604820f,
      -0.969842f, 0.053195f, -0.237860f,
      0.151736f, -0.631950f, -0.760010f,
      0.078401f, -0.181647f, 0.980233f,
      0.702239f, -0.687852f, -0.183633f,
      -0.707612f, -0.702755f, -0.073631f
    });
  test.AddOutput<TypeParam>("S", {2, 3}, {{
      2.456875f, 1.861905f, 1.231135f,
      2.889926f, 2.222110f, 0.797447f
    }});
  test.AddOutput<TypeParam>("Vh", {2, 3, 4}, {
      -0.393522f, -0.372374f, 0.194451f, 0.817720f,
      -0.602149f, -0.405233f, -0.603078f, -0.330906f,
      0.100150f, 0.529781f, -0.707051f, 0.457582f,
      0.079611f, 0.430731f, 0.689247f, 0.577123f,
      -0.497517f, -0.330651f, -0.342920f, 0.724951f,
      -0.067035f, 0.822999f, -0.560399f, 0.064283f
    });
  test.Run();
}

TYPED_TEST(LinalgSVDContribOpTest, no_batch_full_matrices) {
  OpTester test("LinalgSVD", 1, kMSDomain);
  test.AddAttribute("full_matrices", (int64_t)1);
  test.AddInput<TypeParam>("A", {3, 4}, {
      1.540996f, -0.293429f, -2.178789f, 0.568431f,
      -1.084522f, -1.398595f, 0.403347f, 0.838026f,
      -0.719258f, -0.403344f, -0.596635f, 0.182036f
    });
  test.AddOutput<TypeParam>("U", {3, 3}, {
      -0.928314f, 0.342269f, -0.145207f,
      0.371614f, 0.841924f, -0.391236f,
      0.011654f, 0.417151f, 0.908762f
    });
  test.AddOutput<TypeParam>("S", {3}, {{
      2.862108f, 1.985799f, 0.679939f
    }});
  test.AddOutput<TypeParam>("Vh", {4, 4}, {
      -0.643559f, -0.088063f, 0.756623f, -0.074819f,
      -0.345297f, -0.728271f, -0.329858f, 0.491514f,
      -0.666373f, 0.328333f, -0.564209f, -0.360296f,
      -0.150161f, 0.595033f, 0.019583f, 0.789306f
    });
  test.Run();
}

TYPED_TEST(LinalgSVDContribOpTest, no_batch_no_full_matrices) {
  OpTester test("LinalgSVD", 1, kMSDomain);
  test.AddAttribute("full_matrices", (int64_t)0);
  test.AddInput<TypeParam>("A", {3, 4}, {
      1.540996f, -0.293429f, -2.178789f, 0.568431f,
      -1.084522f, -1.398595f, 0.403347f, 0.838026f,
      -0.719258f, -0.403344f, -0.596635f, 0.182036f
    });
  test.AddOutput<TypeParam>("U", {3, 3}, {
      -0.928314f, 0.342269f, -0.145207f,
      0.371614f, 0.841924f, -0.391236f,
      0.011654f, 0.417151f, 0.908762f
    });
  test.AddOutput<TypeParam>("S", {3}, {{
      2.862108f, 1.985799f, 0.679939f
    }});
  test.AddOutput<TypeParam>("Vh", {3, 4}, {
      -0.643559f, -0.088063f, 0.756623f, -0.074819f,
      -0.345297f, -0.728271f, -0.329858f, 0.491514f,
      -0.666373f, 0.328333f, -0.564209f, -0.360296f
    });
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
