// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

template <typename T>
class LinalgSolveContribOpTest : public ::testing::Test {
};

using LinalgTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LinalgSolveContribOpTest, LinalgTypes);

// DO NOT EDIT following test cases.they are generated with:
// in test_linalg_ops_with_pytorch.py, set generate_testcases to True to print C++ test cases
// python onnxruntime/test/python/test_linalg_ops_with_pytorch.py -k TestLinalgOps.test_linalg_solve
TYPED_TEST(LinalgSolveContribOpTest, no_batch_no_left_no_boardcast_no_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)0);
  test.AddInput<TypeParam>("A", {3, 3}, {
      7.207892f, 4.241426f, 1.942765f,
      4.241426f, 3.455372f, 0.326367f,
      1.942765f, 0.326367f, 1.382308f
    });
  test.AddInput<TypeParam>("B", {4, 3}, {
      -0.403344f, -0.596635f, 0.182036f,
      -0.856675f, 1.100604f, -1.071187f,
      0.122701f, -0.566317f, 0.373115f,
      -0.891995f, -1.509108f, 0.370394f
    });
  test.AddOutput<TypeParam>("X", {4, 3}, {{
      0.235647f, -0.453186f, -0.092502f,
      -3.583405f, 4.413022f, 3.219445f,
      1.368743f, -1.726304f, -1.246193f,
      1.550905f, -2.209154f, -1.390180f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, no_batch_no_left_no_boardcast_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)0);
  test.AddInput<TypeParam>("A", {3, 3}, {
      7.207892f, 4.241426f, 1.942765f,
      4.241426f, 3.455372f, 0.326367f,
      1.942765f, 0.326367f, 1.382308f
    });
  test.AddInput<TypeParam>("B", {1, 3}, {
      -0.403344f, -0.596635f, 0.182036f
    });
  test.AddOutput<TypeParam>("X", {1, 3}, {{
      0.235647f, -0.453186f, -0.092502f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, no_batch_left_no_boardcast_no_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)1);
  test.AddInput<TypeParam>("A", {4, 4}, {
      2.846490f, -0.756021f, 0.871596f, -1.733702f,
      -0.756021f, 5.773203f, -1.363091f, 1.128313f,
      0.871596f, -1.363091f, 1.917311f, -1.210296f,
      -1.733702f, 1.128313f, -1.210296f, 2.854497f
    });
  test.AddInput<TypeParam>("B", {4, 3}, {
      -0.719258f, -0.403344f, -0.596635f,
      0.182036f, -0.856675f, 1.100604f,
      -1.071187f, 0.122701f, -0.566317f,
      0.373115f, -0.891995f, -1.509108f
    });
  test.AddOutput<TypeParam>("X", {4, 3}, {{
      -0.225903f, -0.515750f, -0.785838f,
      -0.112631f, -0.137781f, 0.198996f,
      -0.699208f, -0.218630f, -0.657229f,
      -0.258434f, -0.663969f, -1.363283f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, no_batch_left_no_boardcast_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)1);
  test.AddInput<TypeParam>("A", {4, 4}, {
      2.846490f, -0.756021f, 0.871596f, -1.733702f,
      -0.756021f, 5.773203f, -1.363091f, 1.128313f,
      0.871596f, -1.363091f, 1.917311f, -1.210296f,
      -1.733702f, 1.128313f, -1.210296f, 2.854497f
    });
  test.AddInput<TypeParam>("B", {4}, {
      -0.719258f, -0.403344f, -0.596635f, 0.182036f
    });
  test.AddOutput<TypeParam>("X", {4}, {{
      -0.312115f, -0.167263f, -0.444982f, -0.248349f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, batch_no_left_no_boardcast_no_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)0);
  test.AddInput<TypeParam>("A", {2, 3, 3}, {
      2.916542f, -2.464638f, -1.485766f,
      -2.464638f, 3.406585f, 0.110846f,
      -1.485766f, 0.110846f, 3.769273f,
      0.088604f, -0.320561f, 0.202483f,
      -0.320561f, 5.342275f, 1.497411f,
      0.202483f, 1.497411f, 3.165195f
    });
  test.AddInput<TypeParam>("B", {2, 4, 3}, {
      -0.492677f, 0.248415f, 0.439696f,
      0.112411f, 0.640792f, 0.441156f,
      -0.215863f, -0.742548f, -0.573077f,
      -0.555358f, 0.594323f, 1.541943f,
      0.507334f, -0.591033f, -0.569248f,
      0.919971f, -0.069073f, -0.494925f,
      -1.495915f, -0.193837f, 0.445512f,
      1.325275f, -1.629326f, -0.549744f
    });
  test.AddOutput<TypeParam>("X", {2, 4, 3}, {{
      -0.249439f, -0.108246f, 0.021512f,
      1.203938f, 1.040889f, 0.560996f,
      -1.573670f, -1.332656f, -0.733155f,
      0.727643f, 0.678912f, 0.675938f,
      13.462979f, 1.140217f, -1.580517f,
      25.272738f, 2.306375f, -2.864220f,
      -40.813610f, -3.754412f, 4.527832f,
      32.358044f, 2.611879f, -3.479328f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, batch_no_left_no_boardcast_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)0);
  test.AddInput<TypeParam>("A", {2, 3, 3}, {
      2.916542f, -2.464638f, -1.485766f,
      -2.464638f, 3.406585f, 0.110846f,
      -1.485766f, 0.110846f, 3.769273f,
      0.088604f, -0.320561f, 0.202483f,
      -0.320561f, 5.342275f, 1.497411f,
      0.202483f, 1.497411f, 3.165195f
    });
  test.AddInput<TypeParam>("B", {2, 1, 3}, {
      0.408716f, 1.421418f, 0.149397f,
      -0.670860f, -0.214186f, -0.431969f
    });
  test.AddOutput<TypeParam>("X", {2, 1, 3}, {{
      2.423967f, 2.140646f, 0.932159f,
      -16.843258f, -1.515485f, 1.657973f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, batch_no_left_boardcast_no_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)0);
  test.AddInput<TypeParam>("A", {2, 3, 3}, {
      2.916542f, -2.464638f, -1.485766f,
      -2.464638f, 3.406585f, 0.110846f,
      -1.485766f, 0.110846f, 3.769273f,
      0.088604f, -0.320561f, 0.202483f,
      -0.320561f, 5.342275f, 1.497411f,
      0.202483f, 1.497411f, 3.165195f
    });
  test.AddInput<TypeParam>("B", {4, 3}, {
      0.408716f, 1.421418f, 0.149397f,
      -0.670860f, -0.214186f, -0.431969f,
      -0.707878f, -0.106434f, -1.242732f,
      -0.476232f, -0.685918f, -1.505142f
    });
  test.AddOutput<TypeParam>("X", {2, 4, 3}, {{
      2.423967f, 2.140646f, 0.932159f,
      -1.617166f, -1.209568f, -0.716484f,
      -2.049185f, -1.478216f, -1.093974f,
      -2.506701f, -1.971671f, -1.329424f,
      13.611500f, 1.514484f, -1.540033f,
      -16.843258f, -1.515485f, 1.657973f,
      -15.300617f, -1.270853f, 1.187406f,
      -9.874021f, -0.881538f, 0.573173f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, batch_no_left_boardcast_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)0);
  test.AddInput<TypeParam>("A", {2, 3, 3}, {
      2.916542f, -2.464638f, -1.485766f,
      -2.464638f, 3.406585f, 0.110846f,
      -1.485766f, 0.110846f, 3.769273f,
      0.088604f, -0.320561f, 0.202483f,
      -0.320561f, 5.342275f, 1.497411f,
      0.202483f, 1.497411f, 3.165195f
    });
  test.AddInput<TypeParam>("B", {1, 3}, {
      0.408716f, 1.421418f, 0.149397f
    });
  test.AddOutput<TypeParam>("X", {2, 1, 3}, {{
      2.423967f, 2.140646f, 0.932159f,
      13.611500f, 1.514484f, -1.540033f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, batch_left_no_boardcast_no_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)1);
  test.AddInput<TypeParam>("A", {2, 4, 4}, {
      2.846490f, -0.756021f, 0.871596f, -1.733702f,
      -0.756021f, 5.773203f, -1.363091f, 1.128313f,
      0.871596f, -1.363091f, 1.917311f, -1.210296f,
      -1.733702f, 1.128313f, -1.210296f, 2.854497f,
      5.656603f, 3.104256f, 0.025553f, -4.702889f,
      3.104256f, 6.327088f, 1.758924f, -3.521260f,
      0.025553f, 1.758924f, 0.969322f, -0.205388f,
      -4.702889f, -3.521260f, -0.205388f, 6.054066f
    });
  test.AddInput<TypeParam>("B", {2, 4, 3}, {
      -0.613583f, 0.031593f, -0.492677f,
      0.248415f, 0.439696f, 0.112411f,
      0.640792f, 0.441156f, 0.205526f,
      -0.450330f, -0.573077f, -0.555358f,
      0.594323f, 1.541943f, 0.507334f,
      -0.591033f, -1.325326f, 0.188554f,
      -0.069073f, -0.494925f, -1.495915f,
      -0.193837f, 0.445512f, 1.325275f
    });
  test.AddOutput<TypeParam>("X", {2, 4, 3}, {{
      -0.524159f, -0.190020f, -0.465158f,
      0.150501f, 0.166460f, 0.067125f,
      0.466158f, 0.264487f, 0.066190f,
      -0.337954f, -0.269829f, -0.475542f,
      0.380720f, 1.101162f, -0.025249f,
      -0.539868f, -0.609932f, 2.319154f,
      0.894120f, 0.693820f, -5.462131f,
      -0.019942f, 0.597768f, 1.362889f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, batch_left_no_boardcast_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)1);
  test.AddInput<TypeParam>("A", {2, 4, 4}, {
      2.846490f, -0.756021f, 0.871596f, -1.733702f,
      -0.756021f, 5.773203f, -1.363091f, 1.128313f,
      0.871596f, -1.363091f, 1.917311f, -1.210296f,
      -1.733702f, 1.128313f, -1.210296f, 2.854497f,
      5.656603f, 3.104256f, 0.025553f, -4.702889f,
      3.104256f, 6.327088f, 1.758924f, -3.521260f,
      0.025553f, 1.758924f, 0.969322f, -0.205388f,
      -4.702889f, -3.521260f, -0.205388f, 6.054066f
    });
  test.AddInput<TypeParam>("B", {2, 4}, {
      -0.566317f, 0.373115f, -0.891995f, -1.509108f,
      0.370394f, 1.456503f, 0.939810f, 0.774849f
    });
  test.AddOutput<TypeParam>("X", {2, 4}, {{
      -0.749342f, 0.003339f, -1.015990f, -1.415893f,
      0.574198f, -0.054876f, 1.177326f, 0.582058f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, batch_left_boardcast_no_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)1);
  test.AddInput<TypeParam>("A", {2, 4, 4}, {
      2.846490f, -0.756021f, 0.871596f, -1.733702f,
      -0.756021f, 5.773203f, -1.363091f, 1.128313f,
      0.871596f, -1.363091f, 1.917311f, -1.210296f,
      -1.733702f, 1.128313f, -1.210296f, 2.854497f,
      5.656603f, 3.104256f, 0.025553f, -4.702889f,
      3.104256f, 6.327088f, 1.758924f, -3.521260f,
      0.025553f, 1.758924f, 0.969322f, -0.205388f,
      -4.702889f, -3.521260f, -0.205388f, 6.054066f
    });
  test.AddInput<TypeParam>("B", {4, 3}, {
      -0.566317f, 0.373115f, -0.891995f,
      -1.509108f, 0.370394f, 1.456503f,
      0.939810f, 0.774849f, 0.191869f,
      1.263795f, -1.290435f, -0.791103f
    });
  test.AddOutput<TypeParam>("X", {2, 4, 3}, {{
      0.034407f, -0.244951f, -0.766153f,
      -0.226307f, 0.215456f, 0.354410f,
      0.904993f, 0.321732f, 0.195547f,
      0.936803f, -0.549595f, -0.799650f,
      0.880404f, 0.072070f, -1.015060f,
      -2.001246f, -1.122906f, 0.889210f,
      4.553026f, 2.682681f, -1.484702f,
      -0.116869f, -0.719277f, -0.452360f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}

TYPED_TEST(LinalgSolveContribOpTest, batch_left_boardcast_b_as_vector) {
  OpTester test("LinalgSolve", 1, kMSDomain);
  test.AddAttribute("left", (int64_t)1);
  test.AddInput<TypeParam>("A", {2, 4, 4}, {
      2.846490f, -0.756021f, 0.871596f, -1.733702f,
      -0.756021f, 5.773203f, -1.363091f, 1.128313f,
      0.871596f, -1.363091f, 1.917311f, -1.210296f,
      -1.733702f, 1.128313f, -1.210296f, 2.854497f,
      5.656603f, 3.104256f, 0.025553f, -4.702889f,
      3.104256f, 6.327088f, 1.758924f, -3.521260f,
      0.025553f, 1.758924f, 0.969322f, -0.205388f,
      -4.702889f, -3.521260f, -0.205388f, 6.054066f
    });
  test.AddInput<TypeParam>("B", {4}, {
      -0.566317f, 0.373115f, -0.891995f, -1.509108f
    });
  test.AddOutput<TypeParam>("X", {2, 4}, {{
      -0.749342f, 0.003339f, -1.015990f, -1.415893f,
      -1.325651f, 1.274423f, -3.335771f, -0.650975f
    }},
    false,
    1e-3f,
    1e-3f);
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
