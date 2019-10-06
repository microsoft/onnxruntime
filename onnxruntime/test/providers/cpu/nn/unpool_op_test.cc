// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

using namespace std;
namespace onnxruntime {
namespace test {

TEST(UnpoolTest, MaxUnPool1D) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 4};

  std::vector<int64_t> i_vals = {1, 3, 4, 6};
  std::vector<int64_t> i_dims = {1, 1, 4};

  std::vector<int64_t> expected_dims = {1, 1, 8};
  std::vector<float> expected_vals = {0, 1, 0, 2, 3, 0, 4, 0};

  std::vector<int64_t> inputDims = {3};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", inputDims, expected_dims);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool2D) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 2, 2};

  std::vector<int64_t> i_vals = {1, 3, 4, 6};
  std::vector<int64_t> i_dims = {1, 1, 2, 2};

  std::vector<int64_t> expected_dims = {1, 1, 4, 4};
  std::vector<float> expected_vals = {0, 1, 0, 2, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int64_t> inputDims = {4};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", inputDims, expected_dims);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool3D) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2, 2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2, 2});

  std::vector<float> t_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> t_dims = {1, 1, 2, 2, 2};

  std::vector<int64_t> i_vals = {1, 3, 24, 30, 32, 38, 60, 62};
  std::vector<int64_t> i_dims = {1, 1, 2, 2, 2};

  std::vector<int64_t> expected_dims = {1, 1, 4, 4, 4};
  std::vector<int64_t> expectedDims_Size = {5};

  std::vector<float> expected_vals =
      {
          //slice 1
          0, 1, 0, 2,
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,

          // slice 2
          0, 0, 0, 0,
          0, 0, 0, 0,
          3, 0, 0, 0,
          0, 0, 4, 0,

          //slice 3
          5, 0, 0, 0,
          0, 0, 6, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,

          // slice 4
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,
          7, 0, 8, 0};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", expectedDims_Size, expected_dims);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool1D_Without_OutputShape) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 4};

  std::vector<int64_t> i_vals = {1, 3, 4, 6};
  std::vector<int64_t> i_dims = {1, 1, 4};

  std::vector<int64_t> expected_dims = {1, 1, 8};
  std::vector<float> expected_vals = {0, 1, 0, 2, 3, 0, 4, 0};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool2D_Without_OutputShape) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 2, 2};

  std::vector<int64_t> i_vals = {1, 3, 4, 6};
  std::vector<int64_t> i_dims = {1, 1, 2, 2};

  std::vector<int64_t> expected_dims = {1, 1, 4, 4};
  std::vector<float> expected_vals = {0, 1, 0, 2, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool3D_Without_OutputShape) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2, 2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2, 2});

  std::vector<float> t_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> t_dims = {1, 1, 2, 2, 2};

  std::vector<int64_t> i_vals = {1, 3, 24, 30, 32, 38, 60, 62};
  std::vector<int64_t> i_dims = {1, 1, 2, 2, 2};

  std::vector<int64_t> expected_dims = {1, 1, 4, 4, 4};

  std::vector<float> expected_vals =
      {
          //slice 1
          0, 1, 0, 2,
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,

          // slice 2
          0, 0, 0, 0,
          0, 0, 0, 0,
          3, 0, 0, 0,
          0, 0, 4, 0,

          //slice 3
          5, 0, 0, 0,
          0, 0, 6, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,

          // slice 4
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,
          7, 0, 8, 0};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool1D_Padding) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});
  test.AddAttribute("pads", vector<int64_t>{1, 0});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 4};

  std::vector<int64_t> i_vals = {1, 3, 4, 6};
  std::vector<int64_t> i_dims = {1, 1, 4};

  std::vector<int64_t> expected_dims = {1, 1, 7};
  std::vector<float> expected_vals = {0, 1, 0, 2, 3, 0, 4};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);

  test.AddOutput<float>("YP", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool2D_Padding) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
  test.AddAttribute("pads", vector<int64_t>{1, 1, 0, 0});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 2, 2};

  std::vector<int64_t> i_vals = {1, 3, 4, 6};
  std::vector<int64_t> i_dims = {1, 1, 2, 2};

  std::vector<int64_t> expected_dims = {1, 1, 3, 3};
  std::vector<float> expected_vals = {0, 1, 0, 2, 3, 0, 4, 0, 0};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool3D_Padding) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2, 2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2, 2});
  test.AddAttribute("pads", vector<int64_t>{0, 1, 1, 0, 0, 0});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 1, 2, 2};

  std::vector<int64_t> i_vals = {1, 4, 8, 12};
  std::vector<int64_t> i_dims = {1, 1, 1, 2, 2};

  std::vector<int64_t> expected_dims = {1, 1, 2, 3, 3};

  std::vector<float> expected_vals = {
      0, 1, 0,
      0, 2, 0,
      0, 0, 3,
      0, 0, 0,
      4, 0, 0,
      0, 0, 0};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool1D_WithPaddedOutput) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 4};

  std::vector<int64_t> i_vals = {1, 3, 4, 6};
  std::vector<int64_t> i_dims = {1, 1, 4};

  std::vector<int64_t> expected_dims = {1, 1, 10};
  std::vector<float> expected_vals = {0, 0, 1, 0, 2, 3, 0, 4, 0, 0};

  std::vector<int64_t> inputDims = {3};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", inputDims, expected_dims);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool2D_WithPaddedOutput) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 2, 2};

  std::vector<int64_t> i_vals = {1, 3, 8, 10};
  std::vector<int64_t> i_dims = {1, 1, 2, 2};

  std::vector<int64_t> expected_dims = {1, 1, 5, 5};
  std::vector<float> expected_vals = {
      0, 1, 0, 2, 0,
      0, 0, 0, 0, 0,
      3, 0, 4, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0};

  std::vector<int64_t> inputDims = {4};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", inputDims, expected_dims);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool3D_WithPaddedOutput) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2, 2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2, 2});

  std::vector<float> t_vals = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> t_dims = {1, 1, 2, 2, 2};

  std::vector<int64_t> i_vals = {1, 3, 24, 30, 32, 38, 60, 62};
  std::vector<int64_t> i_dims = {1, 1, 2, 2, 2};

  std::vector<int64_t> expected_dims = {1, 1, 4, 4, 5};
  std::vector<int64_t> expectedDims_Size = {5};

  std::vector<float> expected_vals =
      {
          //slice 1
          0, 1, 0, 2, 0,
          0, 0, 0, 0, 0,
          0, 0, 0, 0, 0,
          0, 0, 0, 0, 0,

          // slice 2
          0, 0, 0, 0, 0,
          0, 0, 0, 0, 0,
          3, 0, 0, 0, 0,
          0, 0, 4, 0, 0,

          //slice 3
          5, 0, 0, 0, 0,
          0, 0, 6, 0, 0,
          0, 0, 0, 0, 0,
          0, 0, 0, 0, 0,

          // slice 4
          0, 0, 0, 0, 0,
          0, 0, 0, 0, 0,
          0, 0, 0, 0, 0,
          7, 0, 8, 0, 0};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", expectedDims_Size, expected_dims);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool_DefaultStrides) {
  OpTester test("MaxUnpool", 11);

  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> t_vals = {1, 2, 4, 8};
  std::vector<int64_t> t_dims = {1, 1, 4};

  std::vector<int64_t> i_vals = {1, 2, 3, 4};
  std::vector<int64_t> i_dims = {1, 1, 4};

  std::vector<int64_t> expected_dims = {1, 1, 5};
  std::vector<float> expected_vals = {0, 1, 2, 4, 8};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
