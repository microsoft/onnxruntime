// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

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
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1876): The parameter is incorrect.";
  }

  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2, 2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2, 2});

  // NOTE: This input doesn't make sense as MaxPool output, but strictly speaking it doesn't need to be
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
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

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
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

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

TEST(UnpoolTest, MaxUnPool1D_WithOutputShape) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 4};

  std::vector<int64_t> i_vals = {1, 3, 4, 6};
  std::vector<int64_t> i_dims = {1, 1, 4};

  std::vector<int64_t> expected_dims = {1, 1, 9};
  std::vector<float> expected_vals = {0, 1, 0, 2, 3, 0, 4, 0, 0};

  std::vector<int64_t> expected_dim_size = {3};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", expected_dim_size, expected_dims);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool2D_WithOutputShape) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 2, 2};

  std::vector<int64_t> i_vals = {1, 3, 10, 12};
  std::vector<int64_t> i_dims = {1, 1, 2, 2};

  std::vector<int64_t> expected_dims = {1, 1, 5, 5};
  std::vector<float> expected_vals = {
      0, 1, 0, 2, 0,
      0, 0, 0, 0, 0,
      3, 0, 4, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0};

  std::vector<int64_t> expected_dims_size = {4};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", expected_dims_size, expected_dims);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run();
}

TEST(UnpoolTest, MaxUnPool3D_WithOutputShape) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1876): The parameter is incorrect.";
  }

  OpTester test("MaxUnpool", 9);
  // original input 1, 1, 3, 3, 3
  // with these strides and kernel shape there should only be one value
  /* Python to check the MaxPool output that is the theoretical input to the MaxUnpool in this test.
import onnx
from onnx import helper, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
import onnxruntime

graph = helper.make_graph(
    [helper.make_node("MaxPool", inputs=["X"], outputs=["Y", "indices"], name="MaxPool", kernel_shape=(2,2,2), strides=(2,2,2))],
    "the graph",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 1, 3, 3, 3))],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
     helper.make_tensor_value_info("indices", TensorProto.INT64, None)]
)

# Create the model (ModelProto)
model_def = helper.make_model(graph, producer_name='me')
onnx.save(model_def, "maxpool_model.onnx")
sess = onnxruntime.InferenceSession("maxpool_model.onnx")
X = np.arange(0, 27, dtype=np.float32).reshape((1, 1, 3, 3, 3))
print(X)
(Y, indices) = sess.run(None, {"X" : X})
print(Y)
print(indices)

  */

  test.AddAttribute("strides", std::vector<int64_t>{2, 2, 2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2, 2});

  std::vector<float> t_vals = {3};
  std::vector<int64_t> t_dims = {1, 1, 1, 1, 1};

  std::vector<int64_t> i_vals = {13};
  std::vector<int64_t> i_dims = {1, 1, 1, 1, 1};

  std::vector<int64_t> expected_dims = {1, 1, 3, 3, 3};

  std::vector<float> expected_vals = {
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,

      0, 0, 0,
      0, 3, 0,
      0, 0, 0,

      0, 0, 0,
      0, 0, 0,
      0, 0, 0};

  std::vector<int64_t> expected_dims_size = {5};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", expected_dims_size, expected_dims);

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
