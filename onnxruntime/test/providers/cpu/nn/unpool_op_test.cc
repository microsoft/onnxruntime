// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/graph/constants.h"
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
          // slice 1
          0, 1, 0, 2,
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,

          // slice 2
          0, 0, 0, 0,
          0, 0, 0, 0,
          3, 0, 0, 0,
          0, 0, 4, 0,

          // slice 3
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
          // slice 1
          0, 1, 0, 2,
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,

          // slice 2
          0, 0, 0, 0,
          0, 0, 0, 0,
          3, 0, 0, 0,
          0, 0, 4, 0,

          // slice 3
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

TEST(UnpoolTest, MaxUnpoolInvalidIndices) {
  OpTester test("MaxUnpool", 9);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 4};

  std::vector<int64_t> i_vals = {1, 3, 4, 8};  // 8 is out of bounds
  std::vector<int64_t> i_dims = {1, 1, 4};

  std::vector<int64_t> expected_dims = {1, 1, 8};
  std::vector<float> expected_vals = {0, 1, 0, 2, 3, 0, 4, 0};

  std::vector<int64_t> inputDims = {3};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", inputDims, expected_dims);

  test.AddOutput<float>("Y", expected_dims, expected_vals);
  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_provider;
  cpu_execution_provider.push_back(DefaultCpuExecutionProvider());
  test.Run(BaseTester::ExpectResult::kExpectFailure, "Index value out of bounds", {}, nullptr,
           &cpu_execution_provider);
}

// Rank-0 indices tensor should be rejected.
// The ONNX shape inference fix for this is in onnx/onnx#7997. Once that lands
// and we update the vendored ONNX, this test will verify the shape inference
// catches it. For now, shape inference on rank-0 indices is not safe, so we
// cannot test that path here without the upstream fix.

// Mismatched indices shape should be rejected at runtime by the kernel.
// DML does not produce the same error message for this validation, so we exclude it.
TEST(UnpoolTest, MaxUnpoolMismatchedIndicesShape) {
  OpTester test("MaxUnpool", 11);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 4};

  // Different spatial shape than X
  std::vector<int64_t> i_vals = {0, 1, 2, 3, 4, 5};
  std::vector<int64_t> i_dims = {1, 1, 6};

  // Use output_shape to avoid shape inference deducing from indices shape
  std::vector<int64_t> expected_dims = {1, 1, 8};
  std::vector<float> expected_vals(8, 0.f);
  std::vector<int64_t> output_shape_dims = {3};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", output_shape_dims, expected_dims);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  // DML validates input shapes differently and produces its own error messages.
  test.Run(BaseTester::ExpectResult::kExpectFailure,
           "Index tensor shape should be same as that of the input data tensor to unpool",
           {kDmlExecutionProvider});
}

// Rank-0 input X tensor should be rejected at runtime by the kernel.
// Shape inference is disabled because the upstream ONNX shape inference fix
// (onnx/onnx#7997) has not landed yet, and we need to validate the kernel check.
// DML does not produce the same error message for this validation, so we exclude it.
TEST(UnpoolTest, MaxUnpoolInvalidInputRank0) {
  OpTester test("MaxUnpool", 11);

  // Disable shape inference so we reach the kernel validation.
  test.AddShapeToTensorData(false);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  // Rank-0 input tensor (scalar)
  std::vector<float> t_vals = {1};
  std::vector<int64_t> t_dims = {};

  std::vector<int64_t> i_vals = {0};
  std::vector<int64_t> i_dims = {};

  std::vector<int64_t> expected_dims = {1, 1, 2};
  std::vector<float> expected_vals(2, 0.f);

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Input dimension cannot be less than 3",
           {kDmlExecutionProvider});
}

// Input with rank less than 3 should be rejected at runtime by the kernel.
// Shape inference is disabled to ensure we reach the kernel validation.
// DML does not produce the same error message for this validation, so we exclude it.
TEST(UnpoolTest, MaxUnpoolInvalidInputRank2) {
  OpTester test("MaxUnpool", 11);

  // Disable shape inference so we reach the kernel validation.
  test.AddShapeToTensorData(false);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  // Rank-2 input tensor (no spatial dimensions)
  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {2, 2};

  std::vector<int64_t> i_vals = {0, 1, 2, 3};
  std::vector<int64_t> i_dims = {2, 2};

  std::vector<int64_t> expected_dims = {2, 2};
  std::vector<float> expected_vals(4, 0.f);

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Input dimension cannot be less than 3",
           {kDmlExecutionProvider});
}

// Negative index should be rejected at runtime by the kernel.
// DML does not produce the same error message for this validation, so we exclude it.
TEST(UnpoolTest, MaxUnpoolNegativeIndex) {
  OpTester test("MaxUnpool", 11);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 4};

  std::vector<int64_t> i_vals = {-1, 3, 4, 6};  // -1 is invalid
  std::vector<int64_t> i_dims = {1, 1, 4};

  std::vector<int64_t> expected_dims = {1, 1, 8};
  std::vector<float> expected_vals(8, 0.f);

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(BaseTester::ExpectResult::kExpectFailure, "Index value out of bounds",
           {kDmlExecutionProvider});
}

// output_shape with wrong number of elements should be rejected.
// Shape inference is disabled to ensure we reach the kernel validation.
// DML does not produce the same error message for this validation, so we exclude it.
TEST(UnpoolTest, MaxUnpoolOutputShapeWrongElementCount) {
  OpTester test("MaxUnpool", 11);

  // Disable shape inference so we reach the kernel validation.
  test.AddShapeToTensorData(false);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 2, 2};  // rank 4

  std::vector<int64_t> i_vals = {1, 3, 4, 6};
  std::vector<int64_t> i_dims = {1, 1, 2, 2};

  // output_shape has 3 elements but X is rank 4
  std::vector<int64_t> output_shape_vals = {1, 4, 4};
  std::vector<int64_t> output_shape_dims = {3};

  std::vector<int64_t> expected_dims = {1, 1, 4, 4};
  std::vector<float> expected_vals(16, 0.f);

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", output_shape_dims, output_shape_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(BaseTester::ExpectResult::kExpectFailure,
           "output_shape must have the same number of elements as the rank of input",
           {kDmlExecutionProvider});
}

// Pads attribute with wrong number of elements should be rejected during kernel construction.
// Shape inference is disabled to reach the kernel validation.
// DML does not produce the same error message for this validation, so we exclude it.
TEST(UnpoolTest, MaxUnpoolInvalidPadsSize) {
  OpTester test("MaxUnpool", 11);

  test.AddShapeToTensorData(false);

  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  // pads should have 4 elements (2 * kernel_shape.size()) but we provide 3
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 2, 2};

  std::vector<int64_t> i_vals = {0, 1, 2, 3};
  std::vector<int64_t> i_dims = {1, 1, 2, 2};

  std::vector<int64_t> expected_dims = {1, 1, 4, 4};
  std::vector<float> expected_vals(16, 0.f);

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Pads attribute size must be twice the kernel_shape size",
           {kDmlExecutionProvider});
}

// Computed output dimension is not positive due to large pads.
// Shape inference is disabled to reach the kernel validation.
// DML does not produce the same error message for this validation, so we exclude it.
TEST(UnpoolTest, MaxUnpoolNegativeComputedDimension) {
  OpTester test("MaxUnpool", 11);

  test.AddShapeToTensorData(false);

  test.AddAttribute("kernel_shape", vector<int64_t>{2});
  test.AddAttribute("strides", std::vector<int64_t>{1});
  // pads sum (1+1=2) makes dim_value = (1-1)*1 - (1+1) + 2 = 0, which is not positive
  test.AddAttribute("pads", vector<int64_t>{1, 1});

  std::vector<float> t_vals = {1};
  std::vector<int64_t> t_dims = {1, 1, 1};

  std::vector<int64_t> i_vals = {0};
  std::vector<int64_t> i_dims = {1, 1, 1};

  std::vector<int64_t> expected_dims = {1, 1, 1};
  std::vector<float> expected_vals = {0.f};

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Computed output dimension is not positive",
           {kDmlExecutionProvider});
}

// output_shape has correct element count but spatial dims smaller than inferred minimum.
// DML does not produce the same error message for this validation, so we exclude it.
TEST(UnpoolTest, MaxUnpoolOutputShapeSmallerThanMinimum) {
  OpTester test("MaxUnpool", 11);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  // X is 1x1x4, inferred output = 1x1x8
  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 4};

  std::vector<int64_t> i_vals = {0, 1, 2, 3};
  std::vector<int64_t> i_dims = {1, 1, 4};

  // output_shape = {1, 1, 4} — correct element count (3 for rank-3) but 4 < 8
  std::vector<int64_t> output_shape_vals = {1, 1, 4};
  std::vector<int64_t> output_shape_dims = {3};

  std::vector<int64_t> expected_dims = {1, 1, 4};
  std::vector<float> expected_vals(4, 0.f);

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", output_shape_dims, output_shape_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.Run(BaseTester::ExpectResult::kExpectFailure,
           "output_shape is smaller than minimum required",
           {kDmlExecutionProvider});
}

TEST(UnpoolTest, MaxUnpoolOutputShapeBatchChannelMismatch) {
  OpTester test("MaxUnpool", 11);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  // X is 1x1x4, inferred output = 1x1x8
  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 4};

  std::vector<int64_t> i_vals = {0, 1, 2, 3};
  std::vector<int64_t> i_dims = {1, 1, 4};

  // output_shape has different batch/channel dims than X
  std::vector<int64_t> output_shape_vals = {2, 1, 8};
  std::vector<int64_t> output_shape_dims = {3};

  std::vector<int64_t> expected_dims = {2, 1, 8};
  std::vector<float> expected_vals(16, 0.f);

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", output_shape_dims, output_shape_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);
  test.AddShapeToTensorData(false);
  test.Run(BaseTester::ExpectResult::kExpectFailure,
           "output_shape batch and channel dimensions must match input",
           {kDmlExecutionProvider});
}

#ifdef USE_DML
// DML-specific tests: verify that DML rejects invalid inputs.
// Empty expected error string is intentional — DML produces its own error messages
// that differ from the CPU kernel, and we cannot capture them without a DML device.
// These tests confirm that invalid inputs are rejected rather than causing undefined behavior.

TEST(UnpoolTest, MaxUnpoolInvalidInputRank2_DML) {
  OpTester test("MaxUnpool", 11);

  test.AddShapeToTensorData(false);

  test.AddAttribute("strides", std::vector<int64_t>{2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {2, 2};

  std::vector<int64_t> i_vals = {0, 1, 2, 3};
  std::vector<int64_t> i_dims = {2, 2};

  std::vector<int64_t> expected_dims = {2, 2};
  std::vector<float> expected_vals(4, 0.f);

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);

  std::vector<std::unique_ptr<IExecutionProvider>> dml_ep;
  dml_ep.push_back(DefaultDmlExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &dml_ep);
}

TEST(UnpoolTest, MaxUnpoolOutputShapeWrongElementCount_DML) {
  OpTester test("MaxUnpool", 11);

  test.AddShapeToTensorData(false);

  test.AddAttribute("strides", std::vector<int64_t>{2, 2});
  test.AddAttribute("kernel_shape", vector<int64_t>{2, 2});

  std::vector<float> t_vals = {1, 2, 3, 4};
  std::vector<int64_t> t_dims = {1, 1, 2, 2};

  std::vector<int64_t> i_vals = {1, 3, 4, 6};
  std::vector<int64_t> i_dims = {1, 1, 2, 2};

  std::vector<int64_t> output_shape_vals = {1, 4, 4};
  std::vector<int64_t> output_shape_dims = {3};

  std::vector<int64_t> expected_dims = {1, 1, 4, 4};
  std::vector<float> expected_vals(16, 0.f);

  test.AddInput<float>("xT", t_dims, t_vals);
  test.AddInput<int64_t>("xI", i_dims, i_vals);
  test.AddInput<int64_t>("output_shape", output_shape_dims, output_shape_vals);
  test.AddOutput<float>("Y", expected_dims, expected_vals);

  std::vector<std::unique_ptr<IExecutionProvider>> dml_ep;
  dml_ep.push_back(DefaultDmlExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &dml_ep);
}

// Note: No DML negative index test — DML does not reject negative indices at runtime.
#endif  // USE_DML

}  // namespace test
}  // namespace onnxruntime
