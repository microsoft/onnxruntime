// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

struct MulIntegerInput {
  std::string name;
  int elem_type;
  std::vector<int64_t> shape;
};

void CheckMulIntegerInference(const std::vector<MulIntegerInput>& inputs,
                              const std::vector<int64_t>& expected_shape,
                              bool expect_success = true) {
  Model model("mul_integer_shape_inference", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<NodeArg*> input_args;
  for (const auto& input : inputs) {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor_type = type.mutable_tensor_type();
    tensor_type->set_elem_type(input.elem_type);
    auto* shape = tensor_type->mutable_shape();
    for (int64_t dim : input.shape) {
      shape->add_dim()->set_dim_value(dim);
    }
    input_args.push_back(&graph.GetOrCreateNodeArg(input.name, input.name.empty() ? nullptr : &type));
  }

  // Leave the output untyped so both its element type and shape must be inferred.
  auto& output = graph.GetOrCreateNodeArg("C", nullptr);
  graph.AddNode("mul", "MulInteger", "", input_args, {&output}, nullptr, kMSDomain);
  graph.SetOutputs({&output});
  const auto status = graph.Resolve();
  if (!expect_success) {
    EXPECT_FALSE(status.IsOK()) << "Invalid MulInteger inputs were accepted";
    return;
  }

  ASSERT_STATUS_OK(status);
  const auto* output_type = output.TypeAsProto();
  ASSERT_NE(output_type, nullptr);
  ASSERT_TRUE(output_type->has_tensor_type());
  EXPECT_EQ(output_type->tensor_type().elem_type(), ONNX_NAMESPACE::TensorProto::INT32);
  ASSERT_TRUE(output_type->tensor_type().has_shape());
  const auto& shape = output_type->tensor_type().shape();
  ASSERT_EQ(shape.dim_size(), expected_shape.size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    ASSERT_TRUE(shape.dim(i).has_dim_value());
    EXPECT_EQ(shape.dim(i).dim_value(), expected_shape[i]);
  }
}

class MulIntegerShapeInferenceTest : public ::testing::TestWithParam<int> {};

TEST_P(MulIntegerShapeInferenceTest, OptionalZeroPoints) {
  const int type = GetParam();
  const MulIntegerInput empty{"", type, {}};
  const MulIntegerInput az{"az", type, {}};
  const MulIntegerInput bz{"bz", type, {}};
  struct Shapes {
    std::vector<int64_t> a;
    std::vector<int64_t> b;
    std::vector<int64_t> output;
  };
  const std::vector<Shapes> shapes{
      {{2, 1}, {1, 3}, {2, 3}},
      {{2, 3}, {2, 3}, {2, 3}},
      {{}, {2, 3}, {2, 3}},
      {{2, 3}, {}, {2, 3}},
      {{}, {}, {}},
      {{0, 1}, {1, 3}, {0, 3}},
  };
  for (const auto& shape : shapes) {
    const MulIntegerInput a{"a", type, shape.a};
    const MulIntegerInput b{"b", type, shape.b};
    const std::vector<std::vector<MulIntegerInput>> forms{
        {a, empty, b},
        {a, empty, b, empty},
        {a, az, b},
        {a, az, b, empty},
        {a, empty, b, bz},
        {a, az, b, bz},
    };
    for (size_t i = 0; i < forms.size(); ++i) {
      SCOPED_TRACE(::testing::Message() << "input form " << i
                                        << ", A shape " << ::testing::PrintToString(shape.a)
                                        << ", B shape " << ::testing::PrintToString(shape.b));
      CheckMulIntegerInference(forms[i], shape.output);
    }
  }
}

#if !defined(ORT_NO_EXCEPTIONS)
TEST_P(MulIntegerShapeInferenceTest, RejectsInvalidPresentZeroPoints) {
  const int type = GetParam();
  const int other_type = type == ONNX_NAMESPACE::TensorProto::INT8
                             ? ONNX_NAMESPACE::TensorProto::UINT8
                             : ONNX_NAMESPACE::TensorProto::INT8;
  const MulIntegerInput a{"a", type, {2, 1}};
  const MulIntegerInput b{"b", type, {1, 3}};
  const MulIntegerInput az{"az", type, {}};
  const MulIntegerInput bz{"bz", type, {}};
  const MulIntegerInput empty{"", type, {}};
  for (int slot : {1, 3}) {
    for (int invalid_type : {other_type, static_cast<int>(ONNX_NAMESPACE::TensorProto::FLOAT)}) {
      SCOPED_TRACE(::testing::Message() << "zero point slot " << slot << ", type " << invalid_type);
      std::vector<MulIntegerInput> inputs{a, az, b, bz};
      inputs[slot].elem_type = invalid_type;
      CheckMulIntegerInference(inputs, {}, false);
    }
    for (const auto& invalid_shape : std::vector<std::vector<int64_t>>{{1}, {2}, {1, 1}}) {
      SCOPED_TRACE(::testing::Message() << "zero point slot " << slot
                                        << ", shape " << ::testing::PrintToString(invalid_shape));
      std::vector<MulIntegerInput> inputs{a, az, b, bz};
      inputs[slot].shape = invalid_shape;
      CheckMulIntegerInference(inputs, {}, false);
      // An omitted peer must not disable validation of the present zero point.
      inputs[slot == 1 ? 3 : 1] = empty;
      CheckMulIntegerInference(inputs, {}, false);
    }
  }
}

TEST_P(MulIntegerShapeInferenceTest, RejectsInvalidRequiredInputs) {
  const int type = GetParam();
  const int other_type = type == ONNX_NAMESPACE::TensorProto::INT8
                             ? ONNX_NAMESPACE::TensorProto::UINT8
                             : ONNX_NAMESPACE::TensorProto::INT8;
  const MulIntegerInput a{"a", type, {2, 1}};
  const MulIntegerInput b{"b", type, {1, 3}};
  const MulIntegerInput empty{"", type, {}};
  for (int slot : {0, 2}) {
    SCOPED_TRACE(::testing::Message() << "required input slot " << slot);
    for (int invalid_type : {other_type, static_cast<int>(ONNX_NAMESPACE::TensorProto::FLOAT)}) {
      std::vector<MulIntegerInput> inputs{a, empty, b};
      inputs[slot].elem_type = invalid_type;
      CheckMulIntegerInference(inputs, {}, false);
    }
    std::vector<MulIntegerInput> inputs{a, empty, b};
    inputs[slot] = empty;
    CheckMulIntegerInference(inputs, {}, false);
  }
  CheckMulIntegerInference({a}, {}, false);
  CheckMulIntegerInference({a, empty, {"b", type, {3, 2}}}, {}, false);
}
#endif

INSTANTIATE_TEST_SUITE_P(ByteTypes, MulIntegerShapeInferenceTest,
                         ::testing::Values(ONNX_NAMESPACE::TensorProto::INT8,
                                           ONNX_NAMESPACE::TensorProto::UINT8));

}  // namespace
}  // namespace test
}  // namespace onnxruntime
