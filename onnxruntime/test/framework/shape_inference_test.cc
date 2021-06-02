// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <unordered_map>

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "test/framework/model_builder_utils.h"
#include "test/test_environment.h"

using namespace ONNX_NAMESPACE;
using namespace std;

namespace onnxruntime {
namespace test {

using namespace modelbuilder;

class ShapeInferenceTest : public ::testing::Test {
 protected:
  onnxruntime::Model model_;
  int node_count_;
  std::unordered_map<string, std::unique_ptr<onnxruntime::NodeArg>> name_to_arg_;

 public:
  ShapeInferenceTest() : model_("Test", false, DefaultLoggingManager().DefaultLogger()), node_count_(0) {}

  void Input(const std::string& name, const Type& type) {
    name_to_arg_[name] = std::make_unique<onnxruntime::NodeArg>(name, &type.value);
  }

  onnxruntime::NodeArg* Arg(const std::string& name) {
    if (name_to_arg_.count(name) == 0)
      name_to_arg_[name] = std::make_unique<onnxruntime::NodeArg>(name, nullptr);
    return name_to_arg_[name].get();
  }

  onnxruntime::Node& Node(const std::string& op, const std::string& input, const std::string& output) {
    std::vector<onnxruntime::NodeArg*> input_args({Arg(input)});
    std::vector<onnxruntime::NodeArg*> output_args({Arg(output)});
    int num = node_count_++;
    return model_.MainGraph().AddNode("node" + std::to_string(num), op, "test op", input_args, output_args);
  }

  void DoShapeInference() {
    auto status = model_.MainGraph().Resolve();
    EXPECT_TRUE(status.IsOK()) << "Graph resolve failed: " << status.ErrorMessage();
  }

  const TensorShapeProto* InputShape(onnxruntime::Node& node, int arg_num = 0) {
    return node.InputDefs()[arg_num]->Shape();
  }

  const TensorShapeProto* OutputShape(onnxruntime::Node& node, int arg_num = 0) {
    return node.OutputDefs()[arg_num]->Shape();
  }

  void CheckShapeEquality(const TensorShapeProto* shape1, const TensorShapeProto* shape2) {
    EXPECT_NE(shape1, nullptr);
    EXPECT_NE(shape2, nullptr);
    if ((shape1 != nullptr) && (shape2 != nullptr)) {
      EXPECT_EQ(shape1->dim_size(), shape2->dim_size()) << "Shapes do not have same rank";
      auto min_dims = std::min(shape1->dim_size(), shape2->dim_size());
      for (int i = 0; i < min_dims; ++i) {
        auto dim1 = shape1->dim(i);
        auto dim2 = shape2->dim(i);
        EXPECT_EQ(dim1.has_dim_value(), dim2.has_dim_value());
        if (dim1.has_dim_value()) {
          EXPECT_EQ(dim1.dim_value(), dim2.dim_value());
        }
        EXPECT_EQ(dim1.has_dim_param(), dim2.has_dim_param());
        if (dim1.has_dim_param()) {
          EXPECT_EQ(dim1.dim_param(), dim2.dim_param());
        }
      }
    }
  }

};  // namespace test

TEST_F(ShapeInferenceTest, BasicTest) {
  Type type1({1, 50, 100});
  Input("X1", type1);

  auto& node = Node("Cast", "X1", "Y1");
  //AttributeProto squeezed_axes;
  //squeezed_axes.set_name("axes");
  //squeezed_axes.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
  //squeezed_axes.add_ints(0);
  //p_node->AddAttribute("axes", squeezed_axes);
  AttributeProto cast_to;
  cast_to.set_name("to");
  cast_to.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  cast_to.set_i(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  //cast_to.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  //cast_to.set_s("INT16");
  node.AddAttribute("to", cast_to);

  DoShapeInference();
  // check inferred shapes
  Shape expected_shape({1, 50, 100});
  CheckShapeEquality(OutputShape(node), &expected_shape.value);
  CheckShapeEquality(InputShape(node), OutputShape(node));
}

}  // namespace test
}  // namespace onnxruntime
