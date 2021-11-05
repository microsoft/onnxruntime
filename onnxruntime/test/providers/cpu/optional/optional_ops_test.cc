// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(OptionalOpTest, OptionalTensorCreateFromTensor) {
  OpTester test("Optional", 15);

  std::initializer_list<float> data = {-1.0856307f, 0.99734545f};

  test.AddInput<float>("A", {2}, data);
  test.AddOptionalTypeTensorOutput<float>("Y", {2}, &data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}

TEST(OptionalOpTest, OptionalSeqTensorCreateFromSeqTensor) {
  OpTester test("Optional", 15);

  SeqTensors<float> data;
  data.AddTensor({3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  data.AddTensor({3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  test.AddSeqInput("S", data);
  test.AddOptionalTypeSeqOutput<float>("Y", &data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}
TEST(OptionalOpTest, OptionalTensorCreateFromTypeProto) {
  OpTester test("Optional", 15);

  onnx::TypeProto tp;
  tp.mutable_tensor_type()
      ->set_elem_type(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT);

  test.AddAttribute<onnx::TypeProto>("type", tp);
  // expected value is nullptr because we expect a "None" output
  test.AddOptionalTypeTensorOutput<float>("Y", {}, nullptr);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}

TEST(OptionalOpTest, OptionalSeqTensorCreateFromTypeProto) {
  OpTester test("Optional", 15);

  onnx::TypeProto tensor_tp;
  tensor_tp.mutable_tensor_type()
      ->set_elem_type(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT);

  onnx::TypeProto tp;
  tp.mutable_sequence_type()->mutable_elem_type()->CopyFrom(tensor_tp);

  test.AddAttribute<onnx::TypeProto>("type", tp);
  // expected value is nullptr because we expect a "None" output
  test.AddOptionalTypeSeqOutput<float>("Y", nullptr);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}

TEST(OptionalOpTest, OptionalTensorHasElement_True) {
  OpTester test("OptionalHasElement", 15);

  std::initializer_list<float> data = {-1.0856307f, 0.99734545f};

  test.AddOptionalTypeTensorInput<float>("A", {2}, &data);
  test.AddOutput<bool>("Y", {}, {true});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}

TEST(OptionalOpTest, OptionalSeqTensorHasElement_True) {
  OpTester test("OptionalHasElement", 15);

  SeqTensors<float> data;
  data.AddTensor({3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  data.AddTensor({3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  test.AddOptionalTypeSeqInput<float>("A", &data);
  test.AddOutput<bool>("Y", {}, {true});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}

TEST(OptionalOpTest, OptionalTensorHasElement_False) {
  OpTester test("OptionalHasElement", 15);

  // Input is an optional type and is None
  test.AddOptionalTypeTensorInput<float>("A", {}, nullptr);
  test.AddOutput<bool>("Y", {}, {false});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}

TEST(OptionalOpTest, OptionalSeqTensorHasElement_False) {
  OpTester test("OptionalHasElement", 15);

  // Input is an optional type and is None
  test.AddOptionalTypeSeqInput<float>("A", nullptr);
  test.AddOutput<bool>("Y", {}, {false});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}

TEST(OptionalOpTest, OptionalTensorGetElement) {
  OpTester test("OptionalGetElement", 15);

  std::initializer_list<float> data = {-1.0856307f, 0.99734545f};

  test.AddOptionalTypeTensorInput<float>("A", {2}, &data);
  test.AddOutput<float>("Y", {2}, {-1.0856307f, 0.99734545f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}

TEST(OptionalOpTest, OptionalSeqTensorGetElement) {
  OpTester test("OptionalGetElement", 15);

  SeqTensors<float> data;
  data.AddTensor({3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  data.AddTensor({3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddOptionalTypeSeqInput<float>("A", &data);
  test.AddSeqOutput("S", data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}

class OptionalOpTester : public OpTester {
 public:
  explicit OptionalOpTester(bool is_seq_tensor = false)
      : OpTester("Optional", 15), is_seq_tensor_(is_seq_tensor) {}

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    /*
    * Both the Optional* ops produce intermediate outputs
    
                      Graph Input  
                          |
                          |
                      Optional
                          |
                          |
                      OptionalGetElement
                         |
                         |
                       Shape
                         |
                         | 
                      Graph output
      */

    onnx::TypeProto tensor_type_proto;
    tensor_type_proto.mutable_tensor_type()
        ->set_elem_type(onnx::TensorProto_DataType::TensorProto_DataType_FLOAT);

    onnx::TypeProto seq_tensor_type_proto;
    if (is_seq_tensor_) {
      seq_tensor_type_proto.mutable_sequence_type()->mutable_elem_type()->CopyFrom(tensor_type_proto);
    }

    onnx::TypeProto optional_type_proto;
    optional_type_proto.mutable_optional_type()->mutable_elem_type()->CopyFrom(
        is_seq_tensor_ ? seq_tensor_type_proto : tensor_type_proto);

    auto& optional_node_arg = graph.GetOrCreateNodeArg("optional_output", &optional_type_proto);
    ORT_IGNORE_RETURN_VALUE(graph.AddNode("optional_create", "Optional", "Create optional type",
                                          {graph_input_defs[0]}, {&optional_node_arg},
                                          nullptr));

    auto& optional_parsed_node_arg = graph.GetOrCreateNodeArg("parsed_output",
                                                              is_seq_tensor_
                                                                  ? &seq_tensor_type_proto
                                                                  : &tensor_type_proto);

    ORT_IGNORE_RETURN_VALUE(graph.AddNode("optional_get_element", "OptionalGetElement",
                                          "Parse optional type",
                                          {&optional_node_arg}, {&optional_parsed_node_arg},
                                          nullptr));

    if (!is_seq_tensor_) {
      ORT_IGNORE_RETURN_VALUE(graph.AddNode("Size", "Size",
                                            "Size",
                                            {&optional_parsed_node_arg}, {graph_output_defs[0]},
                                            nullptr));
    } else {
      ORT_IGNORE_RETURN_VALUE(graph.AddNode("SequenceLength", "SequenceLength",
                                            "SequenceLength",
                                            {&optional_parsed_node_arg}, {graph_output_defs[0]},
                                            nullptr));
    }
  }

 private:
  bool is_seq_tensor_;
};

TEST(OptionalOpTest, OptionalOpsValidateOrtValueReUseForOptionalTensors) {
  // We create a simple model with 2 optional ops - Optional and OptionalGetElement
  // that don't produce graph outputs and we have logic in the allocation planner to
  // re-use input OrtValues as output OrtValues for such cases. We test that the model
  // executes fine with such logic.
  // The allocation planner already tests the re-use of inputs as outputs if the kernel def
  // requests that, so there is no need to test that re-use took place here.
  // We are just triggering that behavior in the allocation planner by ensuring
  // that 'OptionalGetElement' op doesn't produce a graph output, in which case,
  // input re-use wouldn't have happened.

  OptionalOpTester test;

  std::initializer_list<float> data = {1.f, 2.f};

  test.AddInput<float>("A", {2}, data);
  test.AddOutput<int64_t>("Y", {}, {2});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}

TEST(OptionalOpTest, OptionalOpsValidateOrtValueReUseForOptionalTensorSequence) {
  // We create a simple model with 2 optional ops - Optional and OptionalGetElement
  // that don't produce graph outputs and we have logic in the allocation planner to
  // re-use input OrtValues as output OrtValues for such cases. We test that the model
  // executes fine with such logic.
  // The allocation planner already tests the re-use of inputs as outputs if the kernel def
  // requests that, so there is no need to test that re-use took place here.
  // We are just triggering that behavior in the allocation planner by ensuring
  // that 'OptionalGetElement' op doesn't produce a graph output, in which case,
  // input re-use wouldn't have happened.
  OptionalOpTester test(true);

  SeqTensors<float> data;
  data.AddTensor({3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  test.AddSeqInput("S", data);
  test.AddOutput<int64_t>("Y", {}, {1});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: unsupported ONNX type
}
}  // namespace test
}  // namespace onnxruntime
