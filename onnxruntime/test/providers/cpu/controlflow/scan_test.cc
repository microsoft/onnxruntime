// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"
// #include "core/framework/customregistry.h"
#include "core/framework/session_state.h"
#include "test/providers/provider_test_utils.h"
#include "core/session/inference_session.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

struct RunOptions {
  bool include_shapes_in_main_graph = false;

  bool include_shapes_in_subgraph = true;
  bool include_types_in_subgraph = true;

  bool include_outer_scope_add = false;
};

static void CreateSubgraph(Graph& graph, RunOptions options);

static const float kOuterNodeAddValue = 42.f;

class ScanOpTester : public OpTester {
 public:
  ScanOpTester() : OpTester("Scan", 8) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    // add outer_scope_0 node
    {
      TypeProto float_scalar;
      float_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
      auto mutable_dim = float_scalar.mutable_tensor_type()->mutable_shape()->add_dim();
      mutable_dim->set_dim_value(1);

      {
        auto& output_arg = graph.GetOrCreateNodeArg("outer_scope_0", &float_scalar);
        auto* constant = graph.AddNode("outer_scope_constant", "Constant", "Constant with value -1", {}, {&output_arg});

        TensorProto value_tensor;
        value_tensor.add_dims(1);
        value_tensor.add_float_data(kOuterNodeAddValue);
        value_tensor.set_data_type(onnx::TensorProto_DataType_FLOAT);

        constant->AddAttribute("value", value_tensor);
      }
    }

    // call base implementation to add the Scan node as per usual
    OpTester::AddNodes(graph, graph_input_defs, graph_output_defs, add_attribute_funcs);
  }
};

static void CreateSubgraph(Graph& graph, RunOptions options) {
  bool include_shapes = options.include_shapes_in_subgraph;
  bool include_types = options.include_types_in_subgraph;

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  /* Subgraph looks like this. 

    [constant_1]  loop_state_in_1             concat_in_0      concat_in_1
            \           |                                 \     /
             \--------[Add]                               [Concat]
                        |                                    |
                        |                                 concat_out_1
                        |                                    |
                        |                                  [Add]----------outer_scope_0   < Optional
                        |                                    |                            < based on RunOptions
                        |                                add_out_1                        <
                        |                                    |                            <
                        |                                 [Split]
                        |                               /  |   |   \
                   loop_state_out_1           split_out_0   ...     split_out_3
  */

  // constant of 1 to add to the loop state on each iteration
  {
    inputs = {};
    outputs = {};

    TypeProto float_scalar;
    // inputs must have type information and a rank
    float_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    auto mutable_dim = float_scalar.mutable_tensor_type()->mutable_shape()->add_dim();
    if (include_shapes)
      mutable_dim->set_dim_value(1);

    {
      auto& output_arg = graph.GetOrCreateNodeArg("constant_1", &float_scalar);
      outputs.push_back(&output_arg);

      auto* constant = graph.AddNode("constant", "Constant", "Constant with value 1", inputs, outputs);

      TensorProto value_tensor;
      value_tensor.add_dims(1);
      value_tensor.add_float_data(1.f);
      value_tensor.set_data_type(onnx::TensorProto_DataType_FLOAT);

      constant->AddAttribute("value", value_tensor);
    }

    inputs = outputs;  // start with output from Constant node
    outputs = {};

    auto& input_arg = graph.GetOrCreateNodeArg("loop_state_in_1", &float_scalar);
    inputs.push_back(&input_arg);

    TypeProto loop_state_output_tensor;
    loop_state_output_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

    // as this is a subgraph output we need a shape to come from somewhere, so if the main graph isn't providing it,
    // it has to come from here.
    bool type_and_shape_required = options.include_shapes_in_main_graph == false;

    if (include_shapes || type_and_shape_required)
      loop_state_output_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto* type_proto = include_types || type_and_shape_required ? &loop_state_output_tensor : nullptr;
    auto& output_arg = graph.GetOrCreateNodeArg("loop_state_out_1", type_proto);
    outputs.push_back(&output_arg);

    auto* add = graph.AddNode("add", "Add", "Add 1 to the loop state", inputs, outputs);
    (void)add;
  }

  // subgraph with multiple inputs and outputs to test variadic behaviour.
  // 2 inputs of 2 that are concatenated and then split into 4 outputs of 1

  // Concat node
  {
    inputs = {};
    outputs = {};

    // input of 2 x {2} tensors
    TypeProto concat_input_tensor;
    // inputs must have type information and rank, but dimension can have no value if we're not providing shape info.
    concat_input_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    auto mutable_dim = concat_input_tensor.mutable_tensor_type()->mutable_shape()->add_dim();
    if (include_shapes)
      mutable_dim->set_dim_value(2);

    for (int i = 0, num_inputs = 2; i < num_inputs; ++i) {
      auto& input_arg = graph.GetOrCreateNodeArg("concat_in_" + std::to_string(i), &concat_input_tensor);
      inputs.push_back(&input_arg);
    }

    // one output from concatenate of {4} tensor
    TypeProto concat_output_tensor;
    concat_output_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    if (include_shapes)
      concat_output_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);

    TypeProto* type_proto = include_types ? &concat_output_tensor : nullptr;
    auto& output_arg = graph.GetOrCreateNodeArg("concat_out_1", type_proto);
    outputs.push_back(&output_arg);

    auto* concat = graph.AddNode("concat", "Concat", "concat 2 inputs", inputs, outputs);

    concat->AddAttribute("axis", int64_t{0});
  }

  // Post-Concat Add Node
  if (options.include_outer_scope_add) {
    TypeProto float_scalar;
    float_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    float_scalar.mutable_tensor_type()->mutable_shape();  // create it to make checker happy but don't put any dims in it

    auto& outer_scope_input_arg = graph.GetOrCreateNodeArg("outer_scope_0", &float_scalar);

    // add so that we don't end up with it being considered a graph input
    graph.AddOuterScopeNodeArg(outer_scope_input_arg.Name());

    inputs = outputs;
    outputs = {};

    inputs.push_back(&outer_scope_input_arg);

    TypeProto post_concat_add_output_tensor;  // should be same type/shape as concat_out_0
    post_concat_add_output_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    post_concat_add_output_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);

    auto& output_arg = graph.GetOrCreateNodeArg("add_out_1", &post_concat_add_output_tensor);
    outputs.push_back(&output_arg);

    graph.AddNode("post_concat_add", "Add", "Add outer scope value to concat output", inputs, outputs);
  }

  // Split node
  {
    // setup Split to run using the Concat output
    inputs = outputs;
    outputs = {};

    // split output of 4 x {1} tensors
    TypeProto split_output_tensor;
    split_output_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    // the Split shape inferencing assumes that the shape is already set to each value in the
    // 'split' attribute. that seems reasonable (if you set that attribute you can set the shape)
    // so we must set the shape here.
    split_output_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    for (int i = 0, num_outputs = 4; i < num_outputs; ++i) {
      auto& output_arg = graph.GetOrCreateNodeArg("split_out_" + std::to_string(i), &split_output_tensor);
      outputs.push_back(&output_arg);
    }

    auto* split = graph.AddNode("split", "Split", "split into 4 outputs", inputs, outputs);
    split->AddAttribute("axis", int64_t{0});
    split->AddAttribute("split", std::vector<int64_t>{1, 1, 1, 1});
  }

  auto status = graph.Resolve();

  EXPECT_EQ(status, Status::OK());
}

void RunTest(const std::string test_name, int64_t batch_size, int64_t max_sequence_len, int64_t input_size,
             std::vector<int64_t>* directions,
             std::vector<int64_t>* sequence_lens,
             std::vector<float>& loop_state_in_0,
             std::vector<float> input_0,
             std::vector<float> input_1,
             std::vector<float>& loop_state_out_0,
             std::vector<float> output_0,
             std::vector<float> output_1,
             std::vector<float> output_2,
             std::vector<float> output_3,
             RunOptions options = {},
             OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
             const std::string& failure_message = "") {
  // create model that will be used to initialize subgraph. currently there's no direct way to create a Graph instance.
  Model model(test_name);
  auto& graph = model.MainGraph();
  CreateSubgraph(graph, options);
  auto& proto = graph.ToGraphProto();

  ScanOpTester test;

  test.AddShapeToTensorData(options.include_shapes_in_main_graph);

  test.AddAttribute("body", proto);
  test.AddAttribute<int64_t>("num_scan_inputs", 2);

  if (directions != nullptr) {
    test.AddAttribute<std::vector<int64_t>>("directions", *directions);
  }

  if (sequence_lens == nullptr) {
    test.AddMissingOptionalInput<int64_t>();
  } else {
    std::vector<int64_t> sequence_lens_dims{batch_size};
    test.AddInput<int64_t>("sequence_lens", sequence_lens_dims, *sequence_lens);
  }

  test.AddInput<float>("scan_loop_state_in_0", {batch_size, 1}, loop_state_in_0);

  std::vector<int64_t> input_shape{batch_size, max_sequence_len, input_size};
  test.AddInput<float>("scan_input_0", input_shape, input_0);
  test.AddInput<float>("scan_input_1", input_shape, input_1);

  test.AddOutput<float>("scan_loop_state_out_0", {batch_size, 1}, loop_state_out_0);

  std::vector<int64_t> output_shape{batch_size, max_sequence_len, 1};
  test.AddOutput<float>("scan_output_0", output_shape, output_0);
  test.AddOutput<float>("scan_output_1", output_shape, output_1);
  test.AddOutput<float>("scan_output_2", output_shape, output_2);
  test.AddOutput<float>("scan_output_3", output_shape, output_3);

  test.Run(expect_result, failure_message);
}

void ShortSequenceOneInBatchOneLoopStateVar(RunOptions options) {
  const int64_t batch_size = 1;
  const int64_t sequence_len = 2;
  const int64_t input_size = 2;

  std::vector<float> iteration_count_in{0.f};

  // batch_size, max_sequence_len, input_size
  std::vector<float> input_0{1.f, 2.f,
                             4.f, 3.f};
  std::vector<float> input_1{3.f, 4.f,
                             2.f, 1.f};

  std::vector<float> iteration_count_out{2.f};  // iteration_count_in + 1 for each item in sequence

  float output_adjust = options.include_outer_scope_add ? kOuterNodeAddValue : 0.f;

  // batch_size, max_sequence_len, 1
  std::vector<float> output_0{1.f + output_adjust, 4.f + output_adjust};
  std::vector<float> output_1{2.f + output_adjust, 3.f + output_adjust};
  std::vector<float> output_2{3.f + output_adjust, 2.f + output_adjust};
  std::vector<float> output_3{4.f + output_adjust, 1.f + output_adjust};

  RunTest("ShortSequenceOneInBatchOneLoopStateVar", batch_size, sequence_len, input_size,
          nullptr, nullptr,
          iteration_count_in, input_0, input_1,
          iteration_count_out, output_0, output_1, output_2, output_3,
          options);
}

TEST(Scan, ShortSequenceOneInBatchOneLoopStateVar_NoShapeInMainGraph_TypeAndShapeInSubgraph) {
  RunOptions options;
  options.include_shapes_in_main_graph = false;
  options.include_types_in_subgraph = true;
  options.include_shapes_in_subgraph = true;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST(Scan, ShortSequenceOneInBatchOneLoopStateVar_ShapeInMainGraph_NoTypeAndShapeInSubgraph) {
  RunOptions options;
  options.include_shapes_in_main_graph = true;
  options.include_types_in_subgraph = false;
  options.include_shapes_in_subgraph = false;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST(Scan, ShortSequenceOneInBatchOneLoopStateVar_NoShapeInMainGraph_NoTypeAndShapeInSubgraph) {
  RunOptions options;
  options.include_shapes_in_main_graph = false;
  options.include_types_in_subgraph = false;
  options.include_shapes_in_subgraph = false;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

// test when there is an operator in the subgraph that uses a value coming from outer scope
TEST(Scan, OuterScopeAccess_NoShapeInMainGraph_TypeAndShapeInSubgraph) {
  RunOptions options;
  options.include_shapes_in_main_graph = false;
  options.include_types_in_subgraph = true;
  options.include_shapes_in_subgraph = true;

  options.include_outer_scope_add = true;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST(Scan, OuterScopeAccess_ShapeInMainGraph_NoTypeAndShapeInSubgraph) {
  RunOptions options;
  options.include_shapes_in_main_graph = true;
  options.include_types_in_subgraph = false;
  options.include_shapes_in_subgraph = false;

  options.include_outer_scope_add = true;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST(Scan, OuterScopeAccess_NoShapeInMainGraph_NoTypeAndShapeInSubgraph) {
  RunOptions options;
  options.include_shapes_in_main_graph = false;
  options.include_types_in_subgraph = false;
  options.include_shapes_in_subgraph = false;

  options.include_outer_scope_add = true;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST(Scan, ShortSequenceTwoInBatchOneLoopStateVar) {
  const int64_t batch_size = 2;
  const int64_t sequence_len = 2;
  const int64_t input_size = 2;

  std::vector<float> iteration_count_in{0.f, 10.f};  // start at 0 for first item in batch, and 10 for second

  // batch_size, max_sequence_len, input_size
  std::vector<float> input_0{1.f, 2.f,
                             4.f, 3.f,

                             -1.f, -2.f,
                             -4.f, -3.f};

  std::vector<float> input_1{3.f, 4.f,
                             2.f, 1.f,

                             -3.f, -4.f,
                             -2.f, -1.f};

  std::vector<float> iteration_count_out{2.f, 12.f};  // iteration_count_in + 1 for each item in sequence

  // batch_size, max_sequence_len, 1
  std::vector<float> output_0{1.f, 4.f, -1.f, -4.f};
  std::vector<float> output_1{2.f, 3.f, -2.f, -3.f};
  std::vector<float> output_2{3.f, 2.f, -3.f, -2.f};
  std::vector<float> output_3{4.f, 1.f, -4.f, -1.f};

  RunTest("ShortSequenceTwoInBatchOneLoopStateVar", batch_size, sequence_len, input_size,
          nullptr, nullptr,
          iteration_count_in, input_0, input_1,
          iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan, MixedSequenceLens) {
  const int64_t batch_size = 2;
  const int64_t max_sequence_len = 2;
  const int64_t input_size = 2;

  std::vector<int64_t> sequence_lens{1, 2};

  std::vector<float> iteration_count_in{0.f, 10.f};  // start at 0 for first item in batch, and 10 for second

  // batch_size, max_sequence_len, input_size
  std::vector<float> input_0{1.f, 2.f,
                             4.f, 3.f,  // <- this should be ignored

                             -1.f, -2.f,
                             -4.f, -3.f};

  std::vector<float> input_1{3.f, 4.f,
                             2.f, 1.f,  // <- this should be ignored

                             -3.f, -4.f,
                             -2.f, -1.f};

  // iteration_count_in + 1 for each item in sequence.
  // as sequence_len is 1 for the first item in the batch, the final value should be 0 + 1.
  // as sequence_len is 2 for the second item in the batch, the final value should be 10 + 1 + 1.
  std::vector<float> iteration_count_out{1.f, 12.f};

  // batch_size, max_sequence_len, 1
  // as sequence_len is 1 for the first item in the batch we expect 0.f's for the second value in the output
  // (which technically is undefined, but 0.f is consistent with other RNN ops)
  std::vector<float> output_0{1.f, 0.f, -1.f, -4.f};
  std::vector<float> output_1{2.f, 0.f, -2.f, -3.f};
  std::vector<float> output_2{3.f, 0.f, -3.f, -2.f};
  std::vector<float> output_3{4.f, 0.f, -4.f, -1.f};

  RunTest("MixedSequenceLens", batch_size, max_sequence_len, input_size,
          nullptr, &sequence_lens,
          iteration_count_in, input_0, input_1,
          iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan, ShortSequenceTwoInBatchOneLoopStateVarReverseFirstInput) {
  const int64_t batch_size = 2;
  const int64_t sequence_len = 2;
  const int64_t input_size = 2;

  std::vector<float> iteration_count_in{0.f, 10.f};  // start at 0 for first item in batch, and 10 for second

  std::vector<int64_t> directions{1, 0};  // reverse for input_0, forward for input_1

  // batch_size, max_sequence_len, input_size
  std::vector<float> input_0{1.f, 2.f,
                             4.f, 3.f,

                             -1.f, -2.f,
                             -4.f, -3.f};

  std::vector<float> input_1{3.f, 4.f,
                             2.f, 1.f,

                             -3.f, -4.f,
                             -2.f, -1.f};

  std::vector<float> iteration_count_out{2.f, 12.f};  // iteration_count_in + 1 for each item in sequence

  // batch_size, max_sequence_len, 1
  // the sequence of input0 is reversed, so the subgraph will get {4.f, 3.f} then {1.f, 2.f} for batch 0
  // and {-4.f, -3.f} then {-1.f, -2.f} for batch 0.
  std::vector<float> output_0{4.f, 1.f, -4.f, -1.f};
  std::vector<float> output_1{3.f, 2.f, -3.f, -2.f};
  std::vector<float> output_2{3.f, 2.f, -3.f, -2.f};
  std::vector<float> output_3{4.f, 1.f, -4.f, -1.f};

  RunTest("ShortSequenceTwoInBatchOneLoopStateVarReverseFirstInput", batch_size, sequence_len, input_size,
          &directions, nullptr,
          iteration_count_in, input_0, input_1,
          iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan, InvalidInput) {
  const int64_t batch_size = 1;
  const int64_t sequence_len = 1;
  const int64_t input_size = 2;

  std::vector<float> iteration_count_in{0.f};

  // batch_size, max_sequence_len, input_size
  std::vector<float> input_0{1.f, 2.f};
  std::vector<float> input_1{3.f, 4.f};

  std::vector<float> iteration_count_out{1.f};

  // batch_size, max_sequence_len, 1
  std::vector<float> output_0{1.f};
  std::vector<float> output_1{2.f};
  std::vector<float> output_2{3.f};
  std::vector<float> output_3{4.f};

  // invalid direction value - only 0 or 1 are valid
  std::vector<int64_t> directions = {2, 1};

  RunTest("InvalidDirectionsValue", batch_size, sequence_len, input_size,
          &directions, nullptr,
          iteration_count_in, input_0, input_1,
          iteration_count_out, output_0, output_1, output_2, output_3,
          {},
          OpTester::ExpectResult::kExpectFailure,
          "Invalid values in 'directions'.");

  // mismatch between direction entries and num inputs
  directions = {1, 0, 1};  // too many entries - should match the number of inputs (2)

  RunTest("InvalidNumEntriesInDirections", batch_size, sequence_len, input_size,
          &directions, nullptr,
          iteration_count_in, input_0, input_1,
          iteration_count_out, output_0, output_1, output_2, output_3,
          {},
          OpTester::ExpectResult::kExpectFailure,
          "Number of entries in 'directions' was 3. Must match 'num_scan_inputs' of 2");
}

}  // namespace test
}  // namespace onnxruntime
