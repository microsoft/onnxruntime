// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "core/framework/session_state.h"
#include "core/session/inference_session.h"
#include "core/providers/common.h"
#include "core/providers/cpu/controlflow/scan_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

struct RunOptions {
  bool is_v8 = true;
  bool include_dim_values_in_main_graph = false;
  bool include_dim_values_in_subgraph = true;
  bool include_types_in_subgraph = true;
  bool include_outer_scope_add = false;
  bool scalar_loop_state_value = false;
  bool add_bad_shape = false;
  bool mixed_execution_providers = false;
  // Disable TensorRT because its parser fails, and it can't handle unknown dimensions
  std::unordered_set<std::string> excluded_provider_types{kTensorrtExecutionProvider, kOpenVINOExecutionProvider};
};

static common::Status CreateSubgraph(Graph& graph, RunOptions& options, const std::string& failure_message = "");

static constexpr float kOuterNodeAddValue = 42.f;

class ScanOpTester : public OpTester {
 public:
  ScanOpTester(int opset_version = 8) : OpTester("Scan", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
    // add outer_scope_0 node. push the value through an extra Identity node as a Constant gets lifted into an
    // initializer which results in different treatment by the allocation planner
    {
      TypeProto float_single_value;
      float_single_value.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
      auto mutable_dim = float_single_value.mutable_tensor_type()->mutable_shape()->add_dim();
      mutable_dim->set_dim_value(1);

      {
        auto& outer_scope_constant = graph.GetOrCreateNodeArg("outer_scope_constant", &float_single_value);
        auto& constant = graph.AddNode("outer_scope_constant", "Constant", "Constant with value kOuterNodeAddValue",
                                       {}, {&outer_scope_constant});

        TensorProto value_tensor;
        value_tensor.add_dims(1);
        value_tensor.add_float_data(kOuterNodeAddValue);
        value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

        constant.AddAttribute("value", value_tensor);

        auto& outer_scope_node_arg = graph.GetOrCreateNodeArg("outer_scope_0", &float_single_value);
        graph.AddNode("outer_scope_id", "Identity", "Identity for outer_scope_0",
                      {&outer_scope_constant}, {&outer_scope_node_arg});
      }
    }

    // call base implementation to add the Scan node as per usual
    OpTester::AddNodes(graph, graph_input_defs, graph_output_defs, add_attribute_funcs);
  }
};

static common::Status CreateSubgraph(Graph& graph, RunOptions& options, const std::string& failure_message) {
  bool include_dim_values = options.include_dim_values_in_subgraph;
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

    TypeProto float_input;
    // inputs must have type information and a rank
    float_input.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    auto mutable_shape = float_input.mutable_tensor_type()->mutable_shape();
    if (options.scalar_loop_state_value) {
      // no dims
    } else {
      auto mutable_dim = mutable_shape->add_dim();  // set rank
      if (include_dim_values)
        mutable_dim->set_dim_value(1);
    }

    {
      auto& output_arg = graph.GetOrCreateNodeArg("constant_1", &float_input);
      outputs.push_back(&output_arg);

      auto& constant = graph.AddNode("constant", "Constant", "Constant with value 1", inputs, outputs);

      TensorProto value_tensor;
      if (!options.scalar_loop_state_value)
        value_tensor.add_dims(1);
      value_tensor.add_float_data(1.f);
      value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      constant.AddAttribute("value", value_tensor);
    }

    inputs = outputs;  // start with output from Constant node
    outputs = {};

    auto& input_arg = graph.GetOrCreateNodeArg("loop_state_in_1", &float_input);
    inputs.push_back(&input_arg);

    TypeProto loop_state_output_tensor;
    loop_state_output_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

    // as this is a subgraph output we need a shape to come from somewhere, so if the main graph isn't providing it,
    // it has to come from here.
    bool type_and_shape_required = options.include_dim_values_in_main_graph == false;

    if (include_dim_values || type_and_shape_required) {
      mutable_shape = loop_state_output_tensor.mutable_tensor_type()->mutable_shape();
      if (!options.scalar_loop_state_value)
        mutable_shape->add_dim()->set_dim_value(1);
    }

    TypeProto* type_proto = include_types || type_and_shape_required ? &loop_state_output_tensor : nullptr;
    auto& output_arg = graph.GetOrCreateNodeArg("loop_state_out_1", type_proto);
    outputs.push_back(&output_arg);

    graph.AddNode("add", "Add", "Add 1 to the loop state", inputs, outputs);
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
    if (include_dim_values) {
      mutable_dim->set_dim_value(2);

      if (options.add_bad_shape) {
        concat_input_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(99);
      }
    }

    for (int i = 0, num_inputs = 2; i < num_inputs; ++i) {
      auto& input_arg = graph.GetOrCreateNodeArg("concat_in_" + std::to_string(i), &concat_input_tensor);
      inputs.push_back(&input_arg);
    }

    // one output from concatenate of {4} tensor
    TypeProto concat_output_tensor;
    concat_output_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    if (include_dim_values)
      concat_output_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);

    TypeProto* type_proto = include_types ? &concat_output_tensor : nullptr;
    auto& output_arg = graph.GetOrCreateNodeArg("concat_out_1", type_proto);
    outputs.push_back(&output_arg);

    auto& concat = graph.AddNode("concat", "Concat", "concat 2 inputs", inputs, outputs);

    concat.AddAttribute("axis", int64_t{0});
  }

  // Post-Concat Add Node
  if (options.include_outer_scope_add) {
    TypeProto float_scalar;
    float_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

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

    auto& split = graph.AddNode("split", "Split", "split into 4 outputs", inputs, outputs);
    split.AddAttribute("axis", int64_t{0});
    split.AddAttribute("split", std::vector<int64_t>{1, 1, 1, 1});
  }

  auto status = graph.Resolve();

  if (failure_message.empty()) {
    EXPECT_EQ(status, Status::OK());
  } else {
    EXPECT_TRUE(!status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(failure_message));
  }

  return status;
}

static void RunTest_v8(const std::string test_name, int64_t batch_size, int64_t max_sequence_len, int64_t input_size,
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
  Model model(test_name, false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 8}},
              {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto status = CreateSubgraph(graph, options, options.add_bad_shape ? failure_message : "");
  ASSERT_STATUS_OK(status);
  auto& proto = graph.ToGraphProto();

  ScanOpTester test{8};

  test.AddAttribute("body", proto);
  test.AddAttribute<int64_t>("num_scan_inputs", 2);

  if (directions != nullptr) {
    test.AddAttribute<std::vector<int64_t>>("directions", *directions);
  }

  if (sequence_lens == nullptr) {
    test.AddOptionalInputEdge<int64_t>();
  } else {
    std::vector<int64_t> sequence_lens_dims{batch_size};
    test.AddInput<int64_t>("sequence_lens", sequence_lens_dims, *sequence_lens);
  }

  test.AddShapeToTensorData(options.include_dim_values_in_main_graph);

  std::vector<int64_t> loop_state_shape{batch_size};
  if (!options.scalar_loop_state_value) {
    loop_state_shape.push_back(1);
  }

  test.AddInput<float>("scan_loop_state_in_0", loop_state_shape, loop_state_in_0);

  std::vector<int64_t> input_shape{batch_size, max_sequence_len, input_size};
  test.AddInput<float>("scan_input_0", input_shape, input_0);
  test.AddInput<float>("scan_input_1", input_shape, input_1);

  test.AddOutput<float>("scan_loop_state_out_0", loop_state_shape, loop_state_out_0);

  std::vector<int64_t> output_shape{batch_size, max_sequence_len, 1};
  test.AddOutput<float>("scan_output_0", output_shape, output_0);
  test.AddOutput<float>("scan_output_1", output_shape, output_1);
  test.AddOutput<float>("scan_output_2", output_shape, output_2);
  test.AddOutput<float>("scan_output_3", output_shape, output_3);

  test.Run(expect_result, failure_message, options.excluded_provider_types);
}

static void RunTest_v9(const std::string test_name, int64_t sequence_len, int64_t input_size,
                       std::vector<int64_t>* input_directions,
                       std::vector<int64_t>* output_directions,
                       std::vector<int64_t>* input_axes,
                       std::vector<int64_t>* output_axes,
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
  Model model(test_name, false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 11}},
              {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto status = CreateSubgraph(graph, options, options.add_bad_shape ? failure_message : "");
  if (!status.IsOK()) {
    return;
  }
  auto& proto = graph.ToGraphProto();

  ScanOpTester test{(options.add_bad_shape) ? -1 : 11};  // use latest version - no significant change over 9

  test.AddAttribute("body", proto);
  test.AddAttribute<int64_t>("num_scan_inputs", 2);

  if (input_directions != nullptr) {
    test.AddAttribute<std::vector<int64_t>>("scan_input_directions", *input_directions);
  }

  if (output_directions != nullptr) {
    test.AddAttribute<std::vector<int64_t>>("scan_output_directions", *output_directions);
  }

  if (input_axes != nullptr) {
    test.AddAttribute<std::vector<int64_t>>("scan_input_axes", *input_axes);
  }

  if (output_axes != nullptr) {
    test.AddAttribute<std::vector<int64_t>>("scan_output_axes", *output_axes);
  }

  test.AddShapeToTensorData(options.include_dim_values_in_main_graph);

  std::vector<int64_t> loop_state_shape;
  if (!options.scalar_loop_state_value) {
    loop_state_shape.push_back(1);
  }

  test.AddInput<float>("scan_loop_state_in_0", loop_state_shape, loop_state_in_0);

  std::vector<int64_t> input_shape{sequence_len, input_size};
  test.AddInput<float>("scan_input_0", input_shape, input_0);
  test.AddInput<float>("scan_input_1", input_shape, input_1);

  test.AddOutput<float>("scan_loop_state_out_0", loop_state_shape, loop_state_out_0);

  TensorShape output_shape{sequence_len, 1};

  auto calculate_output_shape = [&](size_t output_index) {
    if (output_axes && output_axes->size() > output_index) {
      const auto axis = output_axes->at(output_index);
      const auto rank = gsl::narrow_cast<int64_t>(output_shape.NumDimensions());

      // skip if this is an invalid input test and axis is out of the valid range
      if (axis >= -rank && axis < rank) {
        InlinedVector<size_t> permutations;
        TensorShapeVector new_shape;
        scan::detail::CalculateTransposedShapeForOutput(output_shape, HandleNegativeAxis(axis, rank),
                                                        permutations, new_shape);
        return std::vector<int64_t>(new_shape.cbegin(), new_shape.cend());
      }
    }

    const auto output_dims = output_shape.GetDims();
    return std::vector<int64_t>(output_dims.begin(), output_dims.end());
  };

  test.AddOutput<float>("scan_output_0", calculate_output_shape(0), output_0);
  test.AddOutput<float>("scan_output_1", calculate_output_shape(1), output_1);
  test.AddOutput<float>("scan_output_2", calculate_output_shape(2), output_2);
  test.AddOutput<float>("scan_output_3", calculate_output_shape(3), output_3);

  if (options.mixed_execution_providers) {
    // we want the CUDA provider to be first, and the CPU provider second. all except the Scan node should run on
    // CUDA given that, which creates the scenario where we need to copy to/from CPU to execute the Scan node correctly.
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    execution_providers.push_back(DefaultCpuExecutionProvider());

    test.Run(expect_result, failure_message, options.excluded_provider_types, nullptr, &execution_providers);
  } else {
    test.Run(expect_result, failure_message, options.excluded_provider_types);
  }
}

static void ShortSequenceOneInBatchOneLoopStateVar(const RunOptions& options, const std::string& expected_error = "") {
  constexpr int64_t batch_size = 1;
  constexpr int64_t sequence_len = 2;
  constexpr int64_t input_size = 2;

  std::vector<float> iteration_count_in{0.f};

  // v8: batch_size, max_sequence_len, input_size
  // v9: sequence_len, input_size
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

  if (options.is_v8) {
    RunTest_v8("ShortSequenceOneInBatchOneLoopStateVar", batch_size, sequence_len, input_size,
               nullptr, nullptr,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               options,
               expected_error.empty() ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
               expected_error);
  } else {
    RunTest_v9("ShortSequenceOneInBatchOneLoopStateVar", input_size, sequence_len,
               nullptr, nullptr, nullptr, nullptr,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               options,
               expected_error.empty() ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
               expected_error);
  }
}

#define TEST_8_AND_9(function) \
  TEST(Scan8, function) {      \
    function(true);            \
  }                            \
                               \
  TEST(Scan9, function) {      \
    function(false);           \
  }

static void ShortSequenceOneInBatchOneLoopStateVar_NoShapeInMainGraph_TypeAndShapeInSubgraph(bool is_v8) {
  RunOptions options{};
  options.is_v8 = is_v8;
  options.include_dim_values_in_main_graph = false;
  options.include_types_in_subgraph = true;
  options.include_dim_values_in_subgraph = true;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST_8_AND_9(ShortSequenceOneInBatchOneLoopStateVar_NoShapeInMainGraph_TypeAndShapeInSubgraph);

static void ShortSequenceOneInBatchOneLoopStateVar_ShapeInMainGraph_NoTypeAndShapeInSubgraph(bool is_v8) {
  RunOptions options{};
  options.is_v8 = is_v8;
  options.include_dim_values_in_main_graph = true;
  options.include_types_in_subgraph = false;
  options.include_dim_values_in_subgraph = false;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST_8_AND_9(ShortSequenceOneInBatchOneLoopStateVar_ShapeInMainGraph_NoTypeAndShapeInSubgraph);

static void ShortSequenceOneInBatchOneLoopStateVar_NoShapeInMainGraph_NoTypeAndShapeInSubgraph(bool is_v8) {
  RunOptions options{};
  options.is_v8 = is_v8;
  options.include_dim_values_in_main_graph = false;
  options.include_types_in_subgraph = false;
  options.include_dim_values_in_subgraph = false;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST_8_AND_9(ShortSequenceOneInBatchOneLoopStateVar_NoShapeInMainGraph_NoTypeAndShapeInSubgraph);

static void OnnxScalarLoopState(bool is_v8) {
  RunOptions options{};
  options.is_v8 = is_v8;
  options.include_dim_values_in_main_graph = true;
  options.include_types_in_subgraph = false;
  options.include_dim_values_in_subgraph = false;
  options.scalar_loop_state_value = true;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST_8_AND_9(OnnxScalarLoopState);

// test when there is an operator in the subgraph that uses a value coming from outer scope
static void OuterScopeAccess_NoShapeInMainGraph_TypeAndShapeInSubgraph(bool is_v8) {
  RunOptions options{};
  options.is_v8 = is_v8;
  options.include_dim_values_in_main_graph = false;
  options.include_types_in_subgraph = true;
  options.include_dim_values_in_subgraph = true;

  options.include_outer_scope_add = true;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST_8_AND_9(OuterScopeAccess_NoShapeInMainGraph_TypeAndShapeInSubgraph);

static void OuterScopeAccess_ShapeInMainGraph_NoTypeAndShapeInSubgraph(bool is_v8) {
  RunOptions options{};
  options.is_v8 = is_v8;
  options.include_dim_values_in_main_graph = true;
  options.include_types_in_subgraph = false;
  options.include_dim_values_in_subgraph = false;

  options.include_outer_scope_add = true;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST_8_AND_9(OuterScopeAccess_ShapeInMainGraph_NoTypeAndShapeInSubgraph);

static void OuterScopeAccess_NoShapeInMainGraph_NoTypeAndShapeInSubgraph(bool is_v8) {
  RunOptions options{};
  options.is_v8 = is_v8;
  options.include_dim_values_in_main_graph = false;
  options.include_types_in_subgraph = false;
  options.include_dim_values_in_subgraph = false;

  options.include_outer_scope_add = true;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST_8_AND_9(OuterScopeAccess_NoShapeInMainGraph_NoTypeAndShapeInSubgraph);

// shape inferencing is only strict for the latest version so only test BadShape with that
// Scan test uses Split operator in the subgraph. It was updated for opset13
// Enable this test once Split for op13 is implemented.
TEST(Scan9, DISABLED_BadShape) {
  RunOptions options{};
  options.is_v8 = false;
  options.include_dim_values_in_main_graph = false;
  options.include_types_in_subgraph = true;
  options.include_dim_values_in_subgraph = true;
  options.add_bad_shape = true;

  ShortSequenceOneInBatchOneLoopStateVar(
      options,
      "Node:concat Output:concat_out_1 [ShapeInferenceError] Mismatch between number of source and target dimensions. "
      "Source=2 Target=1");
}

TEST(Scan8, ShortSequenceTwoInBatchOneLoopStateVar) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t sequence_len = 2;
  constexpr int64_t input_size = 2;

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

  RunTest_v8("ShortSequenceTwoInBatchOneLoopStateVar", batch_size, sequence_len, input_size,
             nullptr, nullptr,
             iteration_count_in, input_0, input_1,
             iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan8, MixedSequenceLens) {
  constexpr int64_t batch_size = 3;
  constexpr int64_t max_sequence_len = 2;
  constexpr int64_t input_size = 2;

  std::vector<int64_t> sequence_lens{1, 2, 2};

  std::vector<float> iteration_count_in{0.f, 10.f, 1.f};  // start at 0 for first item in batch, and 10 for second

  // batch_size, max_sequence_len, input_size
  std::vector<float> input_0{1.f, 2.f,
                             4.f, 3.f,  // <- this should be ignored

                             -1.f, -2.f,
                             -4.f, -3.f,

                             10.f, 11.f,
                             12.f, 13.f};

  std::vector<float> input_1{3.f, 4.f,
                             2.f, 1.f,  // <- this should be ignored

                             -3.f, -4.f,
                             -2.f, -1.f,

                             22.f, 33.f,
                             44.f, 55.f};

  // iteration_count_in + 1 for each item in sequence.
  // as sequence_len is 1 for the first item in the batch, the final value should be 0 + 1.
  // as sequence_len is 2 for the second item in the batch, the final value should be 10 + 1 + 1.
  std::vector<float> iteration_count_out{1.f, 12.f, 3.f};

  // batch_size, max_sequence_len, 1
  // as sequence_len is 1 for the first item in the batch we expect 0.f's for the second value in the output
  // (which technically is undefined, but 0.f is consistent with other RNN ops)
  std::vector<float> output_0{1.f, 0.f, -1.f, -4.f, 10.f, 12.f};
  std::vector<float> output_1{2.f, 0.f, -2.f, -3.f, 11.f, 13.f};
  std::vector<float> output_2{3.f, 0.f, -3.f, -2.f, 22.f, 44.f};
  std::vector<float> output_3{4.f, 0.f, -4.f, -1.f, 33.f, 55.f};

  RunTest_v8("MixedSequenceLens", batch_size, max_sequence_len, input_size,
             nullptr, &sequence_lens,
             iteration_count_in, input_0, input_1,
             iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan8, MixedSequenceLensReverse) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t max_sequence_len = 2;
  constexpr int64_t input_size = 2;

  std::vector<int64_t> sequence_lens{1, 2};
  std::vector<int64_t> directions{1, 1};  // reverse both inputs

  std::vector<float> iteration_count_in{0.f, 10.f};  // start at 0 for first item in batch, and 10 for second

  // batch_size, max_sequence_len, input_size
  std::vector<float> input_0{1.f, 2.f,
                             400.f, 300.f,  // <- this should be ignored

                             -1.f, -2.f,
                             -4.f, -3.f};

  std::vector<float> input_1{3.f, 4.f,
                             200.f, 100.f,  // <- this should be ignored

                             -3.f, -4.f,
                             -2.f, -1.f};

  // iteration_count_in + 1 for each item in sequence.
  // as sequence_len is 1 for the first item in the batch, the final value should be 0 + 1.
  // as sequence_len is 2 for the second item in the batch, the final value should be 10 + 1 + 1.
  std::vector<float> iteration_count_out{1.f, 12.f};

  // batch_size, max_sequence_len, 1
  // as sequence_len is 1 for the first item in the batch we expect 0.f's for the second value in the output
  // (which technically is undefined, but 0.f is consistent with other RNN ops)
  // as the first sequence only contains one entry, the output should actually be the same as if the direction
  // was forward.
  std::vector<float> output_0{1.f, 0.f, -4.f, -1.f};
  std::vector<float> output_1{2.f, 0.f, -3.f, -2.f};
  std::vector<float> output_2{3.f, 0.f, -2.f, -3.f};
  std::vector<float> output_3{4.f, 0.f, -1.f, -4.f};

  RunTest_v8("MixedSequenceLensReverse", batch_size, max_sequence_len, input_size,
             &directions, &sequence_lens,
             iteration_count_in, input_0, input_1,
             iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan8, ShortSequenceTwoInBatchOneLoopStateVarReverseFirstInput) {
  constexpr int64_t batch_size = 2;
  constexpr int64_t sequence_len = 2;
  constexpr int64_t input_size = 2;

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

  RunTest_v8("ShortSequenceTwoInBatchOneLoopStateVarReverseFirstInput", batch_size, sequence_len, input_size,
             &directions, nullptr,
             iteration_count_in, input_0, input_1,
             iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan9, ReversedInput) {
  constexpr int64_t sequence_len = 2;
  constexpr int64_t input_size = 2;

  std::vector<float> iteration_count_in{0.f};

  std::vector<int64_t> input_directions{0, 1};  // reverse second input

  // max_sequence_len, input_size
  std::vector<float> input_0{1.f, 2.f,
                             3.f, 4.f};

  std::vector<float> input_1{11.f, 12.f,
                             13.f, 14.f};

  std::vector<float> iteration_count_out{2.f};  // iteration_count_in + 1 for each item in sequence

  // max_sequence_len, 1
  std::vector<float> output_0{1.f, 3.f};
  std::vector<float> output_1{2.f, 4.f};
  std::vector<float> output_2{13.f, 11.f};
  std::vector<float> output_3{14.f, 12.f};

  RunTest_v9("ReversedInput", sequence_len, input_size,
             &input_directions, nullptr, nullptr, nullptr,
             iteration_count_in, input_0, input_1,
             iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan9, ReversedOutput) {
  constexpr int64_t sequence_len = 2;
  constexpr int64_t input_size = 2;

  std::vector<float> iteration_count_in{0.f};

  std::vector<int64_t> output_directions{0, 1, 1, 0};  // reverse 2 out of the 4 outputs

  // max_sequence_len, input_size
  std::vector<float> input_0{1.f, 2.f,
                             3.f, 4.f};

  std::vector<float> input_1{11.f, 12.f,
                             13.f, 14.f};

  std::vector<float> iteration_count_out{2.f};  // iteration_count_in + 1 for each item in sequence

  // max_sequence_len, 1
  std::vector<float> output_0{1.f, 3.f};
  std::vector<float> output_1{4.f, 2.f};
  std::vector<float> output_2{13.f, 11.f};
  std::vector<float> output_3{12.f, 14.f};

  RunTest_v9("ReversedOutput", sequence_len, input_size,
             nullptr, &output_directions, nullptr, nullptr,
             iteration_count_in, input_0, input_1,
             iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan9, TransposeInput) {
  constexpr int64_t sequence_len = 2;
  constexpr int64_t input_size = 2;

  std::vector<float> iteration_count_in{0.f};

  // transpose should also support negative axis
  std::vector<int64_t> input_axes{1, -1};  // transpose both inputs on axis 1

  // inputs are {input_size, sequence_len}, but will be transposed to {sequence_len, input_size} by the axes values
  std::vector<float> input_0{1.f, 3.f,
                             2.f, 4.f};
  std::vector<float> input_1{11.f, 13.f,
                             12.f, 14.f};

  std::vector<float> iteration_count_out{2.f};  // iteration_count_in + 1 for each item in sequence

  // max_sequence_len, 1
  // concatenated transposed input should yield 1, 2, 11, 12 for the first item in the sequence, and 3, 4, 12, 14
  // for the second
  std::vector<float> output_0{1.f, 3.f};
  std::vector<float> output_1{2.f, 4.f};
  std::vector<float> output_2{11.f, 13.f};
  std::vector<float> output_3{12.f, 14.f};

  RunTest_v9("TransposeInput", sequence_len, input_size,
             nullptr, nullptr, &input_axes, nullptr,
             iteration_count_in, input_0, input_1,
             iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan9, TransposeOutput) {
  constexpr int64_t sequence_len = 2;
  constexpr int64_t input_size = 2;

  std::vector<float> iteration_count_in{0.f};

  // transpose also supports negative axis
  std::vector<int64_t> output_axes{1, -1, 0, 0};  // transpose two outputs on axis 1, and leave 2 as is by using axis 0

  std::vector<float> input_0{1.f, 2.f,
                             3.f, 4.f};
  std::vector<float> input_1{11.f, 12.f,
                             13.f, 14.f};

  std::vector<float> iteration_count_out{2.f};  // iteration_count_in + 1 for each item in sequence

  // whilst we transpose that only changes the shape from 2, 1 to 1, 2 so the data is the same. the expected
  // shape is validated by RunTest_v9.
  std::vector<float> output_0{1.f, 3.f};
  std::vector<float> output_1{2.f, 4.f};
  std::vector<float> output_2{11.f, 13.f};
  std::vector<float> output_3{12.f, 14.f};

  RunTest_v9("TransposeOutput", sequence_len, input_size,
             nullptr, nullptr, nullptr, &output_axes,
             iteration_count_in, input_0, input_1,
             iteration_count_out, output_0, output_1, output_2, output_3);
}

TEST(Scan9, TransposeOutputDim2) {
  // Construct scan body subgraph with 1 scan inputs, 1 scan outputs
  // scan-in-1 => scan-out-1
  Model model("ScanBody", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& scan_in_1 = graph.GetOrCreateNodeArg("scan_in_1", &float_tensor);
  auto& scan_out_1 = graph.GetOrCreateNodeArg("scan_out_1", &float_tensor);

  graph.AddNode("pass_through", "Identity", "Copy scan_in_1 to scan_out_1", {&scan_in_1}, {&scan_out_1});

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto& scan_body = graph.ToGraphProto();

  ScanOpTester test{9};

  std::vector<int64_t> input_shape{2, 1, 1};

  // transpose on axis 2, so dim 0 of the output (copied directly from input of {2, 1, 1})
  // will move to dim 2 of the output giving shape {1, 1, 2}
  std::vector<int64_t> output_axes{2};
  std::vector<int64_t> output_shape{1, 1, 2};

  test.AddAttribute("body", scan_body);
  test.AddAttribute<int64_t>("num_scan_inputs", 1);
  test.AddAttribute<std::vector<int64_t>>("scan_output_axes", output_axes);

  // the data won't change, but the shape should be transposed from 2, 1, 1 to 1, 1, 2, which
  // OpTester::Run will validate
  test.AddInput<float>("scan_input_1", input_shape, {1.0, 2.0});
  test.AddOutput<float>("scan_output_1", output_shape, {1.0, 2.0});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", RunOptions().excluded_provider_types);
}

static void InvalidInput(bool is_v8) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t sequence_len = 2;
  constexpr int64_t input_size = 2;

  std::vector<float> iteration_count_in{0.f};

  // [batch_size,] max_sequence_len, input_size
  std::vector<float> input_0{1.f, 2.f, 3.f, 4.f};
  std::vector<float> input_1{-1.f, -2.f, -3.f, -4.f};

  std::vector<float> iteration_count_out{1.f};

  // [batch_size,] max_sequence_len, 1
  std::vector<float> output_0{0.f, 0.f};
  std::vector<float> output_1{0.f, 0.f};
  std::vector<float> output_2{0.f, 0.f};
  std::vector<float> output_3{0.f, 0.f};

  // invalid direction value - only 0 or 1 are valid
  std::vector<int64_t> directions = {2, 1};

  if (is_v8) {
    RunTest_v8("InvalidDirectionsValue", batch_size, sequence_len, input_size,
               &directions, nullptr,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               {},
               OpTester::ExpectResult::kExpectFailure,
               "Invalid values in 'directions'.");
  } else {
    RunTest_v9("InvalidInputDirectionsValue", sequence_len, input_size,
               &directions, nullptr, nullptr, nullptr,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               {},
               OpTester::ExpectResult::kExpectFailure,
               "Invalid values in 'scan_input_directions'.");

    std::vector<int64_t> output_directions = {0, 2, 1, 0};

    RunTest_v9("InvalidOutputDirectionsValue", sequence_len, input_size,
               nullptr, &output_directions, nullptr, nullptr,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               {},
               OpTester::ExpectResult::kExpectFailure,
               "Invalid values in 'scan_output_directions'.");
  }

  // mismatch between direction entries and num inputs/outputs
  directions = {1, 0, 1};  // too many entries for the 2 inputs, too few for the 4 outputs

  if (is_v8) {
    RunTest_v8("InvalidNumEntriesInDirections", batch_size, sequence_len, input_size,
               &directions, nullptr,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               {},
               OpTester::ExpectResult::kExpectFailure,
               "Number of entries in 'directions' was 3 but expected 2");
  } else {
    RunTest_v9("InvalidNumEntriesInInputDirections", sequence_len, input_size,
               &directions, nullptr, nullptr, nullptr,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               {},
               OpTester::ExpectResult::kExpectFailure,
               "Number of entries in 'scan_input_directions' was 3 but expected 2");

    RunTest_v9("InvalidNumEntriesInOutputDirections", sequence_len, input_size,
               nullptr, &directions, nullptr, nullptr,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               {},
               OpTester::ExpectResult::kExpectFailure,
               "Number of entries in 'scan_output_directions' was 3 but expected 4");
  }

  if (!is_v8) {
    std::vector<int64_t> input_axes = {2, -1};  // only 2 dims in input so 2 is invalid
    RunTest_v9("InvalidEntryInInputAxes", sequence_len, input_size,
               nullptr, nullptr, &input_axes, nullptr,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               {},
               OpTester::ExpectResult::kExpectFailure,
               "Invalid value in scan_input_axes for input 0 of 2. Input tensor rank was 2");

    input_axes = {0, 1, 2};
    RunTest_v9("InvalidNumEntriesInInputAxes", sequence_len, input_size,
               nullptr, nullptr, &input_axes, nullptr,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               {},
               OpTester::ExpectResult::kExpectFailure,
               "[ShapeInferenceError] Number of scan input axes specified (3) is not equal to number of scan inputs (2).");

    std::vector<int64_t> output_axes = {3, -1, 0, 0};  // 2 dims in output so 3 is invalid
    RunTest_v9("InvalidEntryInOutputAxes", sequence_len, input_size,
               nullptr, nullptr, nullptr, &output_axes,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               {},
               OpTester::ExpectResult::kExpectFailure,
               "[ShapeInferenceError] scan_output_axes axis value 3 is invalid for a tensor of rank 2");

    output_axes = {0, 1, 2};
    RunTest_v9("InvalidNumEntriesInOutputAxes", sequence_len, input_size,
               nullptr, nullptr, nullptr, &output_axes,
               iteration_count_in, input_0, input_1,
               iteration_count_out, output_0, output_1, output_2, output_3,
               {},
               OpTester::ExpectResult::kExpectFailure,
               "[ShapeInferenceError] Number of scan output axes specified (3) is not equal to number of scan outputs (4).");
  }
}

TEST_8_AND_9(InvalidInput);

// Test usage of multiple inputs of different types for variadic inputs
void MixedTypeInputs(bool is_v8) {
  // Construct scan body subgraph with 2 state variables, 2 scan inputs, 2 scan outputs
  // of different types (1 float and 1 int64 of each):
  // state-in-1 => scan-out-1
  // scan-in-1 => state-out-1
  // state-in-2 => scan-out-2
  // scan-in-2 => state-out-2

  Model model("ScanBody", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  TypeProto int_tensor;
  int_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& state_in_1 = graph.GetOrCreateNodeArg("state_in_1", &float_tensor);
  auto& state_in_2 = graph.GetOrCreateNodeArg("state_in_2", &int_tensor);
  auto& scan_in_1 = graph.GetOrCreateNodeArg("scan_in_1", &float_tensor);
  auto& scan_in_2 = graph.GetOrCreateNodeArg("scan_in_2", &int_tensor);

  auto& state_out_1 = graph.GetOrCreateNodeArg("state_out_1", &float_tensor);
  auto& state_out_2 = graph.GetOrCreateNodeArg("state_out_2", &int_tensor);
  auto& scan_out_1 = graph.GetOrCreateNodeArg("scan_out_1", &float_tensor);
  auto& scan_out_2 = graph.GetOrCreateNodeArg("scan_out_2", &int_tensor);

  graph.AddNode("node1", "Identity", "Copy state_in_1 to scan_out_1", {&state_in_1}, {&scan_out_1});
  graph.AddNode("node2", "Identity", "Copy state_in_2 to scan_out_2", {&state_in_2}, {&scan_out_2});
  graph.AddNode("node3", "Identity", "Copy scan_in_1 to state_out_1", {&scan_in_1}, {&state_out_1});
  graph.AddNode("node4", "Identity", "Copy scan_in_2 to state_out_2", {&scan_in_2}, {&state_out_2});

  graph.SetInputs({&state_in_1, &state_in_2, &scan_in_1, &scan_in_2});
  graph.SetOutputs({&state_out_1, &state_out_2, &scan_out_1, &scan_out_2});

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto& scan_body = graph.ToGraphProto();

  // Construct and run scan test
  ScanOpTester test{is_v8 ? 8 : 9};

  int64_t batch_size = 1, sequence_len = 3, input_size = 1;
  std::vector<int64_t> seq_shape{sequence_len, input_size};
  std::vector<int64_t> state_shape{input_size};

  if (is_v8) {
    seq_shape.insert(seq_shape.begin(), batch_size);
    state_shape.insert(state_shape.begin(), batch_size);

    test.AddOptionalInputEdge<int64_t>();
  }

  test.AddAttribute("body", scan_body);
  test.AddAttribute<int64_t>("num_scan_inputs", 2);

  test.AddInput<float>("initial_state_1", state_shape, {0.0});
  test.AddInput<int64_t>("initial_state_2", state_shape, {0});
  test.AddInput<float>("scan_input_1", seq_shape, {1.0, 2.0, 3.0});
  test.AddInput<int64_t>("scan_input_2", seq_shape, {1, 2, 3});

  test.AddOutput<float>("final_state_1", state_shape, {3.0});
  test.AddOutput<int64_t>("final_state_2", state_shape, {3});
  test.AddOutput<float>("scan_output_1", seq_shape, {0.0, 1.0, 2.0});
  test.AddOutput<int64_t>("scan_output_2", seq_shape, {0, 1, 2});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", RunOptions().excluded_provider_types);
}

TEST_8_AND_9(MixedTypeInputs);

// create a subgraph that will have unknown dimensions in both the loop state variable and output
// after shape inferencing.
void UnknownDimInSubgraphOutput(bool is_v8, bool mixed_execution_providers = false) {
  Model model("ScanBody", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("param");
  TypeProto int_tensor;
  int_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("param");

  auto& state_in_1 = graph.GetOrCreateNodeArg("state_in_1", &float_tensor);
  auto& scan_in_1 = graph.GetOrCreateNodeArg("scan_in_1", &float_tensor);

  auto& state_out_1 = graph.GetOrCreateNodeArg("state_out_1", &float_tensor);
  auto& scan_out_1 = graph.GetOrCreateNodeArg("scan_out_1", &float_tensor);

  graph.AddNode("node1", "Identity", "Copy state_in_1 to scan_out_1", {&state_in_1}, {&scan_out_1});
  graph.AddNode("node2", "Identity", "Copy scan_in_1 to state_out_1", {&scan_in_1}, {&state_out_1});

  graph.SetInputs({&state_in_1, &scan_in_1});
  graph.SetOutputs({&state_out_1, &scan_out_1});

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto& scan_body = graph.ToGraphProto();

  // Construct and run scan test
  ScanOpTester test{is_v8 ? 8 : 9};

  int64_t batch_size = 1, sequence_len = 3, input_size = 1;
  std::vector<int64_t> seq_shape{sequence_len, input_size};
  std::vector<int64_t> state_shape{input_size};

  if (is_v8) {
    seq_shape.insert(seq_shape.begin(), batch_size);
    state_shape.insert(state_shape.begin(), batch_size);

    test.AddOptionalInputEdge<int64_t>();
  }

  test.AddAttribute("body", scan_body);
  test.AddAttribute<int64_t>("num_scan_inputs", 1);

  // we add a symbolic dimension to both the initial state and the scan input so we test
  // the path that handles loop state variables (OutputIterator::Initialize) and
  // the path that handles subgraph outputs (OutputIterator::MakeConcrete).
  // Note that we cross the values over in the subgraph, so the symbolic dimension in
  // initial_state_1 affects scan_out_1, and the symbolic dimension in scan_input_1 affects state_out_1.
  test.AddShapeToTensorData(true, is_v8 ? 1 : 0);  // symbolic dim in input size dim for initial_state_1
  test.AddInput<float>("initial_state_1", state_shape, {0.0});
  test.AddShapeToTensorData(true, is_v8 ? 2 : 1);  // symbolic dim in seq length dim for scan_input_1
  test.AddInput<float>("scan_input_1", seq_shape, {1.0, 2.0, 3.0});

  test.AddOutput<float>("final_state_1", state_shape, {3.0});
  test.AddOutput<float>("scan_output_1", seq_shape, {0.0, 1.0, 2.0});

  if (mixed_execution_providers) {
    // we want the CUDA provider to be first, and the CPU provider second. all except the Scan node should run on
    // CUDA given that, which creates the scenario where we need to copy to/from CPU to execute the Scan node correctly.
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    execution_providers.push_back(DefaultCpuExecutionProvider());

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", RunOptions().excluded_provider_types, nullptr,
             &execution_providers);
  } else {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", RunOptions().excluded_provider_types);
  }
}

TEST_8_AND_9(UnknownDimInSubgraphOutput);

#ifdef USE_CUDA
TEST(Scan, MixedExecutionProviders) {
  RunOptions options{};
  options.is_v8 = false;
  options.mixed_execution_providers = true;

  ShortSequenceOneInBatchOneLoopStateVar(options);
}

TEST(Scan, MixedExecutionProvidersUnknownDimInSubgraphOutput) {
  UnknownDimInSubgraphOutput(/*is_v8*/ true, /*mixed_execution_providers*/ true);
  UnknownDimInSubgraphOutput(/*is_v8*/ false, /*mixed_execution_providers*/ true);
}

#endif

}  // namespace test
}  // namespace onnxruntime
