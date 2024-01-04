// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <future>
#include <thread>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/logging/logging.h"
#include "core/framework/session_state.h"
#include "core/session/inference_session.h"

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/framework/test_utils.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

namespace {
struct RunOptions {
  bool include_dim_values_in_main_graph = true;
  bool include_dim_values_in_subgraph = false;
  bool include_types_in_subgraph = false;
  bool mixed_execution_providers = false;
  bool init_cond_1d_tensor = true;
  bool init_iter_num_1d_tensor = true;
  bool subgraph_cond_1d_tensor = true;
  bool subgraph_iter_num_1d_tensor = true;
};
}  // namespace

static const ONNX_NAMESPACE::GraphProto CreateSubgraph(const RunOptions& options);

static constexpr float kOuterNodeAddValue = 3.f;
static constexpr float kSumMax = 8.f;

class LoopOpTester : public OpTester {
 public:
  using SubgraphFunc = std::function<const ONNX_NAMESPACE::GraphProto(const RunOptions& options)>;

  LoopOpTester(const RunOptions& options, SubgraphFunc create_subgraph = CreateSubgraph)
      : OpTester("Loop", 8), options_{options}, create_subgraph_{create_subgraph} {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    // last entry in graph_input_defs is outer_scope_0
    // last entry in graph_output_defs is the output from casting outer_scope_0
    // all other inputs/outputs go to Loop
    // We do this to validate the handling of graph inputs that are used as both implicit inputs in a Loop node
    // and directly by another node.
    TypeProto float_scalar;
    float_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    auto mutable_dim = float_scalar.mutable_tensor_type()->mutable_shape()->add_dim();
    mutable_dim->set_dim_value(1);

    auto& cast_node = graph.AddNode("cast", "Cast", "Use graph input in main graph",
                                    {graph_input_defs.back()}, {graph_output_defs.back()});
    cast_node.AddAttribute("to", int64_t(TensorProto_DataType_INT64));

    // add Loop node
    std::vector<onnxruntime::NodeArg*> loop_input_defs = graph_input_defs;
    std::vector<onnxruntime::NodeArg*> loop_output_defs = graph_output_defs;
    loop_input_defs.pop_back();
    loop_output_defs.pop_back();
    auto& loop_node = graph.AddNode("loop", "Loop", "Loop node", loop_input_defs, loop_output_defs);

    auto body = create_subgraph_(options_);
    loop_node.AddAttribute("body", body);
  }

 private:
  RunOptions options_;
  SubgraphFunc create_subgraph_;
};

static const ONNX_NAMESPACE::GraphProto CreateSubgraph(const RunOptions& options) {
  bool include_dim_value = options.include_dim_values_in_subgraph;
  bool include_types = options.include_types_in_subgraph;

  // a subgraph output needs the type/shape to come from somewhere,
  // so if the main graph isn't providing it, it has to come from here.
  bool graph_output_shape_required = options.include_dim_values_in_main_graph == false;

  bool use_null_typeproto = !include_types && !include_dim_value && !graph_output_shape_required;

  bool is_cond_1d = options.subgraph_cond_1d_tensor;
  bool is_iter_num_1d = options.subgraph_iter_num_1d_tensor;

  // Loop tests use unsqueeze operator in it's subgraph. Unsqueeze was updated in opset13
  // This test can continue to use opset12 or can be updated to use the latest opset once
  // unsqueeze op13 implementation is done.
  Model model("Loop subgraph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 12}},
              {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  /* Subgraph Adds outer_scope_0 to loop_var_0_in,
     Concats the iter_num to loop_var_1_in (test loop var that changes shape) so each iteration appends the iter_num
     to loop_var_1
     Loop output is the iter_num and sum for that iteration, so each iteration adds a pair to the overall output

    Inputs: iter_num, cond_in, loop_var_in

 iter_num_in  loop_var_0_in [outer_scope_0] loop_var_1_in                                    cond_in
       |             |        /                  |                                           (unused)
     [Cast]        [Add]-----/                   |
       |             |                           |                          [Constant]
  iter_num_float  sum_0                          |          sum_0             /
       |           / | \                         |              \            /
  (if scalar)     /  |  \---------------------[Concat]           \--[Less]--/
  [Unsqueeze]    /   |                           |                     |
       |        /    |                           |                     |
    [Concat]---/ [Identity]                      |                     |
       |             |                           |                     |
   loop_out_0   loop_var_0_out             loop_var_1_out           cond_out

  */

  // graph inputs.
  TypeProto int64_scalar;
  int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int64_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto int64_tensor;
  int64_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);

  TypeProto int64_tensor_single_dim{int64_tensor};
  int64_tensor_single_dim.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);

  TypeProto bool_tensor_single_dim{bool_tensor};
  bool_tensor_single_dim.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto float_scalar;
  float_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  // dimension value changes on each iteration so just add a dimension but no value
  TypeProto float_tensor_single_dim{float_tensor};
  float_tensor_single_dim.mutable_tensor_type()->mutable_shape()->add_dim();

  // graph inputs
  auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in",
                                               is_iter_num_1d ? &int64_tensor_single_dim : &int64_scalar);
  auto& cond_in = graph.GetOrCreateNodeArg("cond_in",
                                           is_cond_1d ? &bool_tensor_single_dim : &bool_scalar);
  auto& loop_var_0_in = graph.GetOrCreateNodeArg("loop_var_0_in", &float_tensor_single_dim);
  auto& loop_var_1_in = graph.GetOrCreateNodeArg("loop_var_1_in", &float_tensor_single_dim);

  auto& iter_num_float = graph.GetOrCreateNodeArg("iter_num_float",
                                                  is_iter_num_1d ? &float_tensor_single_dim : &float_scalar);
  auto& iter_num_float_tensor = is_iter_num_1d ? iter_num_float
                                               : graph.GetOrCreateNodeArg("iter_num_float_tensor", &float_tensor_single_dim);

  // outer scope values. need type but not shape.
  auto& outer_scope_0 = graph.GetOrCreateNodeArg("outer_scope_0", &float_tensor);

  // add so that we don't end up with it being considered a graph input
  graph.AddOuterScopeNodeArg("outer_scope_0");

  // graph outputs
  NodeArg* cond_out = nullptr;
  NodeArg* loop_var_0_out = nullptr;
  NodeArg* loop_var_1_out = nullptr;
  NodeArg* loop_out_0 = nullptr;

  TypeProto sum_tensor;
  auto* mutable_tensor_type = sum_tensor.mutable_tensor_type();
  mutable_tensor_type->set_elem_type(TensorProto_DataType_FLOAT);
  if (include_dim_value) {
    mutable_tensor_type->mutable_shape()->add_dim()->set_dim_value(1);
  } /*else if (graph_output_shape_required) {
    mutable_tensor_type->mutable_shape()->add_dim();
  }*/

  TypeProto* type_proto = use_null_typeproto ? nullptr : &sum_tensor;
  auto& sum_0 = graph.GetOrCreateNodeArg("sum_0", type_proto);

  // Add
  {
    inputs = {&outer_scope_0, &loop_var_0_in};
    outputs = {&sum_0};

    graph.AddNode("add", "Add", "Add 1 to the loop carried var", inputs, outputs);
  }

  // Convert iter_num to float
  {
    auto& cast = graph.AddNode("iter_num_cast", "Cast", "Cast iter_num to float", {&iter_num_in}, {&iter_num_float});
    cast.AddAttribute("to", int64_t{TensorProto_DataType_FLOAT});
  }

  // Unsqueeze iter_num_float, if initial iter_num is scalar.
  if (!is_iter_num_1d) {
    auto& unsqueeze = graph.AddNode("iter_num_unsqueeze", "Unsqueeze",
                                    "Unsqueeze iter_num_float to tensor of single dim",
                                    {&iter_num_float}, {&iter_num_float_tensor});
    unsqueeze.AddAttribute("axes", std::vector<int64_t>{0});
  }

  // Concat iter_num and sum to create loop_out_0
  {
    TypeProto loop_out_type;
    mutable_tensor_type = loop_out_type.mutable_tensor_type();
    mutable_tensor_type->set_elem_type(TensorProto_DataType_FLOAT);

    if (include_dim_value) {
      loop_out_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    } else {
      // tensor type must have rank
      loop_out_type.mutable_tensor_type()->mutable_shape()->add_dim();
    }

    loop_out_0 = &graph.GetOrCreateNodeArg("loop_out_0", &loop_out_type);

    inputs = {&iter_num_float_tensor, &sum_0};
    outputs = {loop_out_0};

    auto& concat = graph.AddNode("concat_0", "Concat", "Combine iter num and current sum", inputs, outputs);
    concat.AddAttribute("axis", int64_t{0});
  }

  // output sum_0 as loop_var_0_out
  {
    TypeProto loop_var_0_out_type;
    mutable_tensor_type = loop_var_0_out_type.mutable_tensor_type();
    mutable_tensor_type->set_elem_type(TensorProto_DataType_FLOAT);

    if (include_dim_value) {
      loop_var_0_out_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    } else {
      loop_var_0_out_type.mutable_tensor_type()->mutable_shape()->add_dim();
    }

    loop_var_0_out = &graph.GetOrCreateNodeArg("loop_var_0_out", &loop_var_0_out_type);

    inputs = {&sum_0};
    outputs = {loop_var_0_out};

    graph.AddNode("identity", "Identity", "Output sum as loop_var_0_out", inputs, outputs);
  }

  // Concat sum with loop_var_1
  {
    TypeProto loop_var_1_out_type;
    mutable_tensor_type = loop_var_1_out_type.mutable_tensor_type();
    mutable_tensor_type->set_elem_type(TensorProto_DataType_FLOAT);
    // As we accumulate on each iteration the shape can only have a rank and not a specific value for the dimension.
    // as the type will be inferred it must have a shape
    loop_var_1_out_type.mutable_tensor_type()->mutable_shape()->add_dim();

    loop_var_1_out = &graph.GetOrCreateNodeArg("loop_var_1_out", &loop_var_1_out_type);

    inputs = {&loop_var_1_in, &sum_0};
    outputs = {loop_var_1_out};

    auto& concat = graph.AddNode("concat_1", "Concat", "Append value of sum to loop_var_1_out", inputs, outputs);
    concat.AddAttribute("axis", int64_t{0});
  }

  // Update cond by checking if sum is < kSumMax
  {
    {
      auto& max_value_out = graph.GetOrCreateNodeArg("max_value_out", &float_tensor_single_dim);
      auto& constant = graph.AddNode("constant_max_value", "Constant", "Constant with value kSumMax",
                                     {}, {&max_value_out});

      TensorProto value_tensor;
      value_tensor.add_dims(1);
      value_tensor.add_float_data(kSumMax);
      value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      constant.AddAttribute("value", value_tensor);

      cond_out = &graph.GetOrCreateNodeArg("cond_out", &bool_tensor_single_dim);

      inputs = {&sum_0, &max_value_out};
      outputs = {cond_out};

      graph.AddNode("sum_less_than_max", "Less", "Check sum < kSumMax", inputs, outputs);
    }
  }

  graph.SetInputs({&iter_num_in, &cond_in, &loop_var_0_in, &loop_var_1_in});
  graph.SetOutputs({cond_out, loop_var_0_out, loop_var_1_out, loop_out_0});

  // optional input backed by an initializer to make sure that's handled too.
  // we expect that Graph::InferAndVerifySubgraphTypes will be able to ignore the optional input if not provided
  {
    TensorProto optional_input_tensor;
    optional_input_tensor.set_name("optional_float");
    optional_input_tensor.add_dims(1);
    optional_input_tensor.add_float_data(1.f);
    optional_input_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

    graph.AddInitializedTensor(optional_input_tensor);
  }

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  return graph.ToGraphProto();
}

void RunTest(int64_t max_iterations,
             float loop_var_0_final,
             std::vector<int64_t>& loop_var_1_final_shape,
             std::vector<float>& loop_var_1_final,
             std::vector<int64_t>& loop_out_0_final_shape,
             std::vector<float>& loop_out_0_final,
             const RunOptions& options,
             OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
             const std::string& failure_message = "") {
  LoopOpTester test{options};

  test.AddShapeToTensorData(options.include_dim_values_in_main_graph);

  if (options.init_iter_num_1d_tensor) {
    test.AddInput<int64_t>("M", {1}, {max_iterations});
  } else {
    test.AddInput<int64_t>("M", {}, {max_iterations});
  }
  if (options.init_cond_1d_tensor) {
    test.AddInput<bool>("cond", {1}, {true});
  } else {
    test.AddInput<bool>("cond", {}, {true});
  }

  test.AddInput<float>("loop_var_0_orig", {1}, {0.f});
  test.AddInput<float>("loop_var_1_orig", {1}, {0.f});
  test.AddInput<float>("outer_scope_0", {1}, {kOuterNodeAddValue});

  test.AddOutput<float>("loop_var_0_final", {1}, {loop_var_0_final});
  test.AddOutput<float>("loop_var_1_final", loop_var_1_final_shape, loop_var_1_final);
  test.AddOutput<float>("loop_out_0_final", loop_out_0_final_shape, loop_out_0_final);

  test.AddOutput<int64_t>("outer_scope_0_out", {1}, {int64_t(kOuterNodeAddValue)});

  if (options.mixed_execution_providers) {
    // we want the CUDA provider to be first, and the CPU provider second. all except the Loop node should run on
    // CUDA given that, which creates the scenario where we need to copy to/from CPU to execute the Loop node correctly.
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_CUDA)
    execution_providers.push_back(DefaultCudaExecutionProvider());
#elif defined(USE_ROCM)
    execution_providers.push_back(DefaultRocmExecutionProvider());
#endif
    execution_providers.push_back(DefaultCpuExecutionProvider());

    test.Run(expect_result, failure_message, {kTensorrtExecutionProvider}, nullptr, &execution_providers);
  } else {
    test.Run(expect_result, failure_message, {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // Disable TensorRT because of unsupported data type INT64
  }
}

// exit due to hitting condition that the sum is < kSumMax which is 8
// this should take 3 iterations as we add 3 each time.
void ExitDueToCond(const RunOptions& options) {
  int64_t max_iterations = 5;
  constexpr int64_t expected_num_iterations = 3;

  float loop_var_0_final = kOuterNodeAddValue * expected_num_iterations;

  std::vector<int64_t> loop_var_1_final_shape{1 + expected_num_iterations};
  std::vector<float> loop_var_1_final{0.f, 3.f, 6.f, 9.f};

  std::vector<int64_t> loop_out_0_final_shape{expected_num_iterations, 2};
  std::vector<float> loop_out_0_final{0.f, 3.f,  // iter #, sum for each iteration
                                      1.f, 6.f,
                                      2.f, 9.f};

  RunTest(max_iterations,
          loop_var_0_final,
          loop_var_1_final_shape, loop_var_1_final,
          loop_out_0_final_shape, loop_out_0_final,
          options);
}

#define TEST_EXIT_DUE_TO_COND(name, dim_in_main_graph, iter_num_1d, cond_1d) \
  TEST(Loop, name) {                                                         \
    RunOptions options{};                                                    \
    options.include_dim_values_in_main_graph = dim_in_main_graph;            \
    options.include_dim_values_in_subgraph = !dim_in_main_graph;             \
    options.include_types_in_subgraph = false;                               \
                                                                             \
    options.init_iter_num_1d_tensor = iter_num_1d;                           \
    options.init_cond_1d_tensor = cond_1d;                                   \
    options.subgraph_iter_num_1d_tensor = iter_num_1d;                       \
    options.subgraph_cond_1d_tensor = cond_1d;                               \
                                                                             \
    ExitDueToCond(options);                                                  \
  }

TEST_EXIT_DUE_TO_COND(ExitDueToCond_DimsInMainGraph, true, true, true);
TEST_EXIT_DUE_TO_COND(ExitDueToCond_DimsInMainGraph_ScalarIter, true, false, true);
TEST_EXIT_DUE_TO_COND(ExitDueToCond_DimsInMainGraph_ScalarCond, true, true, false);
TEST_EXIT_DUE_TO_COND(ExitDueToCond_DimsInMainGraph_ScalarBoth, true, false, false);
TEST_EXIT_DUE_TO_COND(ExitDueToCond_DimsInSubGraph, false, true, true);
TEST_EXIT_DUE_TO_COND(ExitDueToCond_DimsInSubGraph_ScalarIter, false, false, true);
TEST_EXIT_DUE_TO_COND(ExitDueToCond_DimsInSubGraph_ScalarCond, false, true, false);
TEST_EXIT_DUE_TO_COND(ExitDueToCond_DimsInSubGraph_ScalarBoth, false, false, false);

// check that a rank mismatch between the Loop 'M' and 'cond' inputs and the subgraph is handled gracefully
// if both equate to a scalar (rank 0 or rank 1 with shape of {1})
TEST(Loop, LoopSubgraphRankMismatch) {
  RunOptions options{};
  options.include_dim_values_in_main_graph = false;
  options.include_dim_values_in_subgraph = false;

  options.init_iter_num_1d_tensor = true;
  options.init_cond_1d_tensor = true;
  options.subgraph_cond_1d_tensor = false;
  options.subgraph_cond_1d_tensor = false;

  ExitDueToCond(options);

  options.init_iter_num_1d_tensor = false;
  options.init_cond_1d_tensor = false;
  options.subgraph_cond_1d_tensor = true;
  options.subgraph_cond_1d_tensor = true;

  ExitDueToCond(options);
}

TEST(Loop, ExitDueToMaxIterations) {
  int64_t max_iterations = 2;
  constexpr int64_t expected_num_iterations = 2;

  float loop_var_0_final = kOuterNodeAddValue * expected_num_iterations;

  std::vector<int64_t> loop_var_1_final_shape{1 + expected_num_iterations};
  std::vector<float> loop_var_1_final{0.f, 3.f, 6.f};

  std::vector<int64_t> loop_out_0_final_shape{expected_num_iterations, 2};
  std::vector<float> loop_out_0_final{0.f, 3.f,  // iter #, sum for each iteration
                                      1.f, 6.f};

  RunTest(max_iterations,
          loop_var_0_final,
          loop_var_1_final_shape, loop_var_1_final,
          loop_out_0_final_shape, loop_out_0_final,
          {});
}

TEST(Loop, ZeroIterations) {
  int64_t max_iterations = 0;

  float loop_var_0_final = 0.f;

  std::vector<int64_t> loop_var_1_final_shape{1};
  std::vector<float> loop_var_1_final{0.f};

  // zero iterations so first dim value is 0. also checking rank is correct.
  std::vector<int64_t> loop_out_0_final_shape{0, 0};
  std::vector<float> loop_out_0_final{};

  RunTest(max_iterations,
          loop_var_0_final,
          loop_var_1_final_shape, loop_var_1_final,
          loop_out_0_final_shape, loop_out_0_final,
          {});
}

TEST(Loop, InfiniteLoopTermination) {
  auto create_subgraph = [](const RunOptions&) {
    Model model("Infinite Loop subgraph", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    /* Never change cond_in so loop is infinite
            Inputs: iter_num, cond_in, loop carried state variables.

         iter_num_in    cond_in     [outer_scope_0]
           (unused)        |                |
                       [Identity]      [Identity]
                           |               |
                        cond_out     loop_var_0_out
    */

    // graph inputs types.
    TypeProto int64_scalar;
    int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
    int64_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto bool_scalar;
    bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
    bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim();

    // graph inputs
    auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
    auto& cond_in = graph.GetOrCreateNodeArg("cond_in", &bool_scalar);

    // outer scope value. need type but not shape.
    auto& outer_scope_0 = graph.GetOrCreateNodeArg("outer_scope_0", &float_tensor);

    // add so that we don't end up with it being considered a graph input
    graph.AddOuterScopeNodeArg("outer_scope_0");

    // graph outputs
    auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
    auto& loop_var_0_out = graph.GetOrCreateNodeArg("loop_var_0_out", &float_tensor);

    // cond_in -> cond_out
    {
      inputs = {&cond_in};
      outputs = {&cond_out};

      graph.AddNode("cond_in_identity", "Identity", "Forward cond_in to cond_out", inputs, outputs);
    }

    // outer_scope_0 -> loop_var_0_out
    {
      inputs = {&outer_scope_0};
      outputs = {&loop_var_0_out};

      graph.AddNode("loop_var_out", "Identity", "Forward outer_scope_0 to loop_var_0_out", inputs, outputs);
    }

    graph.SetInputs({&iter_num_in, &cond_in, &outer_scope_0});
    graph.SetOutputs({&cond_out, &loop_var_0_out});

    auto status = graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return graph.ToGraphProto();
  };

  LoopOpTester test{{}, create_subgraph};

  test.AddInput<int64_t>("M", {1}, {INT64_MAX});
  test.AddInput<bool>("cond", {1}, {true});
  test.AddInput<float>("fake", {1}, {0.f});
  test.AddInput<float>("outer_scope_0", {1}, {kOuterNodeAddValue});

  test.AddOutput<float>("loop_var_0_final", {1}, {0.f});
  test.AddOutput<int64_t>("outer_scope_0_out", {1}, {int64_t(kOuterNodeAddValue)});

  OrtRunOptions session_run_options;
  session_run_options.run_tag = "Loop.InfiniteLoopTermination";

  auto terminator = [&session_run_options]() {
    std::this_thread::sleep_for(std::chrono::seconds(3));
    LOGS_DEFAULT(WARNING) << "Setting terminate flag in run options.";
    session_run_options.terminate = true;
    return;
  };

  std::packaged_task<void()> task{terminator};
  std::future<void> terminator_result = task.get_future();
  std::thread terminator_thread{std::move(task)};

  test.Run(OpTester::ExpectResult::kExpectFailure, "Exiting due to terminate flag being set to true",
           {kTensorrtExecutionProvider, kOpenVINOExecutionProvider}, &session_run_options);  // Disable TensorRT on unsupported data type BOOL

  // call get to propagate any exception
  terminator_result.get();

  // done with the thread
  terminator_thread.join();
}

// Add basic test to trigger types override logic in Graph::InferAndVerifySubgraphTypes as well as
// type/shape inferencing for subgraph to flow the type/shape info through
// subgraph.PerformTypeAndShapeInferencing(options).
// In this test, main graph has original input/expected output defined as "double" where the subgraph as "float".
// Expectation is types should get propagated properly in subgraph and yield correct output
//
// TODO - when the input/output type in main graph is float16, extra Cast nodes will be added and type input type
//        will be changed by InsertCastTransformer for graph execution thus causes type mismatch failure.
//        Need to investigate how InsertCastTransformer works in future.
TEST(Loop, SubgraphTypeOverride) {
  auto create_subgraph = [](const RunOptions&) {
    Model model("Loop subgraph", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    /*
            Inputs: iter_num, cond_in, fake_in, loop carried state variables.

         iter_num_in    cond_in      fake_in   [outer_scope_0]
           (unused)        |            |            |
                       [Identity]  [Identity]    [Identity]
                           |            |            |
                        cond_out    fake_out   loop_var_0_out
    */

    // graph inputs types.
    TypeProto int64_scalar;
    int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
    int64_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto bool_scalar;
    bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
    bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim();

    // graph inputs
    auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
    auto& cond_in = graph.GetOrCreateNodeArg("cond_in", &bool_scalar);
    auto& fake_in = graph.GetOrCreateNodeArg("fake_in", &float_tensor);

    // outer scope value. need type but not shape.
    auto& outer_scope_0 = graph.GetOrCreateNodeArg("outer_scope_0", &float_tensor);

    // add so that we don't end up with it being considered a graph input
    graph.AddOuterScopeNodeArg("outer_scope_0");

    // graph outputs
    auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
    auto& fake_out = graph.GetOrCreateNodeArg("fake_out", &float_tensor);
    auto& loop_var_0_out = graph.GetOrCreateNodeArg("loop_var_0_out", &float_tensor);

    // cond_in -> cond_out
    {
      inputs = {&cond_in};
      outputs = {&cond_out};

      graph.AddNode("cond_in_identity", "Identity", "Forward cond_in to cond_out", inputs, outputs);
    }

    // fake_in -> fake_out
    {
      inputs = {&fake_in};
      outputs = {&fake_out};

      graph.AddNode("fake_in_identity", "Identity", "Forward fake_in to fake_out", inputs, outputs);
    }

    // outer_scope_0 -> loop_var_0_out
    {
      inputs = {&outer_scope_0};
      outputs = {&loop_var_0_out};

      graph.AddNode("loop_var_out", "Identity", "Forward outer_scope_0 to loop_var_0_out", inputs, outputs);
    }

    graph.SetInputs({&iter_num_in, &cond_in, &fake_in});
    graph.SetOutputs({&cond_out, &fake_out, &loop_var_0_out});

    auto status = graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return graph.ToGraphProto();
  };

  LoopOpTester test{{}, create_subgraph};

  test.AddInput<int64_t>("M", {1}, {1});
  test.AddOptionalInputEdge<bool>();  // 'cond' is optional in this test so don't provide it
  test.AddInput<double>("fake", {1}, {0.f});
  test.AddInput<double>("outer_scope_0", {1}, {kOuterNodeAddValue});

  test.AddOutput<double>("loop_fake_final", {1}, {0.f});
  test.AddOutput<double>("loop_var_0_final", {1, 1}, {kOuterNodeAddValue});
  test.AddOutput<int64_t>("outer_scope_0_out", {1}, {int64_t(kOuterNodeAddValue)});

  OrtRunOptions session_run_options;
  session_run_options.run_tag = "Loop.SubgraphTypeOverride";

  Graph::ResolveOptions options;
  options.override_types = true;
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider}, &session_run_options, nullptr,
           ExecutionMode::ORT_SEQUENTIAL, options);
}

// Regression test that a subgraph input overrides an outer scope value of the same name.
// Replicate issue from https://github.com/onnx/onnx/issues/2082
TEST(Loop, SubgraphInputShadowsOuterScopeValue) {
  SessionOptions so;
  so.session_logid = "SubgraphInputShadowsOuterScopeValue";

  InferenceSession session_object{so, GetEnvironment()};
  Status st;
  ASSERT_TRUE((st = session_object.Load("testdata/subgraph_input_shadows_outer_scope_value.onnx")).IsOK()) << st;
  ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st;

  // prepare inputs
  std::vector<int64_t> scalar = {1};
  std::vector<float> a = {3.f}, b = {6.f};
  std::vector<int64_t> trip_count = {10};
  std::vector<bool> keep_going = {true};

  NameMLValMap feeds;
  OrtValue ml_value;

  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], scalar, a, &ml_value);
  feeds.insert(std::make_pair("a", ml_value));
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], scalar, b, &ml_value);
  feeds.insert(std::make_pair("b", ml_value));
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], scalar, trip_count, &ml_value);
  feeds.insert(std::make_pair("max_trip_count", ml_value));
  CreateMLValue<bool>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], scalar, keep_going, &ml_value);
  feeds.insert(std::make_pair("keep_going_inp", ml_value));

  // prepare outputs
  std::vector<std::string> output_names{"b", "user_defined_vals"};
  std::vector<OrtValue> fetches;

  // Now run
  onnxruntime::RunOptions run_options;
  st = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(st.IsOK()) << st;
  ASSERT_EQ(2u, fetches.size());

  // prepare expected outputs
  float expected_value_b = 6.f;
  std::vector<int64_t> expected_dims_user_defined_vals = {2, 1};
  std::vector<float> expected_user_defined_vals = {-6.f, 12.f};

  auto& b_out = fetches[0].Get<Tensor>();
  TensorShape expected_shape(scalar);
  ASSERT_EQ(expected_shape, b_out.Shape());
  ASSERT_EQ(b_out.DataAsSpan<float>()[0], expected_value_b);

  auto user_defined_vals_out = fetches[1].Get<Tensor>().DataAsSpan<float>();
  ASSERT_EQ(expected_user_defined_vals.size(), static_cast<size_t>(user_defined_vals_out.size()));
  for (size_t i = 0, end = expected_user_defined_vals.size(); i < end; ++i) {
    ASSERT_THAT(user_defined_vals_out[i], testing::FloatEq(expected_user_defined_vals[i]));
  }
}

TEST(Loop, Opset11WithNoVariadicInputsAndOutputs) {
  auto create_subgraph = []() {
    Model model("Loop opset 11 op body graph", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    // graph inputs types.
    // iteration number
    TypeProto int64_scalar;
    int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
    int64_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    // loop condition
    TypeProto bool_scalar;
    bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
    bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    // graph output types
    // constant_out
    TypeProto float_scalar;
    float_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

    // graph inputs
    auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
    auto& cond_in = graph.GetOrCreateNodeArg("cond_in", &bool_scalar);

    // graph outputs
    auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
    auto& constant_out = graph.GetOrCreateNodeArg("constant_out", &float_scalar);

    // cond_in -> cond_out
    {
      inputs = {&cond_in};
      outputs = {&cond_out};

      graph.AddNode("cond_in_identity", "Identity", "Forward cond_in to cond_out", inputs, outputs);
    }

    // produce constant_out
    {
      outputs = {&constant_out};

      TensorProto constant_tensor_proto;

      auto& constant_node = graph.AddNode("constant_out", "Constant", "Produce constant_out", {}, outputs);

      AttributeProto attr_proto;
      attr_proto.set_name("value");
      attr_proto.set_type(AttributeProto_AttributeType_TENSOR);

      auto* constant_attribute_tensor_proto = attr_proto.mutable_t();
      constant_attribute_tensor_proto->mutable_dims()->Clear();                    // scalar
      constant_attribute_tensor_proto->set_data_type(TensorProto_DataType_FLOAT);  // float scalar
      *constant_attribute_tensor_proto->mutable_float_data()->Add() = 1.0f;        // float scalar with value 1.0f

      constant_node.AddAttributeProto(std::move(attr_proto));
    }

    graph.SetInputs({&iter_num_in, &cond_in});
    graph.SetOutputs({&cond_out, &constant_out});

    auto status = graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return graph.ToGraphProto();
  };

  OpTester test("Loop", 11);
  auto body = create_subgraph();
  test.AddAttribute<GraphProto>("body", body);
  test.AddInput<int64_t>("M", {1}, {1});
  test.AddInput<bool>("cond", {1}, {true});
  // This 'Loop' has no variadic inputs to test the spec of 'Loop' opset 11 which allows
  // 'Loop' to be used without variadic inputs

  test.AddOutput<float>("loop_scan_out", {1}, {1.0f});

  // Disable TensorRT on unsupported data type BOOL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Test a combination of things:
// Subgraph input for loop state var has no type and is not used in the Loop subgraph (used in nested If subgraph)
// Loop subgraph calls an If where the loop state var is an implicit input so it has no shape due to a loop state
// var being able to change shape on each iteration.
TEST(Loop, PassThroughSubgraphInputNoTypeOrShape) {
  // both subgraphs of the If just pass through an outer scope value
  auto create_if_subgraph = [](bool is_then) {
    Model model("if_branch_subgraph", true, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    auto& outer_scope_0 = graph.GetOrCreateNodeArg("loop_state_var", nullptr);
    graph.AddOuterScopeNodeArg("loop_state_var");

    TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

    auto& if_out = graph.GetOrCreateNodeArg(is_then ? "if_then_out" : "if_else_out", &float_tensor);
    graph.AddNode("if_out", "Identity", "pass through", {&outer_scope_0}, {&if_out});

    auto status = graph.Resolve();
    // Resolve will have actually errored out as we don't have type info for the input. That's valid for a subgraph
    // but not for a main graph but the Resolve doesn't know that it's handling a subgraph. We could add a way to
    // tell it but generally it's only our unit tests creating a graph this way.
    // The GraphProto will still be correct.
    EXPECT_NE(status, Status::OK());

    return graph.ToGraphProto();
  };

  auto create_subgraph = [&create_if_subgraph]() {
    Model model("loop_subgraph", true, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    /*  Inputs: iter_num, cond_in, loop carried state variables.

         iter_num_in    cond_in     [loop_state_var]
           (unused)        |               |
                       [Identity]         [If]   (both branches in If return loop_state_var via Identity node)
                           |               |
                        cond_out     loop_var_0_out
    */

    // graph inputs types.
    TypeProto int64_scalar;
    int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
    int64_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto bool_scalar;
    bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
    bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    // graph inputs
    auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
    auto& cond_in = graph.GetOrCreateNodeArg("cond_in", &bool_scalar);

    auto& loop_state_var = graph.GetOrCreateNodeArg("loop_state_var", nullptr);

    // graph outputs
    auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
    auto& loop_var_0_out = graph.GetOrCreateNodeArg("loop_var_0_out", nullptr);

    // cond_in -> cond_out
    {
      inputs = {&cond_in};
      outputs = {&cond_out};

      graph.AddNode("cond_in_identity", "Identity", "Forward cond_in to cond_out", inputs, outputs);
    }

    // loop_state_var -> If(cond_in) -> loop_var_0_out
    {
      inputs = {&cond_in};
      outputs = {&loop_var_0_out};

      auto& node = graph.AddNode("loop_var_out", "If", "If with loop_state_var as implicit_input", inputs, outputs);
      node.AddAttribute("then_branch", create_if_subgraph(true));
      node.AddAttribute("else_branch", create_if_subgraph(false));
    }

    graph.SetInputs({&iter_num_in, &cond_in, &loop_state_var});
    graph.SetOutputs({&cond_out, &loop_var_0_out});

    auto status = graph.Resolve();
    // same reason this will fail as in create_if_subgraph - still creates a valid subgraph GraphProto
    EXPECT_NE(status, Status::OK());

    return graph.ToGraphProto();
  };

  OpTester test("Loop", 11);
  auto body = create_subgraph();
  test.AddAttribute<GraphProto>("body", body);
  test.AddInput<int64_t>("M", {1}, {1});
  test.AddInput<bool>("cond", {1}, {true});
  test.AddInput<float>("initial_value", {1}, {123.f});

  test.AddOutput<float>("loop_var_0_final", {1}, {123.f});

  // Disable TensorRT on unsupported data type BOOL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(Loop, BugFixIssue4031_implicit_input_handling) {
  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::Level2;  // we need constant folding to run
  InferenceSession session_object{so, GetEnvironment()};
  static constexpr const ORTCHAR_T* MODEL_URI = ORT_TSTR("testdata/ort_github_issue_4031.onnx");

  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  onnxruntime::RunOptions run_options;
  run_options.run_tag = "BugFixIssue4031_implicit_input_handling";

  // prepare inputs
  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], {1}, {123.f},
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("state_var_in", ml_value));

  // prepare outputs
  std::vector<std::string> output_names{"state_var_out"};
  std::vector<OrtValue> fetches;

  // Now run
  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));

  const auto& output = fetches[0].Get<Tensor>();
  ASSERT_TRUE(output.Shape().Size() == 1);
  ASSERT_TRUE(output.Data<float>()[0] == 125.f);
}

// check the optimization in AllocationPlanner doesn't affect the iteration count when it is passed through
// and becomes a subgraph output. if we prevent a separate allocation for the output via the optimization
// the final value will be repeated in the loop output instead of the value from each iteration
TEST(Loop, IterationCountAsOutput) {
  auto create_subgraph = []() {
    Model model("Iter_num_in subgraph output", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    /* Inputs: iter_num, cond_in, loop carried state variables.

         iter_num_in    cond_in
             |             |
         [Identity]   [Identity]
             |             |
     loop_var_0_out    cond_out
    */

    // graph inputs types.
    TypeProto int64_scalar;
    int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
    int64_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto bool_scalar;
    bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
    bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    // graph inputs
    auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
    auto& cond_in = graph.GetOrCreateNodeArg("cond_in", &bool_scalar);

    // graph outputs
    auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
    auto& loop_var_0_out = graph.GetOrCreateNodeArg("loop_var_0_out", &int64_scalar);

    // iter_num_in -> loop_var_0_out
    {
      inputs = {&iter_num_in};
      outputs = {&loop_var_0_out};

      graph.AddNode("loop_var_out", "Identity", "Forward cond_in to loop_var_0_out", inputs, outputs);
    }

    // cond_in -> cond_out
    {
      inputs = {&cond_in};
      outputs = {&cond_out};

      graph.AddNode("cond_in_identity", "Identity", "Forward cond_in to cond_out", inputs, outputs);
    }

    graph.SetInputs({&iter_num_in, &cond_in});
    graph.SetOutputs({&cond_out, &loop_var_0_out});

    auto status = graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return graph.ToGraphProto();
  };

  OpTester test("Loop", 11);
  auto body = create_subgraph();
  test.AddAttribute<GraphProto>("body", body);
  test.AddInput<int64_t>("M", {1}, {3});
  test.AddInput<bool>("cond", {1}, {true});

  test.AddOutput<int64_t>("loop_var_0_final", {3, 1}, {0, 1, 2});

  // Disable TensorRT on unsupported data type BOOL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

#if defined(USE_CUDA) || defined(USE_ROCM)
// test that when part of the subgraph run on CUDA/ROCm it executes successfully
TEST(Loop, MixedExecutionProviders) {
  RunOptions options{};
  options.mixed_execution_providers = true;

  ExitDueToCond(options);
}
#endif

// Test sequence as Loop carried dependency (opset-13 allows this)
TEST(Loop, SequenceAsLoopCarriedDependency) {
  auto create_subgraph = []() {
    Model model("sequence in Loop subgraph carried dependency", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    /* Subgraph Adds inserted_tensor (float tensor) to a sequence across iterations

    Inputs: iter_num, cond_in, loop_var_0_in

          loop_var_0_in   inserted_tensor          cond_in            iter_num
                |             |                      |                (unused)
         [SequenceInsert]-----/                  [Identity]
                |                                    |
                |                                 cond_out
           loop_var_0_out
   */

    // graph inputs types.
    TypeProto int64_scalar;
    int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
    int64_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto bool_scalar;
    bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
    bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto float_tensor_sequence;
    auto* tensor_type = float_tensor_sequence
                            .mutable_sequence_type()
                            ->mutable_elem_type()
                            ->mutable_tensor_type();

    tensor_type->set_elem_type(TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim();

    // graph inputs
    auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
    auto& cond_in = graph.GetOrCreateNodeArg("cond_in", &bool_scalar);
    auto& loop_var_0_in = graph.GetOrCreateNodeArg("loop_var_0_in", &float_tensor_sequence);

    // graph outputs
    auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
    auto& loop_var_0_out = graph.GetOrCreateNodeArg("loop_var_0_out", &float_tensor_sequence);

    // outer scope values. need type but not shape.
    auto& inserted_tensor = graph.GetOrCreateNodeArg("inserted_tensor", &float_tensor);

    // add it to the outer scope so that we don't end up with it being considered a graph input
    graph.AddOuterScopeNodeArg("inserted_tensor");

    // cond_in -> cond_out
    {
      inputs = {&cond_in};
      outputs = {&cond_out};

      graph.AddNode("cond_in_identity", "Identity", "Forward cond_in to cond_out", inputs, outputs);
    }

    // iter_num_in -> loop_var_0_out
    {
      inputs = {&loop_var_0_in, &inserted_tensor};
      outputs = {&loop_var_0_out};

      graph.AddNode("loop_var_out", "SequenceInsert", "append to sequence across iterations", inputs, outputs);
    }

    graph.SetInputs({&iter_num_in, &cond_in, &loop_var_0_in});
    graph.SetOutputs({&cond_out, &loop_var_0_out});

    // add an initializer for the tensor that will be inserted into the sequence on every iterations
    {
      TensorProto inserted_tensor_proto;
      inserted_tensor_proto.set_name("inserted_tensor");
      inserted_tensor_proto.add_dims(1);
      inserted_tensor_proto.add_float_data(1.f);
      inserted_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      graph.AddInitializedTensor(inserted_tensor_proto);
    }

    auto status = graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return graph.ToGraphProto();
  };

  OpTester test("Loop", 13);
  auto body = create_subgraph();
  test.AddAttribute<GraphProto>("body", body);

  test.AddInput<int64_t>("M", {1}, {3});
  test.AddInput<bool>("cond", {1}, {true});

  SeqTensors<float> seq_input;
  test.AddSeqInput("loop_var_0_orig", seq_input);

  SeqTensors<float> seq_output;
  seq_output.AddTensor({1}, {1.f});
  seq_output.AddTensor({1}, {1.f});
  seq_output.AddTensor({1}, {1.f});
  test.AddSeqOutput("loop_var_0_final", seq_output);

  // Disable TensorRT on unsupported data type BOOL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

#if !defined(DISABLE_OPTIONAL_TYPE)

TEST(Loop, OptionalTypeAsLoopCarriedDependency) {
  auto create_subgraph = [](bool is_optional_tensor_type) {
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version.insert({"", 16});  // Opset 16 model

    Model model("optional type in Loop subgraph carried dependency", false, ModelMetaData(), PathString(), {},
                domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>{},
                DefaultLoggingManager().DefaultLogger());

    auto& graph = model.MainGraph();

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    /* Subgraph Passes an optional tensor through an Identity op on each loop iteration

    Inputs: iter_num, cond_in, loop_var_0_in

          loop_var_0_in                           cond_in            iter_num
                |                                    |                (unused)
           [Identity]                            [Identity]
                |                                    |
                |                                 cond_out
           loop_var_0_out
   */

    // graph inputs types.
    TypeProto int64_scalar;
    int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
    int64_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto bool_scalar;
    bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
    bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    TypeProto optional;
    auto* tensor_type = is_optional_tensor_type ? optional
                                                      .mutable_optional_type()
                                                      ->mutable_elem_type()
                                                      ->mutable_tensor_type()
                                                : optional
                                                      .mutable_optional_type()
                                                      ->mutable_elem_type()
                                                      ->mutable_sequence_type()
                                                      ->mutable_elem_type()
                                                      ->mutable_tensor_type();

    tensor_type->set_elem_type(TensorProto_DataType_FLOAT);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(1);

    // graph inputs
    auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
    auto& cond_in = graph.GetOrCreateNodeArg("cond_in", &bool_scalar);
    auto& loop_var_0_in = graph.GetOrCreateNodeArg("loop_var_0_in", &optional);

    // graph outputs
    auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
    auto& loop_var_0_out = graph.GetOrCreateNodeArg("loop_var_0_out", &optional);

    // cond_in -> cond_out
    {
      inputs = {&cond_in};
      outputs = {&cond_out};

      graph.AddNode("cond_in_identity", "Identity", "Forward cond_in to cond_out", inputs, outputs);
    }

    // iter_num_in -> loop_var_0_out
    {
      inputs = {&loop_var_0_in};
      outputs = {&loop_var_0_out};

      graph.AddNode("loop_var_out", "Identity", "optional pass-through", inputs, outputs);
    }

    graph.SetInputs({&iter_num_in, &cond_in, &loop_var_0_in});
    graph.SetOutputs({&cond_out, &loop_var_0_out});

    auto status = graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return graph.ToGraphProto();
  };

  // CASE 1: Optional tensor + none
  {
    OpTester test("Loop", 16);  // Opset 16 supports optional type

    auto body = create_subgraph(true);
    test.AddAttribute<GraphProto>("body", body);

    test.AddInput<int64_t>("M", {1}, {3});
    test.AddInput<bool>("cond", {1}, {true});
    test.AddOptionalTypeTensorInput<float>("A", {}, nullptr);   // None
    test.AddOptionalTypeTensorOutput<float>("Y", {}, nullptr);  // None

    // Disable TensorRT on unsupported data type BOOL
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  }

  // CASE 2: Optional tensor + non-none
  {
    OpTester test("Loop", 16);  // Opset 16 supports optional type

    auto body = create_subgraph(true);
    test.AddAttribute<GraphProto>("body", body);

    test.AddInput<int64_t>("M", {1}, {3});
    test.AddInput<bool>("cond", {1}, {true});
    std::initializer_list<float> data = {-1.0856307f, 0.99734545f};
    test.AddOptionalTypeTensorInput<float>("A", {2}, &data);   // Non-None
    test.AddOptionalTypeTensorOutput<float>("Y", {2}, &data);  // Non-None

    // Disable TensorRT on unsupported data type BOOL
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  }

  // CASE 3: Optional tensor sequence + none
  {
    OpTester test("Loop", 16);  // Opset 16 supports optional type

    auto body = create_subgraph(false);
    test.AddAttribute<GraphProto>("body", body);

    test.AddInput<int64_t>("M", {1}, {3});
    test.AddInput<bool>("cond", {1}, {true});

    test.AddOptionalTypeSeqInput<float>("A", nullptr);   // None
    test.AddOptionalTypeSeqOutput<float>("Y", nullptr);  // None

    // Disable TensorRT on unsupported data type BOOL
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  }

  // CASE 4: Optional tensor sequence + non-none
  {
    OpTester test("Loop", 16);  // Opset 16 supports optional type

    auto body = create_subgraph(false);
    test.AddAttribute<GraphProto>("body", body);

    test.AddInput<int64_t>("M", {1}, {3});
    test.AddInput<bool>("cond", {1}, {true});

    SeqTensors<float> seq;
    seq.AddTensor({1}, {1.f});
    seq.AddTensor({1}, {1.f});
    seq.AddTensor({1}, {1.f});

    test.AddOptionalTypeSeqInput<float>("A", &seq);   // Non-None
    test.AddOptionalTypeSeqOutput<float>("Y", &seq);  // Non-None

    // Disable TensorRT on unsupported data type BOOL
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  }
}

#endif

}  // namespace test
}  // namespace onnxruntime
