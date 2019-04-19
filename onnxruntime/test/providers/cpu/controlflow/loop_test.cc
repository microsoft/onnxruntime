// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <future>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/logging/logging.h"
#include "core/framework/session_state.h"
#include "core/session/inference_session.h"

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

namespace {
struct RunOptions {
  bool include_dim_values_in_main_graph = true;
  bool include_dim_values_in_subgraph = false;
  bool include_types_in_subgraph = false;
  bool mixed_execution_providers = false;
};
}
static const ONNX_NAMESPACE::GraphProto CreateSubgraph(const RunOptions& options);

static const float kOuterNodeAddValue = 3.f;
static const float kSumMax = 8.f;

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
    // add outer_scope_0 node
    {
      TypeProto float_scalar;
      float_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
      auto mutable_dim = float_scalar.mutable_tensor_type()->mutable_shape()->add_dim();
      mutable_dim->set_dim_value(1);

      {
        auto& output_arg = graph.GetOrCreateNodeArg("outer_scope_0", &float_scalar);
        auto& constant = graph.AddNode("outer_scope_constant", "Constant", "Constant in outer scope", {},
                                       {&output_arg});

        TensorProto value_tensor;
        value_tensor.add_dims(1);
        value_tensor.add_float_data(kOuterNodeAddValue);
        value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

        constant.AddAttribute("value", value_tensor);
      }
    }

    // add Loop node
    {
      auto& loop_node = graph.AddNode("loop", "Loop", "Loop node", graph_input_defs, graph_output_defs);

      auto body = create_subgraph_(options_);
      loop_node.AddAttribute("body", {body});
    }
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

  Model model("Loop subgraph");
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  /* Subgraph Adds outer_scope_0 to loop_var_0_in,
     Concats the iter_num to loop_var_1_in (test loop var that changes shape) so each iteration appends the iter_num
     to loop_var_1
     Loop output is the iter_num and sum for that iteration, so each iteration adds a pair to the overall output
     Inputs require Identity nodes to fix their order.

    Inputs: iter_num, cond_in, loop_var_in

 iter_num_in  loop_var_0_in [outer_scope_0] loop_var_1_in                                    cond_in
       |             |        /                  |                                           (unused)
     [Cast]        [Add]-----/                   |
       |             |                           |                          [Constant]
  iter_num_float  sum_0                          |          sum_0             /
       |           / | \                         |              \            /
    [Concat]------/  |  \---------------------[Concat]           \--[Less]--/
       |             |                           |                     |
       |         [Identity]                      |                     |
       |             |                           |                     |
   loop_out_0   loop_var_0_out             loop_var_1_out           cond_out

  */

  // graph inputs. must have type and at least rank
  TypeProto int64_scalar;
  int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int64_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto float_scalar;
  float_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* mutable_dim = float_scalar.mutable_tensor_type()->mutable_shape()->add_dim();
  if (include_dim_value) {
    mutable_dim->set_dim_value(1);
  }

  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  // dimension value changes on each iteration so just add a dimension but no value
  TypeProto float_tensor_single_dim{float_tensor};
  mutable_dim = float_tensor_single_dim.mutable_tensor_type()->mutable_shape()->add_dim();

  // graph inputs
  auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
  auto& cond_in = graph.GetOrCreateNodeArg("cond_in", &bool_scalar);
  auto& loop_var_0_in = graph.GetOrCreateNodeArg("loop_var_0_in", &float_scalar);
  auto& loop_var_1_in = graph.GetOrCreateNodeArg("loop_var_1_in", &float_tensor_single_dim);

  auto& iter_num_float = graph.GetOrCreateNodeArg("iter_num_float", &float_scalar);

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

    inputs = {&iter_num_float, &sum_0};
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
      auto& max_value_out = graph.GetOrCreateNodeArg("max_value_out", &float_scalar);
      auto& constant = graph.AddNode("constant_max_value", "Constant", "Constant with value kSumMax",
                                     {}, {&max_value_out});

      TensorProto value_tensor;
      value_tensor.add_dims(1);
      value_tensor.add_float_data(kSumMax);
      value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      constant.AddAttribute("value", value_tensor);

      cond_out = &graph.GetOrCreateNodeArg("cond_out", &bool_scalar);

      inputs = {&sum_0, &max_value_out};
      outputs = {cond_out};

      graph.AddNode("sum_less_than_max", "Less", "Check sum < kSumMax", inputs, outputs);
    }
  }

  graph.SetInputOrder({&iter_num_in, &cond_in, &loop_var_0_in, &loop_var_1_in});
  graph.SetOutputOrder({cond_out, loop_var_0_out, loop_var_1_out, loop_out_0});

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

  test.AddInput<int64_t>("M", {1}, {max_iterations});
  test.AddInput<bool>("cond", {1}, {true});

  test.AddInput<float>("loop_var_0_orig", {1}, {0.f});
  test.AddInput<float>("loop_var_0_orig", {1}, {0.f});

  test.AddOutput<float>("loop_var_0_final", {1}, {loop_var_0_final});
  test.AddOutput<float>("loop_var_1_final", loop_var_1_final_shape, loop_var_1_final);
  test.AddOutput<float>("loop_out_0_final", loop_out_0_final_shape, loop_out_0_final);

  if (options.mixed_execution_providers) {
    // we want the CUDA provider to be first, and the CPU provider second. all except the Loop node should run on
    // CUDA given that, which creates the scenario where we need to copy to/from CPU to execute the Loop node correctly.
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    execution_providers.push_back(DefaultCpuExecutionProvider());

    test.Run(expect_result, failure_message, {kTensorrtExecutionProvider}, nullptr, &execution_providers);
  } else {
    test.Run(expect_result, failure_message, {kTensorrtExecutionProvider});// Disable TensorRT because of unsupported data type INT64
  }
}

// exit due to hitting condition that the sum is < kSumMax which is 8
// this should take 3 iterations as we add 3 each time.
void ExitDueToCond(const RunOptions& options) {
  int64_t max_iterations = 5;
  const int64_t expected_num_iterations = 3;

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

TEST(Loop, ExitDueToCond_DimsInMainGraph) {
  RunOptions options{};
  options.include_dim_values_in_main_graph = true;
  options.include_dim_values_in_subgraph = false;
  options.include_dim_values_in_subgraph = false;

  ExitDueToCond(options);
}

TEST(Loop, ExitDueToCond_DimsInSubgraph) {
  RunOptions options{};
  options.include_dim_values_in_main_graph = false;
  options.include_dim_values_in_subgraph = true;
  options.include_types_in_subgraph = false;

  ExitDueToCond(options);
}

TEST(Loop, ExitDueToMaxIterations) {
  int64_t max_iterations = 2;
  const int64_t expected_num_iterations = 2;

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

TEST(Loop, InfiniteLoopTermination) {
  auto create_subgraph = [](const RunOptions&) {
    Model model("Infinite Loop subgraph");
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

    // graph inputs types. must have type and at least rank
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

    graph.SetInputOrder({&iter_num_in, &cond_in, &outer_scope_0});
    graph.SetOutputOrder({&cond_out, &loop_var_0_out});

    auto status = graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return graph.ToGraphProto();
  };

  LoopOpTester test{{}, create_subgraph};

  test.AddInput<int64_t>("M", {1}, {INT64_MAX});
  test.AddInput<bool>("cond", {1}, {true});
  test.AddInput<float>("fake", {1}, {0.f});

  test.AddOutput<float>("loop_var_0_final", {1}, {0.f});

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

  test.Run(OpTester::ExpectResult::kExpectFailure, "Exiting due to terminate flag being set to true", {kTensorrtExecutionProvider},
           &session_run_options);// Disable TensorRT on unsupported data type BOOL

  // call get to propagate any exception
  terminator_result.get();

  // done with the thread
  terminator_thread.join();
}

#ifdef USE_CUDA
// test that when part of the subgraph run on CUDA it executes successfully
TEST(Loop, MixedExecutionProviders) {
  RunOptions options{};
  options.mixed_execution_providers = true;

  ExitDueToCond(options);
}
#endif

}  // namespace test
}  // namespace onnxruntime
