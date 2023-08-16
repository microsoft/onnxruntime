// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"
// #include "core/framework/customregistry.h"
#include "core/framework/session_state.h"
#include "core/providers/cpu/controlflow/if.h"
#include "test/providers/provider_test_utils.h"
#include "core/session/inference_session.h"

#include "test/util/include/default_providers.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

namespace {
struct RunOptions {
  bool include_dim_values_in_main_graph = false;
  int symbolic_dim_value_in_main_graph = -1;
  bool include_dim_values_in_subgraph = true;
  bool mixed_execution_providers = false;
};
}  // namespace

static const ONNX_NAMESPACE::GraphProto CreateSubgraph(bool then_branch, const RunOptions& options);

/*
 Main graph

 split_input          if_cond      if_graph_input_0,
      |                   |              |
   [Split]                |          [Identity]
      |                   |              |
      |                   |         if_input_0
      |  split_out_0      |              |
      ------------------[If]--------------   (see below for then/else subgraphs in If node)
         split_out_1      |
                          |
                       if_out_0
*/

class IfOpTester : public OpTester {
 public:
  IfOpTester(const RunOptions& options, int opset_version = 10)
      : OpTester("If", opset_version), options_{options}, opset_version_(opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    // Graph inputs are 0:Split input, 1:Cond for If, 2:if input
    ASSERT_EQ(graph_input_defs.size(), 3u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* split_input = graph_input_defs[0];
    NodeArg* if_cond_input = graph_input_defs[1];
    NodeArg* if_input = graph_input_defs[2];

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    // add Split node
    {
      TypeProto split_out_type;
      split_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
      auto& split_out_0 = graph.GetOrCreateNodeArg("split_out_0", &split_out_type);
      auto& split_out_1 = graph.GetOrCreateNodeArg("split_out_1", &split_out_type);

      inputs = {split_input};
      outputs = {&split_out_0, &split_out_1};

      auto& split_node = graph.AddNode("split", "Split", "Split into 2", inputs, outputs);
      if (opset_version_ > 10) {
        AttributeProto attr_proto;
        attr_proto.set_name("split");
        attr_proto.set_type(AttributeProto_AttributeType_INTS);

        auto* split_attribute = attr_proto.mutable_ints();
        *split_attribute->Add() = 1;  // split "unevenly" to create different shapes across the "then" and "else" branches
        *split_attribute->Add() = 2;

        split_node.AddAttributeProto(std::move(attr_proto));
      }
    }

    // add If node
    {
      inputs = {if_cond_input};
      outputs = {graph_output_defs[0]};

      auto& if_node = graph.AddNode("if", "If", "If node", inputs, outputs);

      auto then_proto = CreateSubgraph(true, options_);
      auto else_proto = CreateSubgraph(false, options_);
      if_node.AddAttribute("then_branch", then_proto);
      if_node.AddAttribute("else_branch", else_proto);
    }

    // add Identity node so if_graph_input_0 comes from graph inputs
    {
      inputs = {if_input};
      outputs = {&graph.GetOrCreateNodeArg("if_input_0", if_input->TypeAsProto())};
      graph.AddNode("identity", "Identity", "Pass if input through from graph inputs.", inputs, outputs);
    }
  }

 private:
  RunOptions options_;
  int opset_version_;
};

/* Subgraphs looks like this. All inputs come from outer scope so we just
   create a NodeArg with the input name. The numbers in [] are the values the tests are expected to produce
   as output from each node.

THEN branch (all opset versions)
    split_out_0    if_input_0   [1]
             \          |
       [1]    \         |
               \------[Add]
                        |
                   add_out_0    [2]

ELSE branch (opset 10 and below)
    split_out_1    if_input_0   [1]
            \          |
      [10]   \         |
              \------[Add]
                        |
                   add_out_1    [11]

ELSE branch (opset 11 and above)
    split_out_1    if_input_0   [1]
            \          |
  [10, 10]   \         |
              \------[Add]
                        |
                   add_out_1    [11, 11]
*/

static const ONNX_NAMESPACE::GraphProto CreateSubgraph(bool then_branch, const RunOptions& options) {
  bool include_dim_values = options.include_dim_values_in_subgraph;
  bool sym_dim_zero = options.symbolic_dim_value_in_main_graph == 0;

  Model model(then_branch ? "If_then" : "If_else", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  const std::string suffix = then_branch ? "0" : "1";

  // graph input has to have type and rank even though it's an outer scope value.
  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* mutable_dim = input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim();
  if (include_dim_values) {
    mutable_dim->set_dim_value(1);
  } else if (sym_dim_zero) {
    mutable_dim->set_dim_param("symbolic");
  }

  // outer scope values
  auto& split_output = graph.GetOrCreateNodeArg("split_out_" + suffix, &input_tensor_type);
  auto& if_input = graph.GetOrCreateNodeArg("if_input_0", &input_tensor_type);

  // add so that we don't end up with it being considered a graph input
  graph.AddOuterScopeNodeArg("split_out_" + suffix);
  graph.AddOuterScopeNodeArg("if_input_0");

  {
    // Add

    // graph output has to have type and shape
    TypeProto add_output_tensor;
    add_output_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    mutable_dim = add_output_tensor.mutable_tensor_type()->mutable_shape()->add_dim();
    if (include_dim_values) {
      mutable_dim->set_dim_value(1);
    } else if (sym_dim_zero) {
      mutable_dim->set_dim_param("symbolic");
    }

    auto& add_out = graph.GetOrCreateNodeArg("add_out_" + suffix, &add_output_tensor);

    inputs = {&split_output, &if_input};
    outputs = {&add_out};

    graph.AddNode("add", "Add", "Add two inputs.", inputs, outputs);
  }

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto& proto = graph.ToGraphProto();

  return proto;
}

void RunTest(bool condition_value,
             RunOptions options,
             bool is_tensorrt_supported = true,
             OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
             const std::string& failure_message = "",
             int opset_version = 10) {
  IfOpTester test{options, opset_version};

  test.AddShapeToTensorData(options.include_dim_values_in_main_graph,
                            options.symbolic_dim_value_in_main_graph);

  // add the main graph inputs and outputs.
  // we will handle the 'If' inputs in the AddNodes override, and as 'If' is the last node
  // it's outputs are 1:1 with the graph outputs.

  // simple tensor that we split into 2, and use one output for the 'then' and one for the 'else' branch in the If
  if (opset_version != 11) {
    test.AddInput<float>("split_input", {2}, {1.f, 10.f});
  } else {
    // in the opset 11 test case, we are going to split "unevenly", to create different shapes in "then" and "else" branches
    test.AddInput<float>("split_input", {3}, {1.f, 10.f, 10.f});
  }

  // graph input to specify which branch to take
  test.AddInput<bool>("if_cond", {1}, {condition_value});

  test.AddInput<float>("if_graph_input_0", {1}, {1.f});

  if (opset_version != 11 && condition_value) {
    test.AddOutput<float>("if_out_0", {1}, {2.f});
  } else if (opset_version != 11) {
    test.AddOutput<float>("if_out_0", {1}, {11.f});
  } else if (opset_version == 11 && condition_value) {
    test.AddOutput<float>("if_out_0", {1}, {2.f});
  } else if (opset_version == 11) {
    test.AddOutput<float>("if_out_0", {2}, {11.f, 11.f});
  }

  std::unordered_set<std::string> excluded_providers;
  // Disable TensorRT on SymbolicShape or NoShape tests
  if (!is_tensorrt_supported) {
    excluded_providers.insert(kTensorrtExecutionProvider);
  }
  if (options.mixed_execution_providers) {
    // we want the CUDA provider to be first, and the CPU provider second. all except the If should run on
    // CUDA given that, which creates the scenario where we need to copy to/from CPU to execute the If node correctly.
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    execution_providers.push_back(DefaultCpuExecutionProvider());

    test.Run(expect_result, failure_message, excluded_providers, nullptr, &execution_providers);
  } else {
    test.Run(expect_result, failure_message, excluded_providers);
  }
}

TEST(If, ShapeInMainGraph_NoShapeInSubgraph_True) {
  RunOptions options{};
  options.include_dim_values_in_main_graph = true;
  options.include_dim_values_in_subgraph = false;

  RunTest(true, options, false);
}

TEST(If, ShapeInMainGraph_NoShapeInSubgraph_False) {
  RunOptions options{};
  options.include_dim_values_in_main_graph = true;
  options.include_dim_values_in_subgraph = false;

  RunTest(false, options, false);
}

TEST(If, NoShapeInMainGraph_ShapeInSubgraph_True) {
  RunOptions options{};
  options.include_dim_values_in_main_graph = false;
  options.include_dim_values_in_subgraph = true;

  RunTest(true, options, false);
}

TEST(If, NoShapeInMainGraph_ShapeInSubgraph_False) {
  RunOptions options{};
  options.include_dim_values_in_main_graph = false;
  options.include_dim_values_in_subgraph = true;

  RunTest(false, options, false);
}

#ifdef USE_CUDA
TEST(If, MixedExecutionProviders) {
  RunOptions options{};
  options.mixed_execution_providers = true;
  RunTest(true, options);
}

TEST(If, MixedExecutionProvidersOpset11) {
  RunOptions options{};
  options.mixed_execution_providers = true;
  RunTest(true, options, false, test::OpTester::ExpectResult::kExpectSuccess, "", 11);
}

TEST(If, MixedExecutionProvidersNoShapeInSubgraph) {
  RunOptions options{};
  options.mixed_execution_providers = true;
  options.include_dim_values_in_main_graph = true;
  options.symbolic_dim_value_in_main_graph = 0;
  options.include_dim_values_in_subgraph = false;
  RunTest(true, options);
}
#endif  // USE_CUDA

TEST(If, SymbolicShapeInMainGraph_NoShapeInSubgraph_True) {
  RunOptions options;
  options.include_dim_values_in_main_graph = true;
  options.symbolic_dim_value_in_main_graph = 0;
  options.include_dim_values_in_subgraph = false;

  RunTest(true, options, false);
}

TEST(If, SymbolicShapeInMainGraph_NoShapeInSubgraph_False) {
  RunOptions options;
  options.include_dim_values_in_main_graph = true;
  options.symbolic_dim_value_in_main_graph = 0;
  options.include_dim_values_in_subgraph = false;

  RunTest(false, options, false);
}

TEST(If, Opset11ThenAndElseBranchesProduceDifferentOutputShapes) {
  RunOptions options{};
  options.include_dim_values_in_main_graph = true;
  options.include_dim_values_in_subgraph = false;

  RunTest(false, options, false, OpTester::ExpectResult::kExpectSuccess, "", 11);
}

// This is to test an "If" node with just "Constant" nodes in the "then" and "else" conditional branches
class IfOpTesterOnlyConstantNodesInConditionalBranches : public OpTester {
 public:
  IfOpTesterOnlyConstantNodesInConditionalBranches() : OpTester("If") {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    // Graph inputs are 0:Cond for If
    ASSERT_EQ(graph_input_defs.size(), 1u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* if_cond_input = graph_input_defs[0];

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    // add If node
    {
      inputs = {if_cond_input};
      outputs = {graph_output_defs[0]};

      auto& if_node = graph.AddNode("if", "If", "If node", inputs, outputs);

      auto CreateSubgraphWithConstantNode = [](bool then_branch, float value, std::vector<NodeArg*> outputs) {
        Model model_then(then_branch ? "Then" : "Else", false, DefaultLoggingManager().DefaultLogger());
        auto& graph_then = model_then.MainGraph();
        auto& then_constant_node = graph_then.AddNode(
            then_branch ? "Constant_Then" : "Constant_Else",
            "Constant",
            then_branch ? "Constant_Then" : "Constant_Else", {}, outputs);

        AttributeProto then_constant_attr_proto;
        then_constant_attr_proto.set_name("value");
        then_constant_attr_proto.set_type(AttributeProto_AttributeType_TENSOR);
        auto* then_constant_attr_tensor_proto = then_constant_attr_proto.mutable_t();
        then_constant_attr_tensor_proto->set_data_type(TensorProto_DataType_FLOAT);
        then_constant_attr_tensor_proto->add_dims(1);
        then_constant_attr_tensor_proto->add_float_data(value);  // Constant value of 10.f

        then_constant_node.AddAttributeProto(std::move(then_constant_attr_proto));

        auto status_then = graph_then.Resolve();
        EXPECT_EQ(status_then, Status::OK());

        auto& graphproto_then = graph_then.ToGraphProto();
        return graphproto_then;
      };

      if_node.AddAttribute("then_branch", CreateSubgraphWithConstantNode(true, 10.f, outputs));
      if_node.AddAttribute("else_branch", CreateSubgraphWithConstantNode(false, 1000.f, outputs));
    }
  }
};

// Context: Github issue #3900
TEST(If, ConditionalBranchesOnlyContainConstantNodes_ThenBranchExecution) {
  IfOpTesterOnlyConstantNodesInConditionalBranches test;
  test.AddInput<bool>("If_input", {1}, {true});
  test.AddOutput<float>("If_output", {1}, {10.f});
  test.Run();
}

// Context: Github issue #3900
TEST(If, ConditionalBranchesOnlyContainConstantNodes_ElseBranchExecution) {
  IfOpTesterOnlyConstantNodesInConditionalBranches test;
  test.AddInput<bool>("If_input", {1}, {false});
  test.AddOutput<float>("If_output", {1}, {1000.f});
  test.Run();
}

// This is to test an "If" node with just a "SequenceEmpty" node in the "then" and "else" conditional branches
class IfOpTesterWithSequencesAsOutput : public OpTester {
 public:
  IfOpTesterWithSequencesAsOutput() : OpTester("If", 13) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    // Graph inputs are 0:Cond
    ASSERT_EQ(graph_input_defs.size(), 1u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* if_cond_input = graph_input_defs[0];

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    // add If node
    {
      inputs = {if_cond_input};
      outputs = {graph_output_defs[0]};

      auto& if_node = graph.AddNode("if", "If", "If node", inputs, outputs);

      auto CreateSubgraphWithSequenceEmptyNode = [](bool then_branch, std::vector<NodeArg*> outputs) {
        Model subgraph(then_branch ? "Then" : "Else", false, DefaultLoggingManager().DefaultLogger());
        auto& graph = subgraph.MainGraph();

        // By default, the SequenceEmpty node will create an empty sequence of float tensors
        ORT_IGNORE_RETURN_VALUE(graph.AddNode(
            then_branch ? "SequenceEmpty_Then" : "SequenceEmpty_Else",
            "SequenceEmpty",
            then_branch ? "SequenceEmpty_Then" : "SequenceEmpty_Else", {}, outputs));

        auto status = graph.Resolve();
        EXPECT_EQ(status, Status::OK());

        auto& graphproto = graph.ToGraphProto();
        return graphproto;
      };

      if_node.AddAttribute("then_branch", CreateSubgraphWithSequenceEmptyNode(true, outputs));
      if_node.AddAttribute("else_branch", CreateSubgraphWithSequenceEmptyNode(false, outputs));
    }
  }
};

// opset-13 allows sequences as outputs for 'If' nodes
TEST(If, TestIfWithSequencesAsOutput) {
  IfOpTesterWithSequencesAsOutput test;
  test.AddInput<bool>("If_input", {1}, {true});
  SeqTensors<float> seq;  // empty sequence of float tensors
  test.AddSeqOutput("If_output", seq);
  test.Run();
}

#if !defined(DISABLE_OPTIONAL_TYPE)
// This is to test an "If" node with just an "Identity" node in the "then" and "else" conditional branches
class IfOpTesterWithOptionalTypeAsOutput : public OpTester {
 public:
  IfOpTesterWithOptionalTypeAsOutput() : OpTester("If", 16) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    // Graph inputs are 0:Cond
    ASSERT_EQ(graph_input_defs.size(), 2u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* identity_input = graph_input_defs[1];

    std::vector<NodeArg*> if_inputs = {graph_input_defs[0]};
    std::vector<NodeArg*> if_outputs = {graph_output_defs[0]};

    auto CreateSubgraphWithIdentityNode = [&identity_input, &if_outputs](bool then_branch) {
      std::unordered_map<std::string, int> domain_to_version;
      domain_to_version.insert({"", 16});  // Opset 16 model

      Model subgraph(then_branch ? "Then_subgraph" : "Else_subgraph", false, ModelMetaData(), PathString(), {},
                     domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>{},
                     DefaultLoggingManager().DefaultLogger());

      auto& graph = subgraph.MainGraph();

      auto& pass_through_identity_input = graph.GetOrCreateNodeArg("pass_through_identity_input",
                                                                   identity_input->TypeAsProto());
      graph.AddOuterScopeNodeArg("pass_through_identity_input");

      graph.AddNode(
          then_branch ? "Identity_Then" : "Identity_Else",
          "Identity",
          then_branch ? "Identity_Then" : "Identity_Else", {&pass_through_identity_input}, if_outputs);

      auto status = graph.Resolve();
      EXPECT_EQ(status, Status::OK());

      auto& graphproto = graph.ToGraphProto();
      return graphproto;
    };

    // If node
    {
      auto& if_node = graph.AddNode("if", "If", "If node", if_inputs, if_outputs);

      if_node.AddAttribute("then_branch", CreateSubgraphWithIdentityNode(true));
      if_node.AddAttribute("else_branch", CreateSubgraphWithIdentityNode(false));
    }

    // add Identity node so if_graph_input_0 comes from graph inputs
    {
      auto inputs = {identity_input};
      auto outputs = {&graph.GetOrCreateNodeArg("pass_through_identity_input", identity_input->TypeAsProto())};
      graph.AddNode("identity", "Identity", "Pass if input through from graph inpu.", inputs, outputs);
    }
  }
};

TEST(If, TestIfWithOptionalTypeTensorAsOutput) {
  // CASE 1: Optional tensor + none
  {
    IfOpTesterWithOptionalTypeAsOutput test;
    test.AddInput<bool>("If_input", {1}, {true});
    test.AddOptionalTypeTensorInput<float>("A", {}, nullptr);                            // None
    test.AddOptionalTypeTensorOutput<float>("Y", {}, nullptr);                           // None
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: opset 16 is not supported yet
  }

  // CASE 2: Optional tensor + non-none
  {
    IfOpTesterWithOptionalTypeAsOutput test;
    test.AddInput<bool>("If_input", {1}, {true});
    std::initializer_list<float> data = {-1.0856307f, 0.99734545f};
    test.AddOptionalTypeTensorInput<float>("A", {2}, &data);                             // Non-None
    test.AddOptionalTypeTensorOutput<float>("Y", {2}, &data);                            // Non-None
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: opset 16 is not supported yet
  }

  // CASE 3: Optional tensor sequence + none
  {
    IfOpTesterWithOptionalTypeAsOutput test;
    test.AddInput<bool>("If_input", {1}, {true});
    test.AddOptionalTypeSeqInput<float>("A", nullptr);                                   // None
    test.AddOptionalTypeSeqOutput<float>("Y", nullptr);                                  // None
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: opset 16 is not supported yet
  }

  // CASE 4: Optional tensor sequence + non-none
  {
    IfOpTesterWithOptionalTypeAsOutput test;
    test.AddInput<bool>("If_input", {1}, {true});

    SeqTensors<float> seq;
    seq.AddTensor({1}, {1.f});
    seq.AddTensor({1}, {1.f});
    seq.AddTensor({1}, {1.f});

    test.AddOptionalTypeSeqInput<float>("A", &seq);                                      // Non-None
    test.AddOptionalTypeSeqOutput<float>("Y", &seq);                                     // Non-None
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: opset 16 is not supported yet
  }
}

#endif

}  // namespace test
}  // namespace onnxruntime
