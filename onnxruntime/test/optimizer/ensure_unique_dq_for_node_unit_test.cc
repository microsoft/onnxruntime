// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/ensure_unique_dq_for_node_unit.h"

#include "gtest/gtest.h"

#include "test/optimizer/graph_transform_test_builder.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {

namespace {

struct GraphConfig {
  size_t num_explicit_consumer_nodes{1};
  size_t num_inputs_per_explicit_consumer_node{1};
  bool has_graph_output{false};
  bool has_subgraph_consumer{false};
};

template <typename QuantType>
std::function<void(ModelTestBuilder&)> GetGraphBuilder(const GraphConfig& config, bool use_ms_domain_qdq_ops) {
  return [config, use_ms_domain_qdq_ops](ModelTestBuilder& builder) {
    const auto input_shape = std::vector<int64_t>{1, 2, 4};
    constexpr float scale = 0.5f;
    constexpr QuantType zero_point = 0;

    auto* dq_input = builder.MakeInput<QuantType>(input_shape, std::numeric_limits<QuantType>::min(),
                                                  std::numeric_limits<QuantType>::max());
    auto* dq_output = config.has_graph_output ? builder.MakeOutput() : builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(dq_input, scale, zero_point, dq_output, use_ms_domain_qdq_ops);

    for (size_t i = 0; i < config.num_explicit_consumer_nodes; ++i) {
      // use Concat for the explicit consumer node as it supports a variadic number of inputs
      auto* explicit_consumer_output = builder.MakeOutput();
      std::vector<NodeArg*> explicit_consumer_inputs(config.num_inputs_per_explicit_consumer_node, dq_output);
      auto& concat = builder.AddNode("Concat", explicit_consumer_inputs, {explicit_consumer_output});
      concat.AddAttribute("axis", int64_t{-1});
    }

    if (config.has_subgraph_consumer) {
      auto create_if_subgraph = [&](bool condition) {
        auto model = Model{"model for generating graph proto", true, DefaultLoggingManager().DefaultLogger()};
        auto& graph = model.MainGraph();

        const auto& dq_output_name = dq_output->Name();

        ONNX_NAMESPACE::TypeProto type_proto{};
        type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

        NodeArg& local_dq_output = graph.GetOrCreateNodeArg(dq_output_name, &type_proto);
        graph.AddOuterScopeNodeArg(local_dq_output.Name());

        NodeArg& out = graph.GetOrCreateNodeArg(condition ? "output_then" : "output_else", &type_proto);
        graph.AddNode("identity", "Identity", "pass through identity", {&local_dq_output}, {&out});

        graph.SetOutputs({&out});

        ORT_THROW_IF_ERROR(graph.Resolve());

        return graph.ToGraphProto();
      };

      auto* if_input = builder.MakeInitializerBool({}, {true});
      auto* if_output = builder.MakeOutput();
      Node& if_node = builder.AddNode("If", {if_input}, {if_output});
      if_node.AddAttribute("then_branch", create_if_subgraph(true));
      if_node.AddAttribute("else_branch", create_if_subgraph(false));
    }
  };
}

void RunEnsureUniqueDQForNodeUnitTest(const GraphConfig& config, int expected_dq_count) {
  auto run_tests = [config, expected_dq_count](bool use_ms_domain_qdq_ops, bool use_16bit_qdq_ops) {
    constexpr int opset_version = 12;
    const char* dequantize_linear_key = use_ms_domain_qdq_ops ? "com.microsoft.DequantizeLinear" : "DequantizeLinear";
    std::function<void(ModelTestBuilder&)> graph_builder_fn = use_16bit_qdq_ops
                                                                  ? GetGraphBuilder<uint16_t>(config, use_ms_domain_qdq_ops)
                                                                  : GetGraphBuilder<uint8_t>(config, use_ms_domain_qdq_ops);

    {
      SCOPED_TRACE("test with standalone transformer");

      auto post_transform_check_fn = [expected_dq_count, dequantize_linear_key](const Graph& graph) {
        const auto op_counts = CountOpsInGraph(graph);
        const auto actual_dq_count = OpCount(op_counts, dequantize_linear_key);
        ORT_RETURN_IF_NOT(actual_dq_count == expected_dq_count,
                          "Expected DQ count: ", expected_dq_count, ", actual: ", actual_dq_count);
        return Status::OK();
      };

      EXPECT_STATUS_OK(TestGraphTransformer(
          graph_builder_fn,
          opset_version,
          DefaultLoggingManager().DefaultLogger(),
          std::make_unique<EnsureUniqueDQForNodeUnit>(),
          TransformerLevel::Level1,
          5,
          {},
          post_transform_check_fn));
    }

    {
      SCOPED_TRACE("test with basic transformers");

      auto post_transform_check_fn = [expected_dq_count,
                                      dequantize_linear_key](const InferenceSessionWrapper& session) {
        const auto& graph = session.GetGraph();
        const auto op_counts = CountOpsInGraph(graph);
        ASSERT_EQ(OpCount(op_counts, dequantize_linear_key), expected_dq_count);
      };

      TransformerTester(
          graph_builder_fn,
          post_transform_check_fn,
          TransformerLevel::Default,
          TransformerLevel::Level1,
          opset_version);
    }
  };

  run_tests(false, false);
#if !defined(DISABLE_CONTRIB_OPS)
  run_tests(true, false);  // Use contrib QDQ ops.
  run_tests(true, true);   // Use 16-bit contrib QDQ ops.
#endif
}

}  // namespace

TEST(EnsureUniqueDQForNodeUnitTests, DQSharedAmongNodes) {
  GraphConfig config{};
  config.num_explicit_consumer_nodes = 3;
  config.num_inputs_per_explicit_consumer_node = 1;

  // expected count = one for each explicit consumer node (3), reusing the original one = 3
  RunEnsureUniqueDQForNodeUnitTest(config, 3);
}

TEST(EnsureUniqueDQForNodeUnitTests, DQSharedAmongNodesWithGraphOutput) {
  GraphConfig config{};
  config.num_explicit_consumer_nodes = 3;
  config.num_inputs_per_explicit_consumer_node = 1;
  config.has_graph_output = true;

  // expected count = preserved original (1) + one for each explicit consumer node (3) = 4
  RunEnsureUniqueDQForNodeUnitTest(config, 4);
}

TEST(EnsureUniqueDQForNodeUnitTests, DQSharedAmongNodesWithSubgraphConsumer) {
  GraphConfig config{};
  config.num_explicit_consumer_nodes = 3;
  config.num_inputs_per_explicit_consumer_node = 1;
  config.has_subgraph_consumer = true;

  // expected count = preserved original (1) + one for each explicit consumer node (3) = 4
  RunEnsureUniqueDQForNodeUnitTest(config, 4);
}

TEST(EnsureUniqueDQForNodeUnitTests, DQSharedAmongNodesWithSubgraphConsumerAndGraphOutput) {
  GraphConfig config{};
  config.num_explicit_consumer_nodes = 3;
  config.num_inputs_per_explicit_consumer_node = 1;
  config.has_graph_output = true;
  config.has_subgraph_consumer = true;

  // expected count = preserved original (1) + one for each explicit consumer node (3) = 4
  RunEnsureUniqueDQForNodeUnitTest(config, 4);
}

TEST(EnsureUniqueDQForNodeUnitTests, DQSharedAmongNodeInputs) {
  GraphConfig config{};
  config.num_explicit_consumer_nodes = 2;
  config.num_inputs_per_explicit_consumer_node = 5;

  // expected count = one for each explicit consumer node input (2 * 5), reusing the original one = 10
  RunEnsureUniqueDQForNodeUnitTest(config, 10);
}

TEST(EnsureUniqueDQForNodeUnitTests, DQSharedAmongNodeInputsWithGraphOutput) {
  GraphConfig config{};
  config.num_explicit_consumer_nodes = 2;
  config.num_inputs_per_explicit_consumer_node = 5;
  config.has_graph_output = true;

  // expected count = preserved original (1) + one for each explicit consumer node input (2 * 5) = 11
  RunEnsureUniqueDQForNodeUnitTest(config, 11);
}

TEST(EnsureUniqueDQForNodeUnitTests, DQSharedAmongNodeInputsWithSubgraphConsumer) {
  GraphConfig config{};
  config.num_explicit_consumer_nodes = 2;
  config.num_inputs_per_explicit_consumer_node = 5;
  config.has_subgraph_consumer = true;

  // expected count = preserved original (1) + one for each explicit consumer node input (2 * 5) = 11
  RunEnsureUniqueDQForNodeUnitTest(config, 11);
}

TEST(EnsureUniqueDQForNodeUnitTests, DQSharedAmongNodeInputsWithSubgraphConsumerAndGraphOutput) {
  GraphConfig config{};
  config.num_explicit_consumer_nodes = 2;
  config.num_inputs_per_explicit_consumer_node = 5;
  config.has_graph_output = true;
  config.has_subgraph_consumer = true;

  // expected count = preserved original (1) + one for each explicit consumer node input (2 * 5) = 11
  RunEnsureUniqueDQForNodeUnitTest(config, 11);
}

TEST(EnsureUniqueDQForNodeUnitTests, QDQWithMultiConsumerDQNodes) {
  constexpr auto model_uri = ORT_TSTR("testdata/qdq_with_multi_consumer_dq_nodes.onnx");

  SessionOptions session_options{};
  // test interaction with level 1 transformers
  session_options.graph_optimization_level = TransformerLevel::Level1;

  InferenceSessionWrapper session{session_options, GetEnvironment()};

  ASSERT_STATUS_OK(session.Load(model_uri));

  const auto op_count_before = CountOpsInGraph(session.GetGraph());

  ASSERT_STATUS_OK(session.Initialize());

  const auto op_count_after = CountOpsInGraph(session.GetGraph());

  // there are 3 DQ nodes with 2 consumers (an earlier Conv and later Add)
  // additionally the last one also provides a graph output
  // based on that there should be 3 new DQ nodes for the internal consumers and 1 new one for the graph output
  EXPECT_EQ(OpCount(op_count_before, "DequantizeLinear") + 4, OpCount(op_count_after, "DequantizeLinear"));
}

TEST(EnsureUniqueDQForNodeUnitTests, QDQWithMultiConsumerDQNodesPreservingAttributes) {
  constexpr auto model_uri = ORT_TSTR("testdata/qdq_with_multi_consumer_q_dq_axis.onnx");

  SessionOptions session_options{};
  // test interaction with level 1 transformers
  session_options.graph_optimization_level = TransformerLevel::Level1;

  InferenceSessionWrapper session{session_options, GetEnvironment()};

  ASSERT_STATUS_OK(session.Load(model_uri));

  const auto op_count_before = CountOpsInGraph(session.GetGraph());

  ASSERT_STATUS_OK(session.Initialize());

  const auto op_count_after = CountOpsInGraph(session.GetGraph());

  EXPECT_EQ(OpCount(op_count_before, "DequantizeLinear") + 8, OpCount(op_count_after, "DequantizeLinear"));

  int64_t given_axis = 0;  // all the following 4 DQ nodes and their duplicated one should have axis = 0
  std::string axis_dq_name0 = "Convolution28_Output_0/fusedmuladd_B/DequantizeLinear";
  std::string axis_dq_name1 = "Parameter5/DequantizeLinear";
  std::string axis_dq_name2 = "Convolution110_Output_0/fusedmuladd_B/DequantizeLinear";
  std::string axis_dq_name3 = "Parameter87/DequantizeLinear";
  for (auto& node : session.GetGraph().Nodes()) {
    if (node.OpType() == "DequantizeLinear") {
      if (node.Name().find(axis_dq_name0) != std::string::npos ||
        node.Name().find(axis_dq_name1) != std::string::npos ||
        node.Name().find(axis_dq_name2) != std::string::npos ||
        node.Name().find(axis_dq_name3) != std::string::npos)
      {
        auto attrs = node.GetAttributes();
        EXPECT_TRUE(attrs.find("axis") != attrs.end());
        auto& axis_attr = attrs.at("axis");
        int64_t axis = axis_attr.i();
        EXPECT_EQ(axis, given_axis);
      }
    }
  }
}

}  // namespace onnxruntime::test
