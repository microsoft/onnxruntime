// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/framework/node_unit.h"
#include "core/framework/compute_capability.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/partitioning_utils.h"

#include "test/optimizer/graph_transform_test_builder.h"
#include "test/optimizer/qdq_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_utils.h"
#include "test/util/include/test/test_environment.h"

namespace onnxruntime {
namespace test {

// Test handling of a DQ node that is connected to an initializer at the start of the graph, but not used
// in a QDQ node group until after an unsupported node in the graph. If we do not process QDQ node units
// correctly this DQ will incorrectly be in the first partition, with the rest of the QDQ node group in
// the second partition.
TEST(PartitioningUtilsTest, TestQDQHandling) {
  constexpr const ORTCHAR_T* model_uri = ORT_TSTR("testdata/ort_github_issue_19590.onnx");
  auto& logger = DefaultLoggingManager().DefaultLogger();

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logger));
  Graph& graph = p_model->MainGraph();
  GraphViewer graph_viewer = GraphViewer(graph);

  // we want everything but the Cast in the test model to be supported
  const auto is_node_supported = [&](const Node& node) -> bool {
    return node.OpType() != "Cast";
  };

  const auto on_group_closed = [&](const std::vector<const Node*>& /*group*/) -> bool {
    return true;
  };

  const auto gen_metadef_name = [&]() {
    static int metadef_id = 0;
    return "TestMetaDef_" + std::to_string(metadef_id++);
  };

  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer);

  auto result = utils::CreateSupportedPartitions(graph_viewer, is_node_supported, on_group_closed,
                                                 gen_metadef_name, "TEST", kCpuExecutionProvider, &node_unit_map,
                                                 true);

  // we should have 2 supported partitions, split by the Cast node.
  // the first should have the Mul and NOT the DQ for the initializer if everything worked correctly.
  ASSERT_EQ(result.size(), size_t(2)) << "Expected 2 partitions";
  ASSERT_EQ(result[0]->sub_graph->nodes.size(), size_t(1)) << "First partition should only have the Mul and not a DQ";
  ASSERT_EQ(result[1]->sub_graph->nodes.size(), size_t(5));  // everything else except the unsupported Cast
}

/// Check that CreateSupportedPartitions processes all nodes without error.
static void CheckAllNodesProcessed(const std::function<void(ModelTestBuilder&)>& build_model) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  const std::unordered_map<std::string, int> domain_to_version = {{"", 15}};

  Model model("PartitioningUtils_TestModel", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, logger);

  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  build_model(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  GraphViewer graph_viewer = GraphViewer(graph);

  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer);

  const auto is_node_supported = [&](const Node& /*node*/) -> bool {
    return true;
  };

  const auto on_group_closed = [&](const std::vector<const Node*>& /*group*/) -> bool {
    return true;
  };

  const auto gen_metadef_name = [&]() {
    static int metadef_id = 0;
    return "TestMetaDef_" + std::to_string(metadef_id++);
  };

  auto result = utils::CreateSupportedPartitions(graph_viewer, is_node_supported, on_group_closed,
                                                 gen_metadef_name, "TEST", kCpuExecutionProvider, &node_unit_map,
                                                 true);

  // the 'real' test is that CreateSupportedPartitions doesn't throw due to a mismatch with expected vs processed nodes
  // as all ops are supported there should only ever be 1 partition
  ASSERT_EQ(result.size(), size_t(1)) << "Expected 1 partition";
}

TEST(PartitioningUtilsTest, TestHandlingQDQNodeUnitWithNoQNodes) {
  // build graph with QDQ node unit for logical operator (Equal) that has no Q node and a downstream node (Cast).
  auto build_model = [](ModelTestBuilder& builder) {
    constexpr uint8_t zero_point = 0;
    constexpr float qdq_scale = 0.0038f;
    const std::vector<int64_t> input_shape = {1, 3, 8, 8};

    auto* input0 = builder.MakeInput<float>(input_shape, -1.0f, 1.0f);
    auto* input1 = builder.MakeInput<float>(input_shape, -1.0f, 1.0f);
    auto* output = builder.MakeOutput();

    // input -> Q -> DQ -> Op
    auto* qdq0_output = AddQDQNodePair<uint8_t>(builder, input0, qdq_scale, zero_point);
    auto* qdq1_output = AddQDQNodePair<uint8_t>(builder, input1, qdq_scale, zero_point);

    // Equal ->
    auto* equal_output = builder.MakeIntermediate();
    builder.AddNode("Equal", {qdq0_output, qdq1_output}, {equal_output});

    // -> Cast -> output
    Node& cast_node = builder.AddNode("Cast", {equal_output}, {output});
    cast_node.AddAttribute("to",
                           static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  };

  CheckAllNodesProcessed(build_model);
}

// TopK produces 2 outputs, one of which is used in a QDQ node group (Q of values output)
// and the other (indices output) is not. A downstream node consuming the indices output has an edge from the target
// node and not a Q node.
// To process this correctly, the QDQ NodeUnit must return output edges for both the Q node/s of the values output,
// and the downstream node (Cast in this case) of the indices output.
TEST(PartitioningUtilsTest, TestQDQNodeGroupWithOutputFromTargetNode) {
  const auto build_model = [](ModelTestBuilder& builder) {
    constexpr uint8_t zero_point = 0;
    constexpr float qdq_scale = 0.0038f;
    const std::vector<int64_t> input_shape = {1, 3, 8, 8};

    auto* input0 = builder.MakeInput<float>(input_shape, -1.0f, 1.0f);

    // input -> Q -> DQ ->
    auto* qdq0_output = AddQDQNodePair<uint8_t>(builder, input0, qdq_scale, zero_point);

    // K input
    NodeArg* k_input = builder.MakeInput<int64_t>({1}, {10});

    // TopK op
    NodeArg* values_output = builder.MakeIntermediate();
    NodeArg* indices_output = builder.MakeIntermediate();
    builder.AddNode("TopK", {qdq0_output, k_input}, {values_output, indices_output});

    // values -> Q -> DQ -> graph output
    AddQDQNodePairWithOutputAsGraphOutput<uint8_t>(builder, values_output, qdq_scale, zero_point);

    // indices -> Cast -> graph output
    auto* i_output = builder.MakeOutput();
    Node& cast_node = builder.AddNode("Cast", {indices_output}, {i_output});
    const auto dst_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32;
    cast_node.AddAttribute("to", static_cast<int64_t>(dst_type));
  };

  CheckAllNodesProcessed(build_model);
}
}  // namespace test
}  // namespace onnxruntime
