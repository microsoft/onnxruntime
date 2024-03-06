// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/framework/compute_capability.h"
#include "core/providers/partitioning_utils.h"
#include "core/framework/node_unit.h"
#include "test/util/include/asserts.h"
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
  std::tie(node_unit_holder, node_unit_map) = GetAllNodeUnits(graph_viewer);

  auto result = utils::CreateSupportedPartitions(graph_viewer, is_node_supported, on_group_closed,
                                                 gen_metadef_name, "TEST", kCpuExecutionProvider, &node_unit_map);

  // we should have 2 supported partitions, split by the Cast node.
  // the first should have the Mul and NOT the DQ for the initializer if everything worked correctly.
  ASSERT_EQ(result.size(), size_t(2)) << "Expected 2 partitions";
  ASSERT_EQ(result[0]->sub_graph->nodes.size(), size_t(1)) << "First partition should only have the Mul and not a DQ";
  ASSERT_EQ(result[1]->sub_graph->nodes.size(), size_t(5));  // everything else except the unsupported Cast
}
}  // namespace test
}  // namespace onnxruntime
