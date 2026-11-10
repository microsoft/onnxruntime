// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/region_discovery.h"

#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "core/graph/graph_viewer.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {
namespace {

class RegionDiscoveryTest : public ::testing::Test {
 protected:
  RegionDiscoveryTest()
      : model_("region_discovery", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
               {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger()) {
    type_.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type_.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(4);
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RegionDiscoveryTest);

  Graph& GraphRef() { return model_.MainGraph(); }

  Node& Identity(std::string_view input, std::string_view output, std::string annotation = {}) {
    auto& graph = GraphRef();
    auto& input_arg = graph.GetOrCreateNodeArg(std::string(input), &type_);
    auto& output_arg = graph.GetOrCreateNodeArg(std::string(output), &type_);
    auto& node = graph.AddNode(std::string(output), "Identity", "", {&input_arg}, {&output_arg});
    node.SetLayeringAnnotation(std::move(annotation));
    return node;
  }

  Node& Join(std::string_view first, std::string_view second, std::string_view output) {
    auto& graph = GraphRef();
    auto& first_arg = graph.GetOrCreateNodeArg(std::string(first), &type_);
    auto& second_arg = graph.GetOrCreateNodeArg(std::string(second), &type_);
    auto& output_arg = graph.GetOrCreateNodeArg(std::string(output), &type_);
    return graph.AddNode(std::string(output), "Add", "", {&first_arg, &second_arg}, {&output_arg});
  }

  Model model_;
  ONNX_NAMESPACE::TypeProto type_;
};

TEST_F(RegionDiscoveryTest, ExactAnnotationsFollowDependenciesNotNamesOrPlacement) {
  auto& first = Identity("X", "A", "block.1");
  auto& second = Identity("A", "B", "block.2");
  auto& last = Identity("B", "Y", "block.10");
  first.SetExecutionProviderType(kCudaExecutionProvider);
  second.SetExecutionProviderType(kCpuExecutionProvider);
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  GraphRegionDiscoveryResult result;
  ASSERT_STATUS_OK(DiscoverGraphRegionsFromAnnotations(graph, result));
  ASSERT_EQ(result.regions.size(), 3u);
  EXPECT_EQ(result.regions[0].id, "block.1");
  EXPECT_EQ(result.regions[1].id, "block.2");
  EXPECT_EQ(result.regions[2].id, "block.10");
  EXPECT_EQ(result.regions[0].nodes, InlinedVector<NodeIndex>{first.Index()});
  EXPECT_EQ(result.regions[2].nodes, InlinedVector<NodeIndex>{last.Index()});
  EXPECT_TRUE(result.ungrouped_nodes.empty());
  EXPECT_EQ(first.GetExecutionProviderType(), kCudaExecutionProvider);
  EXPECT_EQ(second.GetExecutionProviderType(), kCpuExecutionProvider);
  EXPECT_TRUE(last.GetExecutionProviderType().empty());
  EXPECT_EQ(first.GetLayeringAnnotation(), "block.1");
  EXPECT_EQ(graph.NumberOfNodes(), 3);
}

TEST_F(RegionDiscoveryTest, ExplicitMembershipIsNormalizedWithoutChangingIdentity) {
  const auto& first = Identity("X", "A");
  const auto& second = Identity("A", "B");
  const auto& third = Identity("B", "C");
  const auto& last = Identity("C", "Y");
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  const InlinedVector<GraphRegion> requested{
      {"block.10", {last.Index(), third.Index()}}, {"block.2", {second.Index(), first.Index()}}};
  GraphRegionDiscoveryResult result;
  ASSERT_STATUS_OK(DiscoverGraphRegions(graph, requested, result));
  ASSERT_EQ(result.regions.size(), 2u);
  EXPECT_EQ(result.regions[0].id, "block.2");
  EXPECT_EQ(result.regions[1].id, "block.10");
  EXPECT_EQ(result.regions[0].nodes, (InlinedVector<NodeIndex>{first.Index(), second.Index()}));
  EXPECT_EQ(result.regions[1].nodes, (InlinedVector<NodeIndex>{third.Index(), last.Index()}));
  EXPECT_EQ(requested[0].nodes.front(), last.Index());
}

TEST_F(RegionDiscoveryTest, UngroupedNodesPreservePathsAndResidualDependencies) {
  const auto& embedding = Identity("X", "E");
  Identity("E", "A", "block.2");
  const auto& bridge = Identity("A", "B");
  Identity("B", "C", "block.10");
  const auto& head = Join("A", "C", "Y");
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  GraphRegionDiscoveryResult result;
  ASSERT_STATUS_OK(DiscoverGraphRegionsFromAnnotations(graph, result));
  ASSERT_EQ(result.regions.size(), 2u);
  EXPECT_EQ(result.regions[0].id, "block.2");
  EXPECT_EQ(result.regions[1].id, "block.10");
  EXPECT_EQ(result.ungrouped_nodes, (InlinedVector<NodeIndex>{embedding.Index(), bridge.Index(), head.Index()}));
}

TEST_F(RegionDiscoveryTest, MissingAnnotationsLeaveAllNodesExplicitlyUngrouped) {
  const auto& first = Identity("X", "A");
  const auto& last = Identity("A", "Y");
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  GraphRegionDiscoveryResult result;
  ASSERT_STATUS_OK(DiscoverGraphRegionsFromAnnotations(graph, result));
  EXPECT_TRUE(result.regions.empty());
  EXPECT_EQ(result.ungrouped_nodes, (InlinedVector<NodeIndex>{first.Index(), last.Index()}));
}

TEST_F(RegionDiscoveryTest, InvalidMembershipPreservesPreviousResult) {
  const NodeIndex removed = Identity("unused", "removed").Index();
  const auto& first = Identity("X", "A");
  const auto& last = Identity("A", "Y");
  ASSERT_TRUE(GraphRef().RemoveNode(removed));
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  const InlinedVector<InlinedVector<GraphRegion>> invalid{
      {{"", {first.Index()}}},
      {{"empty", {}}},
      {{"duplicate", {first.Index()}}, {"duplicate", {last.Index()}}},
      {{"repeated", {first.Index(), first.Index()}}},
      {{"first", {first.Index()}}, {"overlap", {first.Index(), last.Index()}}},
      {{"removed", {removed}}},
      {{"out_of_range", {std::numeric_limits<NodeIndex>::max()}}}};
  GraphRegionDiscoveryResult result{{{"previous", {first.Index()}}}, {last.Index()}};
  for (const auto& regions : invalid) {
    const auto status = DiscoverGraphRegions(graph, regions, result);
    EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT) << status.ErrorMessage();
    ASSERT_EQ(result.regions.size(), 1u);
    EXPECT_EQ(result.regions[0].id, "previous");
    EXPECT_EQ(result.regions[0].nodes, InlinedVector<NodeIndex>{first.Index()});
    EXPECT_EQ(result.ungrouped_nodes, InlinedVector<NodeIndex>{last.Index()});
  }
}

TEST_F(RegionDiscoveryTest, RejectsGroupingInducedCycle) {
  Identity("X", "A", "outer");
  Identity("A", "B", "middle");
  Identity("B", "Y", "outer");
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  GraphRegionDiscoveryResult result;
  const auto status = DiscoverGraphRegionsFromAnnotations(graph, result);
  EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT);
  EXPECT_NE(status.ErrorMessage().find("cycle"), std::string::npos);
}

TEST_F(RegionDiscoveryTest, RejectsCycleThroughUngroupedWork) {
  Identity("X", "A", "outer");
  Identity("A", "B");
  Identity("B", "Y", "outer");
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  GraphRegionDiscoveryResult result;
  const auto status = DiscoverGraphRegionsFromAnnotations(graph, result);
  EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT);
  EXPECT_NE(status.ErrorMessage().find("cycle"), std::string::npos);
}

TEST_F(RegionDiscoveryTest, RejectsAmbiguityInsteadOfUsingAnUngroupedSchedulingTieBreak) {
  Identity("X", "A", "block.2");
  Identity("Z", "B");
  Identity("B", "C", "block.10");
  Join("A", "C", "Y");
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  GraphRegionDiscoveryResult result;
  const auto status = DiscoverGraphRegionsFromAnnotations(graph, result);
  EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT);
  EXPECT_NE(status.ErrorMessage().find("Ambiguous"), std::string::npos);
}

TEST_F(RegionDiscoveryTest, IndependentUngroupedWorkDoesNotMakeRegionOrderAmbiguous) {
  Identity("X", "A", "block.2");
  const auto& independent = Identity("Z", "Y");
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  GraphRegionDiscoveryResult result;
  ASSERT_STATUS_OK(DiscoverGraphRegionsFromAnnotations(graph, result));
  ASSERT_EQ(result.regions.size(), 1u);
  EXPECT_EQ(result.regions[0].id, "block.2");
  EXPECT_EQ(result.ungrouped_nodes, InlinedVector<NodeIndex>{independent.Index()});
}

TEST_F(RegionDiscoveryTest, RejectsFilteredViewsRatherThanDroppingExternalDependencies) {
  const auto& first = Identity("X", "A", "block.2");
  Identity("A", "Y", "block.10");
  ASSERT_STATUS_OK(GraphRef().Resolve());
  IndexedSubGraph filter;
  filter.nodes = {first.Index()};
  auto meta = std::make_unique<IndexedSubGraph::MetaDef>();
  meta->name = "filtered_region";
  meta->domain = kMSDomain;
  meta->since_version = 1;
  meta->inputs = {"X"};
  meta->outputs = {"A"};
  filter.SetMetaDef(std::move(meta));
  const GraphViewer graph(GraphRef(), filter);
  GraphRegionDiscoveryResult result;
  const auto status = DiscoverGraphRegionsFromAnnotations(graph, result);
  EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT);
  EXPECT_NE(status.ErrorMessage().find("unfiltered"), std::string::npos);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
