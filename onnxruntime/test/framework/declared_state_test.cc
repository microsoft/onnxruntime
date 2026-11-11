// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/declared_state.h"

#include <limits>
#include <memory>
#include <utility>

#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime::test {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kDeclaration = R"({
  "version": 1,
  "max_batch_size": 2,
  "max_sequence_length": 64,
  "max_live_instances": 1,
  "states": [
    {"id": "key", "kind": "kv_cache", "input": "past.{layer}.key", "output": "present.{layer}.key",
     "layers": [0, 1], "capacity": 64, "alias": "past_present",
     "lifetime": "per_generator", "location": "follow_consumer"},
    {"id": "value", "kind": "kv_cache", "input": "past.{layer}.value", "output": "present.{layer}.value",
     "layers": [0, 1], "capacity": 64, "alias": "past_present",
     "lifetime": "per_generator", "location": "follow_consumer"}
  ]
})";

TEST(DeclaredStateParserTest, RejectsMalformedUnknownAndAmbiguousDeclarationsAtomically) {
  const Json valid = Json::parse(kDeclaration);
  InlinedVector<std::string> invalid{"", "{", "[]", "null"};
  const auto add = [&](Json json) { invalid.push_back(json.dump()); };
  Json json = valid;
  json["max_batch_size"] = 0;
  add(json);
  json = valid;
  json["max_sequence_length"] = 1.5;
  add(json);
  json["max_sequence_length"] = std::numeric_limits<uint64_t>::max();
  add(json);
  json = valid;
  json["max_live_instances"] = -1;
  add(json);
  json = valid;
  json["typo"] = 1;
  add(json);
  json = valid;
  json.erase("max_sequence_length");
  add(json);
  json = valid;
  json["states"][0]["capacity"] = true;
  add(json);
  json = valid;
  json["states"][0]["location"] = nullptr;
  add(json);
  json = valid;
  json["states"][0]["layers"] = {0, 0};
  add(json);
  json = valid;
  json["states"][0]["input"] = "past.{L}.key";
  add(json);
  json = valid;
  json["states"][0]["input"] = "past.{layer}.{layer}";
  add(json);
  json = valid;
  json["states"][0]["output"] = "past.{layer}.key";
  add(json);
  json = valid;
  json["states"][1]["id"] = "key";
  add(json);
  json = valid;
  json["states"][0]["alias"] = "share";
  add(json);
  json = valid;
  json["states"][0]["window"] = -1;
  add(json);
  invalid.push_back(std::string(kDeclaration).replace(kDeclaration.find("\"version\": 1"), std::string_view("\"version\": 1").size(), "\"version\": 1, \"version\": 1"));
  DeclaredStateContract previous;
  previous.max_batch_size = 17;
  for (const auto& text : invalid) {
    const auto status = ParseDeclaredStateContract(text, previous);
    EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT) << text;
    EXPECT_EQ(previous.max_batch_size, 17);
    EXPECT_TRUE(previous.states.empty());
  }
}

TEST(DeclaredStateParserTest, UnsupportedVersionKindAndLocationAreNotSilentlyIgnored) {
  const Json valid = Json::parse(kDeclaration);
  for (const auto& field : {"version", "kind", "location"}) {
    Json json = valid;
    if (std::string_view(field) == "version") {
      json[field] = 2;
    } else {
      json["states"][0][field] = "unsupported";
    }
    DeclaredStateContract contract;
    EXPECT_EQ(ParseDeclaredStateContract(json.dump(), contract).Code(), common::NOT_IMPLEMENTED);
  }
}

#if !defined(DISABLE_CONTRIB_OPS)

ONNX_NAMESPACE::TypeProto TensorType(int32_t element_type, std::initializer_list<int64_t> dimensions) {
  ONNX_NAMESPACE::TypeProto type;
  auto* tensor = type.mutable_tensor_type();
  tensor->set_elem_type(element_type);
  auto* shape = tensor->mutable_shape();
  for (int64_t dimension : dimensions) {
    auto* dim = shape->add_dim();
    if (dimension >= 0) {
      dim->set_dim_value(dimension);
    }
  }
  return type;
}

class DeclaredStateTest : public ::testing::Test {
 protected:
  DeclaredStateTest()
      : model_("declared_state", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
               {{kOnnxDomain, 18}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger()) {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DeclaredStateTest);

  Graph& GraphRef() { return model_.MainGraph(); }

  Node& Gqa(int64_t layer, int64_t window = -1, bool sliding = false, int64_t capacity = -1,
            int32_t element_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    auto& graph = GraphRef();
    const auto query_type = TensorType(element_type, {2, 1, 16});
    const auto kv_type = TensorType(element_type, {2, 1, 8});
    const auto cache_type = TensorType(element_type, {2, 2, capacity, 4});
    const auto lengths_type = TensorType(ONNX_NAMESPACE::TensorProto_DataType_INT32, {2});
    const auto total_type = TensorType(ONNX_NAMESPACE::TensorProto_DataType_INT32, {});
    const std::string suffix = std::to_string(layer);
    auto& query = graph.GetOrCreateNodeArg("query." + suffix, &query_type);
    auto& key = graph.GetOrCreateNodeArg("key." + suffix, &kv_type);
    auto& value = graph.GetOrCreateNodeArg("value." + suffix, &kv_type);
    auto& past_key = graph.GetOrCreateNodeArg("past." + suffix + ".key", &cache_type);
    auto& past_value = graph.GetOrCreateNodeArg("past." + suffix + ".value", &cache_type);
    auto& lengths = graph.GetOrCreateNodeArg("lengths." + suffix, &lengths_type);
    auto& total = graph.GetOrCreateNodeArg("total." + suffix, &total_type);
    auto& output = graph.GetOrCreateNodeArg("output." + suffix, &query_type);
    auto& present_key = graph.GetOrCreateNodeArg("present." + suffix + ".key", &cache_type);
    auto& present_value = graph.GetOrCreateNodeArg("present." + suffix + ".value", &cache_type);
    auto& node = graph.AddNode("GQA." + suffix, "GroupQueryAttention", "",
                               {&query, &key, &value, &past_key, &past_value, &lengths, &total},
                               {&output, &present_key, &present_value}, nullptr, kMSDomain);
    node.AddAttribute("num_heads", int64_t{4});
    node.AddAttribute("kv_num_heads", int64_t{2});
    node.AddAttribute("local_window_size", window);
    node.AddAttribute("sliding_window_cache", int64_t{sliding ? 1 : 0});
    return node;
  }

  DeclaredStateContract Contract(int64_t layers = 2) {
    DeclaredStateContract contract;
    ORT_THROW_IF_ERROR(ParseDeclaredStateContract(kDeclaration, contract));
    if (layers == 1) {
      for (auto& state : contract.states) {
        state.layers = {0};
      }
    }
    return contract;
  }

  CapabilityPlanningQuery Query() {
    CapabilityPlanningQuery query;
    query.mode = CapabilityQueryMode::Planning;
    query.query_id = 1;
    query.graph_id = "decoder";
    query.graph_revision = 7;
    return query;
  }

  Model model_;
};

TEST_F(DeclaredStateTest, ResolvesTemplatesAndGraphTypesWithoutChangingPlacementOrShapes) {
  auto& cpu = Gqa(0);
  auto& cuda = Gqa(1);
  cpu.SetExecutionProviderType(kCpuExecutionProvider);
  cuda.SetExecutionProviderType(kCudaExecutionProvider);
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  const auto contract = Contract();
  ResolvedDeclaredStateContract resolved;
  ASSERT_STATUS_OK(ResolveDeclaredStateContract(contract, graph, "decoder", 7, resolved));
  ASSERT_EQ(resolved.states.size(), 4u);
  const auto& key = resolved.states.front();
  EXPECT_EQ(key.input, "past.0.key");
  EXPECT_EQ(key.output, "present.0.key");
  EXPECT_EQ(key.consumer_node_index, cpu.Index());
  EXPECT_EQ(key.input_index, 3u);
  EXPECT_EQ(key.output_index, 1u);
  EXPECT_EQ(key.element_type, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  EXPECT_EQ(key.max_shape, (TensorShapeVector{2, 2, 64, 4}));
  EXPECT_EQ(key.size_bound_bytes, 4096u);
  EXPECT_EQ(key.alias, CapabilityInputOutputAlias::Identical);
  EXPECT_EQ(cpu.GetExecutionProviderType(), kCpuExecutionProvider);
  EXPECT_EQ(cuda.GetExecutionProviderType(), kCudaExecutionProvider);
  EXPECT_FALSE(graph.GetNodeArg(key.input)->Shape()->dim(2).has_dim_value());
}

TEST_F(DeclaredStateTest, RejectsUndeclaredAndMismatchedStateWithoutReplacingPreviousResolution) {
  Gqa(0);
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  auto contract = Contract(1);
  const auto value = contract.states.back();
  contract.states.pop_back();
  ResolvedDeclaredStateContract resolved;
  resolved.graph_id = "previous";
  EXPECT_EQ(ResolveDeclaredStateContract(contract, graph, "decoder", 7, resolved).Code(), common::INVALID_ARGUMENT);
  EXPECT_EQ(resolved.graph_id, "previous");
  contract.states.push_back(value);
  std::swap(contract.states[0].output, contract.states[1].output);
  EXPECT_EQ(ResolveDeclaredStateContract(contract, graph, "decoder", 7, resolved).Code(), common::INVALID_ARGUMENT);
  EXPECT_EQ(resolved.graph_id, "previous");
}

TEST_F(DeclaredStateTest, LiteralPairsSupportSessionLifetimeAndDisjointFP16Storage) {
  Gqa(0, -1, false, -1, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  ASSERT_STATUS_OK(GraphRef().Resolve());
  Json json = Json::parse(kDeclaration);
  for (auto& state : json["states"]) {
    const std::string suffix = state["id"].get<std::string>();
    state["input"] = "past.0." + suffix;
    state["output"] = "present.0." + suffix;
    state["layers"] = {0};
    state["alias"] = "none";
    state["lifetime"] = "per_session";
  }
  DeclaredStateContract contract;
  ASSERT_STATUS_OK(ParseDeclaredStateContract(json.dump(), contract));
  ResolvedDeclaredStateContract resolved;
  ASSERT_STATUS_OK(ResolveDeclaredStateContract(contract, GraphViewer(GraphRef()), "decoder", 7, resolved));
  ASSERT_EQ(resolved.states.size(), 2u);
  EXPECT_EQ(resolved.states.front().element_type, ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  EXPECT_EQ(resolved.states.front().size_bound_bytes, 2048u);
  EXPECT_EQ(resolved.states.front().alias, CapabilityInputOutputAlias::Disjoint);
  EXPECT_EQ(resolved.states.front().lifetime, DeclaredStateLifetime::Session);
}

TEST_F(DeclaredStateTest, RejectsStateWithAnotherConsumerInsteadOfGuessingColocation) {
  Gqa(0);
  auto& graph = GraphRef();
  auto* past = graph.GetNodeArg("past.0.key");
  auto& extra = graph.GetOrCreateNodeArg("extra", past->TypeAsProto());
  graph.AddNode("extra_consumer", "Identity", "", {past}, {&extra});
  ASSERT_STATUS_OK(graph.Resolve());
  ResolvedDeclaredStateContract resolved;
  EXPECT_EQ(ResolveDeclaredStateContract(Contract(1), GraphViewer(graph), "decoder", 7, resolved).Code(),
            common::NOT_IMPLEMENTED);
}

TEST_F(DeclaredStateTest, WindowedCapacityIsIndependentOfAbsoluteSequenceBound) {
  Gqa(0, 8, true);
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  auto contract = Contract(1);
  for (auto& state : contract.states) {
    state.capacity = 8;
    state.window = 8;
  }
  ResolvedDeclaredStateContract resolved;
  ASSERT_STATUS_OK(ResolveDeclaredStateContract(contract, graph, "decoder", 7, resolved));
  EXPECT_EQ(resolved.max_sequence_length, 64);
  EXPECT_EQ(resolved.states.front().max_shape[2], 8);
  EXPECT_EQ(resolved.states.front().window, 8);
  EXPECT_TRUE(resolved.states.front().sliding_window_cache);
  contract.states.front().capacity = 7;
  EXPECT_EQ(ResolveDeclaredStateContract(contract, graph, "decoder", 7, resolved).Code(), common::INVALID_ARGUMENT);
  contract.states.front().capacity = 8;
  contract.states.front().window = 9;
  EXPECT_EQ(ResolveDeclaredStateContract(contract, graph, "decoder", 7, resolved).Code(), common::INVALID_ARGUMENT);
}

TEST_F(DeclaredStateTest, LocalAttentionWithoutEvictionStillRequiresFullCapacity) {
  Gqa(0, 8, false);
  ASSERT_STATUS_OK(GraphRef().Resolve());
  auto contract = Contract(1);
  contract.states.front().capacity = 8;
  ResolvedDeclaredStateContract resolved;
  EXPECT_EQ(ResolveDeclaredStateContract(contract, GraphViewer(GraphRef()), "decoder", 7, resolved).Code(),
            common::INVALID_ARGUMENT);
}

TEST_F(DeclaredStateTest, RejectsIncompatibleFixedAliasExtentAndOverflowingByteBounds) {
  Gqa(0, -1, false, 64);
  ASSERT_STATUS_OK(GraphRef().Resolve());
  const GraphViewer graph(GraphRef());
  auto contract = Contract(1);
  ResolvedDeclaredStateContract resolved;
  contract.states.front().capacity = 128;
  EXPECT_EQ(ResolveDeclaredStateContract(contract, graph, "decoder", 7, resolved).Code(), common::INVALID_ARGUMENT);
  contract.states.front().capacity = 64;
  contract.max_batch_size = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(ResolveDeclaredStateContract(contract, graph, "decoder", 7, resolved).Code(), common::INVALID_ARGUMENT);
}

TEST_F(DeclaredStateTest, ProjectionPreservesPolicyPinsAndKeepsOwnerMetadataOutOfQuery) {
  const auto node = Gqa(0).Index();
  ASSERT_STATUS_OK(GraphRef().Resolve());
  auto contract = Contract(1);
  contract.max_live_instances = 3;
  contract.states.back().alias = CapabilityInputOutputAlias::Disjoint;
  ResolvedDeclaredStateContract resolved;
  ASSERT_STATUS_OK(ResolveDeclaredStateContract(contract, GraphViewer(GraphRef()), "decoder", 7, resolved));
  auto query = Query();
  query.node_constraints = {{node, "CPU:0", InlinedVector<KernelDispatchPolicy>{KernelDispatchPolicy::Memory}}};
  query.node_constraints.front().input_output_aliases = {{3, 1, CapabilityInputOutputAlias::Unknown}};
  ASSERT_STATUS_OK(ProjectDeclaredStateToPlanningQuery(resolved, query));
  ASSERT_EQ(query.max_input_shapes.size(), 2u);
  EXPECT_EQ(query.max_input_shapes.at("past.0.key"), TensorShape({2, 2, 64, 4}));
  ASSERT_EQ(query.node_constraints.size(), 1u);
  const auto& constraint = query.node_constraints.front();
  EXPECT_EQ(constraint.required_ep_instance_id, "CPU:0");
  EXPECT_EQ(constraint.permitted_policies,
            (std::optional<InlinedVector<KernelDispatchPolicy>>{{KernelDispatchPolicy::Memory}}));
  ASSERT_EQ(constraint.input_output_aliases.size(), 2u);
  EXPECT_EQ(constraint.input_output_aliases[0].alias, CapabilityInputOutputAlias::Identical);
  EXPECT_EQ(constraint.input_output_aliases[1].alias, CapabilityInputOutputAlias::Disjoint);
  EXPECT_EQ(query.workload_bounds.size(), 2u);
  EXPECT_EQ(resolved.max_live_instances, 3);
}

TEST_F(DeclaredStateTest, ProjectionRejectsConflictingCapacityAliasBoundsAndStaleGraphAtomically) {
  const auto node = Gqa(0).Index();
  ASSERT_STATUS_OK(GraphRef().Resolve());
  ResolvedDeclaredStateContract resolved;
  ASSERT_STATUS_OK(ResolveDeclaredStateContract(Contract(1), GraphViewer(GraphRef()), "decoder", 7, resolved));
  auto query = Query();
  query.max_input_shapes.emplace("past.0.key", TensorShape({2, 2, 32, 4}));
  EXPECT_EQ(ProjectDeclaredStateToPlanningQuery(resolved, query).Code(), common::INVALID_ARGUMENT);
  EXPECT_TRUE(query.workload_bounds.empty());
  EXPECT_TRUE(query.node_constraints.empty());
  EXPECT_EQ(query.max_input_shapes.size(), 1u);

  query = Query();
  CapabilityNodeConstraint constraint;
  constraint.node_index = node;
  constraint.input_output_aliases = {{3, 1, CapabilityInputOutputAlias::Disjoint}};
  query.node_constraints.push_back(constraint);
  EXPECT_EQ(ProjectDeclaredStateToPlanningQuery(resolved, query).Code(), common::INVALID_ARGUMENT);
  EXPECT_TRUE(query.max_input_shapes.empty());
  EXPECT_TRUE(query.workload_bounds.empty());
  EXPECT_EQ(query.node_constraints.front().input_output_aliases.front().alias, CapabilityInputOutputAlias::Disjoint);

  query = Query();
  query.workload_bounds = {{"declared_state.total_sequence_length", 0, 32}};
  EXPECT_EQ(ProjectDeclaredStateToPlanningQuery(resolved, query).Code(), common::INVALID_ARGUMENT);
  EXPECT_EQ(query.workload_bounds.size(), 1u);
  query = Query();
  ++query.graph_revision;
  EXPECT_EQ(ProjectDeclaredStateToPlanningQuery(resolved, query).Code(), common::INVALID_ARGUMENT);
  EXPECT_TRUE(query.max_input_shapes.empty());
  query = Query();
  resolved.max_batch_size = 0;
  EXPECT_EQ(ProjectDeclaredStateToPlanningQuery(resolved, query).Code(), common::INVALID_ARGUMENT);
  EXPECT_TRUE(query.max_input_shapes.empty());
  EXPECT_TRUE(query.workload_bounds.empty());
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace
}  // namespace onnxruntime::test
