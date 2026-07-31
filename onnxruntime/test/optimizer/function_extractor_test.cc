// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <cstring>

#include "core/common/logging/logging.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/function_extractor.h"
#include "core/optimizer/function_extractor_matcher.h"
#include "core/optimizer/function_extractor_pattern.h"
#include "onnx/defs/function.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"

namespace onnxruntime {
namespace test {

namespace {

using FunctionProto = ONNX_NAMESPACE::FunctionProto;
using NodeDef = ONNX_NAMESPACE::FunctionBodyHelper::NodeDef;

constexpr const char* kFunctionDomain = "test.function";
constexpr int kFunctionOpset = 1;
constexpr int kOnnxOpset = 13;

class FunctionExtractorGraphBuilder final : public ::onnxruntime::test::ModelTestBuilder {
 public:
  using ::onnxruntime::test::ModelTestBuilder::MakeIntermediate;
  using ::onnxruntime::test::ModelTestBuilder::MakeOutput;
  using ::onnxruntime::test::ModelTestBuilder::ModelTestBuilder;

  template <typename T>
  NodeArg* MakeIntermediate(std::initializer_list<int64_t> shape) {
    return ::onnxruntime::test::ModelTestBuilder::MakeIntermediate<T>(
        std::vector<int64_t>{shape});
  }

  template <typename T>
  NodeArg* MakeOutput(std::initializer_list<int64_t> shape) {
    return ::onnxruntime::test::ModelTestBuilder::MakeOutput<T>(
        std::vector<int64_t>{shape});
  }
};

FunctionProto MakeFunction(std::string name,
                           gsl::span<const NodeDef> node_defs,
                           gsl::span<const std::string> inputs,
                           gsl::span<const std::string> outputs) {
  FunctionProto function_proto;
  function_proto.set_domain(kFunctionDomain);
  function_proto.set_name(std::move(name));

  for (const auto& input : inputs) {
    function_proto.add_input(input);
  }

  for (const auto& output : outputs) {
    function_proto.add_output(output);
  }

  const std::vector<NodeDef> node_defs_copy{node_defs.begin(), node_defs.end()};
  for (const auto& node : ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(node_defs_copy)) {
    function_proto.add_node()->CopyFrom(node);
  }

  auto& opset_import = *function_proto.add_opset_import();
  opset_import.set_domain(kOnnxDomain);
  opset_import.set_version(kOnnxOpset);
  return function_proto;
}

FunctionProto MakeLinearFunction(std::string name = "Linear") {
  const std::vector<NodeDef> nodes{
      {{"sum"}, "Add", {"x", "y"}},
      {{"out"}, "Relu", {"sum"}},
  };
  const std::vector<std::string> inputs{"x", "y"};
  const std::vector<std::string> outputs{"out"};
  return MakeFunction(std::move(name), nodes, inputs, outputs);
}

FunctionProto MakeLiteralFunction(std::string name = "Literal") {
  const std::vector<NodeDef> nodes{
      ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("one", 1.0f),
      {{"sum"}, "Add", {"x", "one"}},
      {{"out"}, "Relu", {"sum"}},
  };
  const std::vector<std::string> inputs{"x"};
  const std::vector<std::string> outputs{"out"};
  return MakeFunction(std::move(name), nodes, inputs, outputs);
}

void AddTensorValueInfo(FunctionProto& function_proto,
                        std::string_view name,
                        int32_t element_type,
                        gsl::span<const int64_t> dimensions) {
  auto& value_info = *function_proto.add_value_info();
  value_info.set_name(std::string{name});
  auto& tensor_type = *value_info.mutable_type()->mutable_tensor_type();
  tensor_type.set_elem_type(element_type);
  for (const int64_t dimension : dimensions) {
    tensor_type.mutable_shape()->add_dim()->set_dim_value(dimension);
  }
}

std::unique_ptr<Model> MakeModel(std::vector<FunctionProto> function_protos) {
  const std::unordered_map<std::string, int> imports{
      {kOnnxDomain, kOnnxOpset},
      {kFunctionDomain, kFunctionOpset},
  };
  return std::make_unique<Model>(
      "FunctionExtractorTest", false, ModelMetaData(), PathString(),
      IOnnxRuntimeOpSchemaRegistryList(), imports,
      function_protos,
      DefaultLoggingManager().DefaultLogger());
}

std::unique_ptr<Model> MakeModel(const FunctionProto& function_proto) {
  return MakeModel(std::vector<FunctionProto>{function_proto});
}

size_t CountOp(const Graph& graph, std::string_view domain, std::string_view op_type) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) {
    if (node.Domain() == domain && node.OpType() == op_type) {
      ++count;
    }
  }
  return count;
}

Node& FindOnlyOp(Graph& graph, std::string_view domain, std::string_view op_type) {
  Node* result = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.Domain() == domain && node.OpType() == op_type) {
      EXPECT_EQ(result, nullptr) << "Expected exactly one " << domain << "." << op_type;
      result = &node;
    }
  }

  EXPECT_NE(result, nullptr) << "Expected one " << domain << "." << op_type;
  return *result;
}

void AssertResolved(const Graph& graph) {
  ASSERT_FALSE(graph.GraphResolveNeeded());
  for (const auto& node : graph.Nodes()) {
    ASSERT_NE(node.Op(), nullptr) << node.Name();
  }
}

void AssertCallIO(const Node& node,
                  gsl::span<const std::string> expected_inputs,
                  gsl::span<const std::string> expected_outputs) {
  ASSERT_EQ(node.InputDefs().size(), expected_inputs.size());
  ASSERT_EQ(node.OutputDefs().size(), expected_outputs.size());
  for (size_t i = 0; i < expected_inputs.size(); ++i) {
    EXPECT_EQ(node.InputDefs()[i]->Name(), expected_inputs[i]);
  }
  for (size_t i = 0; i < expected_outputs.size(); ++i) {
    EXPECT_EQ(node.OutputDefs()[i]->Name(), expected_outputs[i]);
  }
}

std::shared_ptr<Model> SerializeAndReload(Model& model) {
  std::string serialized_model;
  EXPECT_TRUE(model.ToProto().SerializeToString(&serialized_model));

  ONNX_NAMESPACE::ModelProto model_proto;
  EXPECT_TRUE(model_proto.ParseFromString(serialized_model));

  std::shared_ptr<Model> reloaded_model;
  EXPECT_STATUS_OK(Model::Load(std::move(model_proto), reloaded_model, nullptr,
                               DefaultLoggingManager().DefaultLogger()));
  return reloaded_model;
}

void BuildLinearTarget(Graph& graph,
                       NodeArg*& x,
                       NodeArg*& y,
                       NodeArg*& sum,
                       NodeArg*& output) {
  FunctionExtractorGraphBuilder builder(graph);
  x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  sum = builder.MakeIntermediate<float>({2});
  output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {output});
  builder.SetGraphOutputs();
}

void AddLinearTarget(FunctionExtractorGraphBuilder& builder,
                     NodeArg* x,
                     NodeArg* y,
                     NodeArg* output) {
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {output});
}

ONNX_NAMESPACE::GraphProto MakeCaptureSubgraph(
    const std::string& captured_name,
    const ONNX_NAMESPACE::TypeProto& captured_type,
    std::string_view op_type = "Identity") {
  Model subgraph_model("CaptureSubgraph", false, ModelMetaData(), PathString(),
                       IOnnxRuntimeOpSchemaRegistryList(),
                       {{kOnnxDomain, kOnnxOpset}}, {},
                       DefaultLoggingManager().DefaultLogger());
  Graph& subgraph = subgraph_model.MainGraph();
  NodeArg& captured = subgraph.GetOrCreateNodeArg(captured_name, &captured_type);
  subgraph.AddOuterScopeNodeArg(captured_name);
  NodeArg& output = subgraph.GetOrCreateNodeArg("subgraph_output", &captured_type);
  subgraph.AddNode("capture", std::string{op_type}, "capture outer-scope value",
                   {&captured}, {&output});
  subgraph.SetOutputs({&output});
  EXPECT_STATUS_OK(subgraph.Resolve());
  return subgraph.ToGraphProto();
}

Node& AddCapturingIf(FunctionExtractorGraphBuilder& builder, NodeArg& captured,
                     std::string_view op_type = "Identity") {
  NodeArg* condition = builder.MakeInput<bool>({}, std::vector<bool>{true});
  NodeArg* if_output = builder.MakeOutput<float>({2});
  Node& if_node = builder.AddNode("If", {condition}, {if_output});
  const ONNX_NAMESPACE::GraphProto branch =
      MakeCaptureSubgraph(captured.Name(), *captured.TypeAsProto(), op_type);
  if_node.AddAttribute("then_branch", branch);
  if_node.AddAttribute("else_branch", branch);
  return if_node;
}

common::Status FailGraphResolve(Graph&, const Graph::ResolveOptions&) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "injected resolve failure");
}

}  // namespace

class FunctionExtractorTest : public ::testing::Test {
 protected:
  static void ExpectConstructionRejected(FunctionProto function_proto) {
    Model model("InvalidFunctionExtractorTest", false,
                DefaultLoggingManager().DefaultLogger());
    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(model.MainGraph());
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
    EXPECT_EQ(model.MainGraph().NumberOfNodes(), 0u);
  }
};

// Pattern validation and registration.

TEST_F(FunctionExtractorTest, RejectsInvalidFormalNames) {
  const std::vector<NodeDef> nodes{
      {{"sum"}, "Add", {"x", "y"}},
      {{"out"}, "Relu", {"sum"}},
  };

  for (const auto& inputs : std::vector<std::vector<std::string>>{
           {"", "y"}, {"x", "x"}, {"x", "out"}}) {
    SCOPED_TRACE(::testing::PrintToString(inputs));
    ExpectConstructionRejected(MakeFunction("InvalidInputs", nodes, inputs, std::vector<std::string>{"out"}));
  }

  for (const auto& outputs : std::vector<std::vector<std::string>>{
           {""}, {"out", "out"}}) {
    SCOPED_TRACE(::testing::PrintToString(outputs));
    ExpectConstructionRejected(MakeFunction("InvalidOutputs", nodes, std::vector<std::string>{"x", "y"}, outputs));
  }

  FunctionProto internally_produced_input =
      MakeFunction("ProducedInput", nodes, std::vector<std::string>{"sum", "y"},
                   std::vector<std::string>{"out"});
  ExpectConstructionRejected(std::move(internally_produced_input));
}

TEST_F(FunctionExtractorTest, RejectsInvalidAttributes) {
  FunctionProto duplicate_declaration = MakeLinearFunction("DuplicateAttribute");
  duplicate_declaration.add_attribute("axis");
  duplicate_declaration.add_attribute("axis");
  ExpectConstructionRejected(std::move(duplicate_declaration));

  FunctionProto duplicate_default = MakeLinearFunction("DuplicateDefault");
  duplicate_default.add_attribute("axis");
  duplicate_default.add_attribute_proto()->CopyFrom(
      ONNX_NAMESPACE::MakeAttribute("axis", int64_t{0}));
  duplicate_default.add_attribute_proto()->CopyFrom(
      ONNX_NAMESPACE::MakeAttribute("axis", int64_t{1}));
  ExpectConstructionRejected(std::move(duplicate_default));

  FunctionProto referenced_attribute = MakeLinearFunction("ReferencedAttribute");
  auto& attribute = *referenced_attribute.mutable_node(0)->add_attribute();
  attribute.set_name("axis");
  attribute.set_ref_attr_name("axis");
  attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  ExpectConstructionRejected(std::move(referenced_attribute));
}

TEST_F(FunctionExtractorTest, RejectsUnusedRequiredFunctionAttribute) {
  FunctionProto function_proto = MakeLinearFunction("RequiredAttribute");
  function_proto.add_attribute("axis");
  ExpectConstructionRejected(std::move(function_proto));
}

TEST_F(FunctionExtractorTest, RejectsMalformedDataflow) {
  const std::vector<std::string> inputs{"x", "y"};
  const std::vector<std::string> outputs{"out"};

  const std::vector<std::vector<NodeDef>> invalid_bodies{
      {{{"sum"}, "Add", {"x", "undefined"}}, {{"out"}, "Relu", {"sum"}}},
      {{{"sum"}, "Add", {"x", "y"}}, {{"sum"}, "Mul", {"x", "y"}}, {{"out"}, "Relu", {"sum"}}},
      {{{"a"}, "Add", {"b", "x"}}, {{"b"}, "Mul", {"a", "y"}}, {{"out"}, "Relu", {"a"}}},
      {{{"sum"}, "Add", {"x", "y"}}, {{"dead"}, "Mul", {"x", "y"}}, {{"out"}, "Relu", {"sum"}}},
      {{{"sum"}, "Add", {"x", ""}}, {{"out"}, "Relu", {"sum"}}},
  };

  for (size_t i = 0; i < invalid_bodies.size(); ++i) {
    SCOPED_TRACE(i);
    ExpectConstructionRejected(MakeFunction("Malformed" + std::to_string(i),
                                            invalid_bodies[i], inputs, outputs));
  }
}

TEST_F(FunctionExtractorTest, RejectsDisconnectedMultiOutputBody) {
  const std::vector<NodeDef> nodes{
      {{"left"}, "Add", {"x", "y"}},
      {{"right"}, "Mul", {"z", "w"}},
  };
  ExpectConstructionRejected(
      MakeFunction("DisconnectedOutputs", nodes,
                   std::vector<std::string>{"x", "y", "z", "w"},
                   std::vector<std::string>{"left", "right"}));
}

TEST_F(FunctionExtractorTest, RejectsOutputUnreachableOperations) {
  const std::vector<std::string> inputs{"x", "y"};
  const std::vector<std::string> outputs{"out"};
  const std::vector<std::vector<NodeDef>> invalid_bodies{
      {{{"out"}, "Add", {"x", "y"}}, {{"after"}, "Relu", {"out"}}},
      {{{"sum"}, "Add", {"x", "y"}}, {{"out"}, "Relu", {"sum"}}, {{"dead"}, "Mul", {"x", "y"}}},
  };

  for (size_t i = 0; i < invalid_bodies.size(); ++i) {
    SCOPED_TRACE(i);
    ExpectConstructionRejected(MakeFunction("Unreachable" + std::to_string(i),
                                            invalid_bodies[i], inputs, outputs));
  }
}

TEST_F(FunctionExtractorTest, RejectsConstantFormalOutput) {
  const std::vector<std::string> no_inputs;
  const std::vector<std::string> outputs{"out"};
  const std::vector<NodeDef> constant_body{
      ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("literal", 1.0f),
      {{"out"}, "Identity", {"literal"}},
  };
  FunctionProto function_proto = MakeFunction("ConstantOutput", constant_body, no_inputs, outputs);
  function_proto.mutable_output(0)->assign("literal");
  ExpectConstructionRejected(std::move(function_proto));
}

TEST_F(FunctionExtractorTest, RejectsSingleOperationPattern) {
  const std::vector<NodeDef> nodes{{{"out"}, "Relu", {"x"}}};
  ExpectConstructionRejected(MakeFunction(
      "Single", nodes, std::vector<std::string>{"x"}, std::vector<std::string>{"out"}));
}

TEST_F(FunctionExtractorTest, RejectsUnsupportedBodyFeatures) {
  ONNX_NAMESPACE::GraphProto subgraph;
  subgraph.set_name("body");
  const std::vector<NodeDef> nodes{
      {{"branch"}, "If", {"condition"}, {ONNX_NAMESPACE::MakeAttribute("then_branch", subgraph), ONNX_NAMESPACE::MakeAttribute("else_branch", subgraph)}},
      {{"out"}, "Identity", {"branch"}},
  };
  ExpectConstructionRejected(
      MakeFunction("ControlFlow", nodes, std::vector<std::string>{"condition"},
                   std::vector<std::string>{"out"}));
}

TEST_F(FunctionExtractorTest, RejectsUnregisteredFunction) {
  FunctionProto function_proto = MakeLinearFunction();
  Model model("UnregisteredFunction", false, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_STATUS_OK(graph.Resolve());
  const size_t original_node_count = graph.NumberOfNodes();

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), original_node_count);
}

TEST_F(FunctionExtractorTest, RejectsDifferentRegisteredDefinition) {
  const FunctionProto registered_function = MakeLinearFunction();
  FunctionProto requested_function = MakeLinearFunction();
  requested_function.mutable_node(0)->set_op_type("Sub");
  auto model = MakeModel(registered_function);
  Graph& graph = model->MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(requested_function);
  const FunctionExtractionResult result = extractor.Extract(graph);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), 2u);
}

TEST_F(FunctionExtractorTest, RejectsContextDependentSchemaFunction) {
  const std::vector<NodeDef> nodes{
      {{"scaled"}, "Mul", {"x", "x"}},
      {{"out"}, "Relu", {"scaled"}},
  };
  FunctionProto function_proto =
      MakeFunction("Celu", nodes, std::vector<std::string>{"x"},
                   std::vector<std::string>{"out"});
  function_proto.set_domain(kOnnxDomain);

  Model model("ContextDependentFunction", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, kOnnxOpset}}, {},
              DefaultLoggingManager().DefaultLogger());
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(model);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, ResolvesNestedLocalFunctionIdentity) {
  const FunctionProto nested_function = MakeLinearFunction("Nested");
  const std::vector<NodeDef> outer_nodes{
      {{"nested_out"}, "Nested", {"x", "y"}, {}, kFunctionDomain},
      {{"out"}, "Identity", {"nested_out"}},
  };
  FunctionProto outer_function =
      MakeFunction("Outer", outer_nodes, std::vector<std::string>{"x", "y"},
                   std::vector<std::string>{"out"});
  auto& function_import = *outer_function.add_opset_import();
  function_import.set_domain(kFunctionDomain);
  function_import.set_version(kFunctionOpset);

  auto model = MakeModel(std::vector<FunctionProto>{nested_function, outer_function});
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* nested_output = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Nested", {x, y}, {nested_output}, kFunctionDomain);
  builder.AddNode("Identity", {nested_output}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(outer_function);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, outer_function.name()), 1u);
}

TEST_F(FunctionExtractorTest, RejectsUnresolvedTargetGraph) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_TRUE(graph.GraphResolveNeeded());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), 2u);
}

TEST_F(FunctionExtractorTest, RejectsImpureOrUnknownOperation) {
  const std::vector<std::string> inputs{"x", "y"};
  const std::vector<std::string> outputs{"out"};
  const std::vector<std::vector<NodeDef>> bodies{
      {{{"random"}, "RandomNormal", {}}, {{"out"}, "Add", {"x", "random"}}},
      {{{"custom"}, "Unknown", {"x"}}, {{"out"}, "Add", {"custom", "y"}}},
  };

  for (size_t i = 0; i < bodies.size(); ++i) {
    SCOPED_TRACE(i);
    FunctionProto function_proto = MakeFunction("Impure" + std::to_string(i), bodies[i], inputs, outputs);
    auto model = MakeModel(function_proto);
    ASSERT_STATUS_OK(model->MainGraph().Resolve());
    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(model->MainGraph());
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
  }
}

TEST_F(FunctionExtractorTest, EnforcesResourceBudgetsBeforeMutation) {
  const FunctionProto function_proto = MakeLinearFunction();
  struct BudgetCase {
    const char* name;
    FunctionExtractorOptions options;
  };

  FunctionExtractorOptions pattern_node_limit;
  pattern_node_limit.max_pattern_nodes = 1;
  FunctionExtractorOptions target_node_limit;
  target_node_limit.max_target_nodes = 1;
  FunctionExtractorOptions root_tuple_limit;
  root_tuple_limit.max_output_root_tuples = 0;
  FunctionExtractorOptions worklist_limit;
  worklist_limit.max_worklist_bindings = 0;
  const std::vector<BudgetCase> budget_cases{
      {"pattern node limit", pattern_node_limit},
      {"target node limit", target_node_limit},
      {"output root tuple limit", root_tuple_limit},
      {"worklist binding limit", worklist_limit},
  };

  for (const auto& budget_case : budget_cases) {
    SCOPED_TRACE(budget_case.name);
    auto model = MakeModel(function_proto);
    Graph& graph = model->MainGraph();
    NodeArg* x;
    NodeArg* y;
    NodeArg* sum;
    NodeArg* output;
    BuildLinearTarget(graph, x, y, sum, output);
    ASSERT_STATUS_OK(graph.Resolve());

    FunctionExtractor extractor(function_proto, budget_case.options);
    const FunctionExtractionResult result = extractor.Extract(graph);
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
    EXPECT_EQ(graph.NumberOfNodes(), 2u);
  }
}

TEST_F(FunctionExtractorTest, RejectsHighArityPatternBeforeMutation) {
  constexpr size_t input_count = 128;
  std::vector<std::string> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back("x" + std::to_string(i));
  }
  const std::vector<NodeDef> nodes{
      {{"joined"}, "Concat", inputs, {ONNX_NAMESPACE::MakeAttribute("axis", int64_t{0})}},
      {{"out"}, "Identity", {"joined"}},
  };
  const FunctionProto function_proto =
      MakeFunction("HighArity", nodes, inputs, std::vector<std::string>{"out"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  std::vector<NodeArg*> target_inputs;
  target_inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    target_inputs.push_back(builder.MakeInput<float>({1}, 0.0f, 1.0f));
  }
  NodeArg* joined = builder.MakeIntermediate<float>({static_cast<int64_t>(input_count)});
  NodeArg* output = builder.MakeOutput<float>({static_cast<int64_t>(input_count)});
  NodeAttributes attributes{{"axis", ONNX_NAMESPACE::MakeAttribute("axis", int64_t{0})}};
  builder.AddNode("Concat", target_inputs, {joined}, kOnnxDomain, &attributes);
  builder.AddNode("Identity", {joined}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  const size_t node_count_before = graph.NumberOfNodes();
  std::vector<std::pair<NodeIndex, const Node*>> node_identities_before;
  for (const Node& node : graph.Nodes()) {
    node_identities_before.emplace_back(node.Index(), &node);
  }
  const std::string graph_proto_before = graph.ToGraphProto().SerializeAsString();
  ASSERT_FALSE(graph.GraphResolveNeeded());

  FunctionExtractorOptions options;
  options.max_worklist_bindings = 16;
  FunctionExtractor extractor(function_proto, options);
  const FunctionExtractionResult result = extractor.Extract(graph);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), node_count_before);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 0u);
  EXPECT_FALSE(graph.GraphResolveNeeded());
  EXPECT_EQ(graph.ToGraphProto().SerializeAsString(), graph_proto_before);

  std::vector<std::pair<NodeIndex, const Node*>> node_identities_after;
  for (const Node& node : graph.Nodes()) {
    node_identities_after.emplace_back(node.Index(), &node);
  }
  EXPECT_EQ(node_identities_after, node_identities_before);
}

// Deterministic structural matching.

TEST_F(FunctionExtractorTest, ExtractsLinearPattern) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_STATUS_OK(graph.Resolve());
  const std::vector<std::string> expected_inputs{x->Name(), y->Name()};
  const std::vector<std::string> expected_outputs{output->Name()};

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Add"), 0u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Relu"), 0u);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  AssertCallIO(call, expected_inputs, expected_outputs);
  AssertResolved(graph);
}

TEST_F(FunctionExtractorTest, ExtractsBranchedMultiOutputPattern) {
  const std::vector<NodeDef> nodes{
      {{"sum"}, "Add", {"x", "y"}},
      {{"scaled"}, "Mul", {"sum", "y"}},
      {{"activated"}, "Relu", {"sum"}},
  };
  const FunctionProto function_proto =
      MakeFunction("Branched", nodes, std::vector<std::string>{"x", "y"},
                   std::vector<std::string>{"scaled", "activated"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* scaled = builder.MakeOutput<float>({2});
  NodeArg* activated = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Mul", {sum, y}, {scaled});
  builder.AddNode("Relu", {sum}, {activated});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  AssertCallIO(call, std::vector<std::string>{x->Name(), y->Name()},
               std::vector<std::string>{scaled->Name(), activated->Name()});
}

TEST_F(FunctionExtractorTest, ExtractsDiamondAndProcessesSharedValueOnce) {
  const std::vector<NodeDef> nodes{
      {{"sum"}, "Add", {"x", "y"}},
      {{"left"}, "Relu", {"sum"}},
      {{"right"}, "Identity", {"sum"}},
      {{"out"}, "Mul", {"left", "right"}},
  };
  const FunctionProto function_proto =
      MakeFunction("Diamond", nodes, std::vector<std::string>{"x", "y"},
                   std::vector<std::string>{"out"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* left = builder.MakeIntermediate<float>({2});
  NodeArg* right = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {left});
  builder.AddNode("Identity", {sum}, {right});
  builder.AddNode("Mul", {left, right}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  using namespace function_extractor_internal;
  const NormalizedFunctionPattern normalized =
      NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
  ASSERT_STATUS_OK(normalized.construction_status);
  CompiledFunctionPattern compiled;
  ASSERT_STATUS_OK(CompileFunctionPattern(normalized, graph, compiled));
  TargetGraphSnapshot snapshot;
  ASSERT_STATUS_OK(BuildTargetGraphSnapshot(graph, compiled, FunctionExtractorOptions{}, snapshot));
  std::vector<ReplacementPlan> plans;
  MatcherDiagnostics diagnostics;
  ASSERT_STATUS_OK(DiscoverReplacementPlans(
      compiled, snapshot, FunctionExtractorOptions{}, plans, &diagnostics));
  ASSERT_EQ(plans.size(), 1u);
  EXPECT_EQ(diagnostics.worklist_bindings_processed, normalized.values.size());
  EXPECT_EQ(diagnostics.worklist_bindings_scheduled, normalized.values.size());
}

TEST_F(FunctionExtractorTest, RejectsInconsistentMultiOutputRootTuple) {
  const std::vector<NodeDef> nodes{
      {{"sum"}, "Add", {"x", "y"}},
      {{"left"}, "Relu", {"sum"}},
      {{"right"}, "Identity", {"sum"}},
  };
  const FunctionProto function_proto =
      MakeFunction("TwoRoots", nodes, std::vector<std::string>{"x", "y"},
                   std::vector<std::string>{"left", "right"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* other = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum_a = builder.MakeIntermediate<float>({2});
  NodeArg* sum_b = builder.MakeIntermediate<float>({2});
  NodeArg* left = builder.MakeOutput<float>({2});
  NodeArg* right = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum_a});
  builder.AddNode("Add", {x, other}, {sum_b});
  builder.AddNode("Relu", {sum_a}, {left});
  builder.AddNode("Identity", {sum_b}, {right});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 0u);
}

TEST_F(FunctionExtractorTest, EnumeratesOutputRootsDeterministically) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* first = builder.MakeOutput<float>({2});
  NodeArg* second = builder.MakeOutput<float>({2});
  AddLinearTarget(builder, x, y, first);
  AddLinearTarget(builder, x, y, second);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  using namespace function_extractor_internal;
  const NormalizedFunctionPattern normalized =
      NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
  CompiledFunctionPattern compiled;
  ASSERT_STATUS_OK(CompileFunctionPattern(normalized, graph, compiled));
  TargetGraphSnapshot snapshot;
  ASSERT_STATUS_OK(BuildTargetGraphSnapshot(graph, compiled, FunctionExtractorOptions{}, snapshot));
  std::vector<ReplacementPlan> plans;
  ASSERT_STATUS_OK(DiscoverReplacementPlans(compiled, snapshot, FunctionExtractorOptions{}, plans));
  ASSERT_EQ(plans.size(), 2u);
  EXPECT_LT(plans[0].primary_root_topological_position,
            plans[1].primary_root_topological_position);
}

TEST_F(FunctionExtractorTest, MatchesRepeatedAndAliasedFormalInputs) {
  const std::vector<NodeDef> nodes{
      {{"sum"}, "Add", {"x", "x"}},
      {{"out"}, "Mul", {"sum", "y"}},
  };
  const FunctionProto function_proto =
      MakeFunction("AliasedInputs", nodes, std::vector<std::string>{"x", "y"},
                   std::vector<std::string>{"out"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* input = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {input, input}, {sum});
  builder.AddNode("Mul", {sum, input}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  AssertCallIO(call, std::vector<std::string>{input->Name(), input->Name()},
               std::vector<std::string>{output->Name()});
}

TEST_F(FunctionExtractorTest, RejectsReversedInternalOperands) {
  const std::vector<NodeDef> nodes{
      {{"difference"}, "Sub", {"x", "y"}},
      {{"out"}, "Div", {"difference", "y"}},
  };
  const FunctionProto function_proto =
      MakeFunction("Positional", nodes, std::vector<std::string>{"x", "y"},
                   std::vector<std::string>{"out"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* difference = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Sub", {x, y}, {difference});
  builder.AddNode("Div", {y, difference}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RootIndexAllowsExternalProducerAndOutputFanout) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* source = builder.MakeInput<float>({2}, -1.0f, 1.0f);
  NodeArg* x = builder.MakeIntermediate<float>({2});
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* matched_output = builder.MakeIntermediate<float>({2});
  NodeArg* output_a = builder.MakeOutput<float>({2});
  NodeArg* output_b = builder.MakeOutput<float>({2});
  builder.AddNode("Abs", {source}, {x});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {matched_output});
  builder.AddNode("Identity", {matched_output}, {output_a});
  builder.AddNode("Neg", {matched_output}, {output_b});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(graph.GetConsumerNodes(matched_output->Name()).size(), 2u);
}

TEST_F(FunctionExtractorTest, RequiresExactOperatorIdentityAndArity) {
  const FunctionProto function_proto = MakeLinearFunction();
  for (const std::string& second_op : {"Sigmoid", "Neg"}) {
    SCOPED_TRACE(second_op);
    auto model = MakeModel(function_proto);
    Graph& graph = model->MainGraph();
    FunctionExtractorGraphBuilder builder(graph);
    NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
    NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
    NodeArg* sum = builder.MakeIntermediate<float>({2});
    NodeArg* output = builder.MakeOutput<float>({2});
    builder.AddNode("Add", {x, y}, {sum});
    builder.AddNode(second_op, {sum}, {output});
    builder.SetGraphOutputs();
    ASSERT_STATUS_OK(graph.Resolve());

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(graph);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, 0u);
  }
}

TEST_F(FunctionExtractorTest, RequiresExactEffectiveAttributes) {
  const std::vector<NodeDef> nodes{
      {{"transposed"}, "Transpose", {"x"}, {ONNX_NAMESPACE::MakeAttribute("perm", std::vector<int64_t>{1, 0})}},
      {{"out"}, "Relu", {"transposed"}},
  };
  const FunctionProto function_proto =
      MakeFunction("Attributes", nodes, std::vector<std::string>{"x"},
                   std::vector<std::string>{"out"});

  for (const std::vector<int64_t>& perm :
       {std::vector<int64_t>{1, 0}, std::vector<int64_t>{0, 1}}) {
    SCOPED_TRACE(::testing::PrintToString(perm));
    auto model = MakeModel(function_proto);
    Graph& graph = model->MainGraph();
    FunctionExtractorGraphBuilder builder(graph);
    NodeArg* input = builder.MakeInput<float>({2, 2}, 0.0f, 1.0f);
    NodeArg* transposed = builder.MakeIntermediate<float>({2, 2});
    NodeArg* output = builder.MakeOutput<float>({2, 2});
    NodeAttributes attributes{{"perm", ONNX_NAMESPACE::MakeAttribute("perm", perm)}};
    builder.AddNode("Transpose", {input}, {transposed}, kOnnxDomain, &attributes);
    builder.AddNode("Relu", {transposed}, {output});
    builder.SetGraphOutputs();
    ASSERT_STATUS_OK(graph.Resolve());

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(graph);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, perm == std::vector<int64_t>({1, 0}) ? 1u : 0u);
  }
}

TEST_F(FunctionExtractorTest, MatchesOptionalAndVariadicSlotsPositionally) {
  const std::vector<NodeDef> nodes{
      {{"clipped"}, "Clip", {"x", "", ""}},
      {{"out"}, "Concat", {"clipped", "y"}, {ONNX_NAMESPACE::MakeAttribute("axis", int64_t{0})}},
  };
  const FunctionProto function_proto =
      MakeFunction("OptionalVariadic", nodes, std::vector<std::string>{"x", "y"},
                   std::vector<std::string>{"out"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({1}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({1}, 0.0f, 1.0f);
  NodeArg* empty = builder.MakeEmptyInput();
  NodeArg* clipped = builder.MakeIntermediate<float>({1});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Clip", {x, empty, empty}, {clipped});
  NodeAttributes attributes{{"axis", ONNX_NAMESPACE::MakeAttribute("axis", int64_t{0})}};
  builder.AddNode("Concat", {clipped, y}, {output}, kOnnxDomain, &attributes);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
}

TEST_F(FunctionExtractorTest, MatchesOperatorWithOmittedOptionalOutput) {
  const std::vector<NodeDef> nodes{
      {{"pooled", ""}, "MaxPool", {"x"}, {ONNX_NAMESPACE::MakeAttribute("kernel_shape", std::vector<int64_t>{2, 2})}},
      {{"out"}, "Identity", {"pooled"}},
  };
  const FunctionProto function_proto =
      MakeFunction("OmittedOptionalOutput", nodes, std::vector<std::string>{"x"},
                   std::vector<std::string>{"out"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({1, 1, 4, 4}, 0.0f, 1.0f);
  NodeArg* pooled = builder.MakeIntermediate<float>({1, 1, 3, 3});
  NodeArg* missing_indices = builder.MakeEmptyInput();
  NodeArg* output = builder.MakeOutput<float>({1, 1, 3, 3});
  NodeAttributes attributes{
      {"kernel_shape", ONNX_NAMESPACE::MakeAttribute(
                           "kernel_shape", std::vector<int64_t>{2, 2})}};
  builder.AddNode("MaxPool", {x}, {pooled, missing_indices}, kOnnxDomain,
                  &attributes);
  builder.AddNode("Identity", {pooled}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
}

TEST_F(FunctionExtractorTest, AppliesKnownTypeCompatibilityRules) {
  const std::vector<NodeDef> nodes{
      {{"intermediate"}, "Identity", {"x"}},
      {{"out"}, "Identity", {"intermediate"}},
  };
  FunctionProto function_proto =
      MakeFunction("KnownTypes", nodes, std::vector<std::string>{"x"},
                   std::vector<std::string>{"out"});
  for (const std::string_view value_name : {"x", "intermediate", "out"}) {
    AddTensorValueInfo(function_proto, value_name,
                       ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                       std::vector<int64_t>{2});
  }

  struct FloatShapeCase {
    const char* name;
    std::vector<int64_t> shape;
    size_t expected_replacements;
  };
  const std::vector<FloatShapeCase> float_shape_cases{
      {"compatible", {2}, 1},
      {"rank mismatch", {2, 1}, 0},
      {"dimension mismatch", {3}, 0},
  };
  for (const auto& test_case : float_shape_cases) {
    SCOPED_TRACE(test_case.name);
    auto model = MakeModel(function_proto);
    Graph& graph = model->MainGraph();
    FunctionExtractorGraphBuilder builder(graph);
    NodeArg* x = builder.MakeInput<float>(test_case.shape, 0.0f, 1.0f);
    NodeArg* intermediate = builder.MakeIntermediate<float>(test_case.shape);
    NodeArg* output = builder.MakeOutput<float>(test_case.shape);
    builder.AddNode("Identity", {x}, {intermediate});
    builder.AddNode("Identity", {intermediate}, {output});
    builder.SetGraphOutputs();
    ASSERT_STATUS_OK(graph.Resolve());

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(graph);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, test_case.expected_replacements);
  }

  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<int32_t>({2}, 0, 10);
  NodeArg* intermediate = builder.MakeIntermediate<int32_t>({2});
  NodeArg* output = builder.MakeOutput<int32_t>({2});
  builder.AddNode("Identity", {x}, {intermediate});
  builder.AddNode("Identity", {intermediate}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RejectsNonTensorValueInfo) {
  FunctionProto function_proto = MakeLinearFunction("NonTensorValueInfo");
  auto& value_info = *function_proto.add_value_info();
  value_info.set_name("sum");
  value_info.mutable_type()
      ->mutable_sequence_type()
      ->mutable_elem_type()
      ->mutable_tensor_type()
      ->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  ExpectConstructionRejected(std::move(function_proto));
}

// Literal matching and preservation.

TEST_F(FunctionExtractorTest, ComparesFloatingAndTensorBitsExactly) {
  auto make_float_tensor = [](uint32_t bits) {
    ONNX_NAMESPACE::TensorProto tensor;
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor.add_dims(1);
    tensor.set_raw_data(&bits, sizeof(bits));
    return tensor;
  };

  using function_extractor_internal::CompareTensorLiterals;
  bool equal = true;
  ASSERT_STATUS_OK(CompareTensorLiterals(
      make_float_tensor(0x00000000u), make_float_tensor(0x80000000u), 1024, equal));
  EXPECT_FALSE(equal);
  ASSERT_STATUS_OK(CompareTensorLiterals(
      make_float_tensor(0x7fc00001u), make_float_tensor(0x7fc00002u), 1024, equal));
  EXPECT_FALSE(equal);
  ASSERT_STATUS_OK(CompareTensorLiterals(
      make_float_tensor(0x7fc00001u), make_float_tensor(0x7fc00001u), 1024, equal));
  EXPECT_TRUE(equal);
}

TEST_F(FunctionExtractorTest, MatchesLiteralFromInitializer) {
  const FunctionProto function_proto = MakeLiteralFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* input = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* literal = builder.MakeScalarInitializer<float>(1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {input, literal}, {sum});
  builder.AddNode("Relu", {sum}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  const std::string literal_name = literal->Name();

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const ONNX_NAMESPACE::TensorProto* retained_literal = nullptr;
  EXPECT_TRUE(graph.GetInitializedTensor(literal_name, retained_literal));
  EXPECT_NE(retained_literal, nullptr);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  AssertCallIO(call, std::vector<std::string>{input->Name()},
               std::vector<std::string>{output->Name()});
}

TEST_F(FunctionExtractorTest, MatchesLiteralFromConstantNode) {
  const FunctionProto function_proto = MakeLiteralFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* input = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* literal = builder.MakeIntermediate<float>({});
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  NodeAttributes constant_attributes{
      {"value", ONNX_NAMESPACE::MakeAttribute(
                    "value", ONNX_NAMESPACE::ToTensor<float>(1.0f))}};
  builder.AddNode("Constant", {}, {literal}, kOnnxDomain, &constant_attributes);
  builder.AddNode("Add", {input, literal}, {sum});
  builder.AddNode("Relu", {sum}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Constant"), 1u);
}

TEST_F(FunctionExtractorTest, LeavesSharedAndDeadLiteralWitnesses) {
  const FunctionProto function_proto = MakeLiteralFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* input = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* literal = builder.MakeScalarInitializer<float>(1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* matched_output = builder.MakeIntermediate<float>({2});
  NodeArg* graph_output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {input, literal}, {sum});
  builder.AddNode("Relu", {sum}, {matched_output});
  builder.AddNode("Add", {matched_output, literal}, {graph_output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  const std::string literal_name = literal->Name();

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const ONNX_NAMESPACE::TensorProto* retained_literal = nullptr;
  EXPECT_TRUE(graph.GetInitializedTensor(literal_name, retained_literal));
  EXPECT_NE(retained_literal, nullptr);
}

TEST_F(FunctionExtractorTest, RejectsOverridableInitializerLiteral) {
  const FunctionProto function_proto = MakeLiteralFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* input = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* literal = builder.MakeScalarInitializer<float>(1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {input, literal}, {sum});
  builder.AddNode("Relu", {sum}, {output});
  builder.SetGraphOutputs();
  graph.SetInputs({input, literal});
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RejectsLiteralFormalInputAlias) {
  const FunctionProto function_proto = MakeLiteralFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* literal = builder.MakeScalarInitializer<float>(1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {literal, literal}, {sum});
  builder.AddNode("Relu", {sum}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, AllowsEqualLiteralsToShareWitness) {
  const std::vector<NodeDef> nodes{
      ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("one_a", 1.0f),
      ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("one_b", 1.0f),
      {{"sum"}, "Add", {"x", "one_a"}},
      {{"out"}, "Mul", {"sum", "one_b"}},
  };
  const FunctionProto function_proto =
      MakeFunction("SharedLiteral", nodes, std::vector<std::string>{"x"},
                   std::vector<std::string>{"out"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* input = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* literal = builder.MakeScalarInitializer<float>(1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {input, literal}, {sum});
  builder.AddNode("Mul", {sum, literal}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
}

// Boundary closure, convexity, and graph scopes.

TEST_F(FunctionExtractorTest, RejectsPrivateIntermediateExternalConsumer) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* matched_output = builder.MakeOutput<float>({2});
  NodeArg* side_output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {matched_output});
  builder.AddNode("Neg", {sum}, {side_output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RejectsPrivateIntermediateGraphOutput) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeOutput<float>({2});
  NodeArg* matched_output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {matched_output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RejectsPrivateIntermediateImplicitCapture) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* matched_output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {matched_output});
  Node& if_node = AddCapturingIf(builder, *sum);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_EQ(if_node.ImplicitInputDefs().size(), 1u);
  ASSERT_EQ(if_node.ImplicitInputDefs()[0]->Name(), sum->Name());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, PreservesFormalOutputConsumersAndImplicitCaptures) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* matched_output = builder.MakeIntermediate<float>({2});
  NodeArg* explicit_output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {matched_output});
  builder.AddNode("Identity", {matched_output}, {explicit_output});
  Node& if_node = AddCapturingIf(builder, *matched_output);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_EQ(if_node.ImplicitInputDefs().size(), 1u);

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  AssertResolved(graph);
  const Node& remaining_if = FindOnlyOp(graph, kOnnxDomain, "If");
  ASSERT_EQ(remaining_if.ImplicitInputDefs().size(), 1u);
  EXPECT_EQ(remaining_if.ImplicitInputDefs()[0]->Name(), matched_output->Name());
  EXPECT_EQ(graph.GetConsumerNodes(matched_output->Name()).size(), 2u);
}

TEST_F(FunctionExtractorTest, RejectsDownstreamFormalInputBinding) {
  const std::vector<NodeDef> nodes{{{"activated"}, "Relu", {"x"}},
                                   {{"out"}, "Add", {"activated", "y"}}};
  const FunctionProto function_proto =
      MakeFunction("DownstreamInput", nodes, std::vector<std::string>{"x", "y"},
                   std::vector<std::string>{"out"});
  for (const bool bind_formal_directly_to_matched_output : {false, true}) {
    SCOPED_TRACE(bind_formal_directly_to_matched_output);
    auto model = MakeModel(function_proto);
    Graph& graph = model->MainGraph();
    FunctionExtractorGraphBuilder builder(graph);
    NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
    NodeArg* activated = builder.MakeIntermediate<float>({2});
    NodeArg* output = builder.MakeOutput<float>({2});
    builder.AddNode("Relu", {x}, {activated});

    NodeArg* formal_input_binding = activated;
    if (!bind_formal_directly_to_matched_output) {
      formal_input_binding = builder.MakeIntermediate<float>({2});
      builder.AddNode("Identity", {activated}, {formal_input_binding});
    }

    builder.AddNode("Add", {activated, formal_input_binding}, {output});
    builder.SetGraphOutputs();
    ASSERT_STATUS_OK(graph.Resolve());

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(graph);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, 0u);
  }
}

TEST_F(FunctionExtractorTest, RejectsNonConvexLeaveAndReenterPath) {
  const std::vector<NodeDef> nodes{{{"activated"}, "Relu", {"x"}},
                                   {{"out"}, "Add", {"activated", "y"}}};
  const FunctionProto function_proto =
      MakeFunction("NonConvex", nodes, std::vector<std::string>{"x", "y"},
                   std::vector<std::string>{"out"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* activated = builder.MakeIntermediate<float>({2});
  NodeArg* outside = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Relu", {x}, {activated});
  builder.AddNode("Identity", {activated}, {outside});
  builder.AddNode("Add", {activated, outside}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RejectsProviderControlOrAnnotationMismatch) {
  const FunctionProto function_proto = MakeLinearFunction();
  enum class RejectionReason {
    ProviderAssignment,
    ControlEdge,
    LayeringAnnotation,
  };
  for (const RejectionReason reason :
       {RejectionReason::ProviderAssignment,
        RejectionReason::ControlEdge,
        RejectionReason::LayeringAnnotation}) {
    SCOPED_TRACE(static_cast<int>(reason));
    auto model = MakeModel(function_proto);
    Graph& graph = model->MainGraph();
    NodeArg* x;
    NodeArg* y;
    NodeArg* sum;
    NodeArg* output;
    BuildLinearTarget(graph, x, y, sum, output);
    NodeIndex control_source_index = 0;
    NodeIndex control_target_index = 0;
    if (reason == RejectionReason::ControlEdge) {
      Node* add_node = nullptr;
      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "Add") {
          add_node = &node;
          break;
        }
      }
      ASSERT_NE(add_node, nullptr);
      FunctionExtractorGraphBuilder builder(graph);
      NodeArg* control_output = builder.MakeIntermediate<float>({2});
      Node& control_source = builder.AddNode("Identity", {x}, {control_output});
      control_source_index = control_source.Index();
      control_target_index = add_node->Index();
    }
    ASSERT_STATUS_OK(graph.Resolve());
    if (reason == RejectionReason::ControlEdge) {
      ASSERT_TRUE(graph.AddControlEdge(control_source_index, control_target_index));
      using namespace function_extractor_internal;
      const NormalizedFunctionPattern normalized =
          NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
      ASSERT_STATUS_OK(normalized.construction_status);
      CompiledFunctionPattern compiled;
      ASSERT_STATUS_OK(CompileFunctionPattern(normalized, graph, compiled));
      TargetGraphSnapshot snapshot;
      ASSERT_STATUS_OK(
          BuildTargetGraphSnapshot(graph, compiled, FunctionExtractorOptions{},
                                   snapshot));
      std::vector<ReplacementPlan> plans;
      ASSERT_STATUS_OK(DiscoverReplacementPlans(
          compiled, snapshot, FunctionExtractorOptions{}, plans));
      EXPECT_TRUE(plans.empty());
      continue;
    }
    auto nodes = graph.Nodes();
    auto node_it = nodes.begin();
    Node& first = *node_it;
    Node& second = *++node_it;
    if (reason == RejectionReason::ProviderAssignment) {
      first.SetExecutionProviderType(kCpuExecutionProvider);
    } else if (reason == RejectionReason::LayeringAnnotation) {
      first.SetLayeringAnnotation("layer-a");
      second.SetLayeringAnnotation("layer-b");
    }

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(graph);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, 0u);
  }
}

TEST_F(FunctionExtractorTest, DoesNotCrossGraphScopes) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  AddCapturingIf(builder, *sum, "Relu");
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Add"), 1u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 0u);
}

// Batching, application, fixpoint, and persistence.

TEST_F(FunctionExtractorTest, AppliesDisjointMatchesInOneBatch) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* first = builder.MakeOutput<float>({2});
  NodeArg* second = builder.MakeOutput<float>({2});
  AddLinearTarget(builder, x, y, first);
  AddLinearTarget(builder, x, y, second);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 2u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 2u);
  EXPECT_EQ(graph.NumberOfNodes(), 2u);
}

TEST_F(FunctionExtractorTest, SelectsOverlappingMatchesDeterministically) {
  const std::vector<NodeDef> nodes{
      {{"sum"}, "Add", {"x", "y"}},
      {{"out"}, "Relu", {"sum"}},
  };
  const FunctionProto function_proto =
      MakeFunction("Overlapping", nodes, std::vector<std::string>{"x", "y"},
                   std::vector<std::string>{"sum", "out"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* first = builder.MakeOutput<float>({2});
  NodeArg* second = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {first});
  builder.AddNode("Relu", {sum}, {second});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  ASSERT_EQ(call.OutputDefs().size(), 2u);
  EXPECT_EQ(call.OutputDefs()[0]->Name(), sum->Name());
  EXPECT_EQ(call.OutputDefs()[1]->Name(), first->Name());
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Relu"), 1u);
}

TEST_F(FunctionExtractorTest, DefersBoundaryAdjacentMatchToNextPass) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* z = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* first_sum = builder.MakeIntermediate<float>({2});
  NodeArg* boundary = builder.MakeIntermediate<float>({2});
  NodeArg* second_sum = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {first_sum});
  builder.AddNode("Relu", {first_sum}, {boundary});
  builder.AddNode("Add", {boundary, z}, {second_sum});
  builder.AddNode("Relu", {second_sum}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 2u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 2u);
  AssertResolved(graph);
}

TEST_F(FunctionExtractorTest, AllowsSharedUnrelatedBoundaryValues) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* shared = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* first = builder.MakeOutput<float>({2});
  NodeArg* second = builder.MakeOutput<float>({2});
  AddLinearTarget(builder, shared, shared, first);
  AddLinearTarget(builder, shared, shared, second);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 2u);
}

// Failure semantics and mutation invariants.

TEST_F(FunctionExtractorTest, RejectsStalePlanBeforeMutation) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_STATUS_OK(graph.Resolve());

  using namespace function_extractor_internal;
  const NormalizedFunctionPattern normalized =
      NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
  CompiledFunctionPattern compiled;
  ASSERT_STATUS_OK(CompileFunctionPattern(normalized, graph, compiled));
  TargetGraphSnapshot snapshot;
  ASSERT_STATUS_OK(BuildTargetGraphSnapshot(graph, compiled, FunctionExtractorOptions{}, snapshot));
  std::vector<ReplacementPlan> plans;
  ASSERT_STATUS_OK(DiscoverReplacementPlans(compiled, snapshot, FunctionExtractorOptions{}, plans));
  ASSERT_EQ(plans.size(), 1u);
  snapshot.graph_viewer.reset();

  const NodeIndex stale_node_index = plans[0].removable_node_indices.back();
  ASSERT_TRUE(graph.RemoveNode(stale_node_index));
  EXPECT_FALSE(PrevalidatePlans(graph, compiled, plans).IsOK());
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 0u);
}

TEST_F(FunctionExtractorTest, ReturnsInvariantErrorAtPassCap) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_STATUS_OK(graph.Resolve());
  const size_t original_node_count = graph.NumberOfNodes();

  using namespace function_extractor_internal;
  const NormalizedFunctionPattern normalized =
      NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
  ASSERT_STATUS_OK(normalized.construction_status);
  ExtractionControls controls;
  controls.maximum_passes = 0;

  const FunctionExtractionResult result =
      ExtractGraph(graph, normalized, FunctionExtractorOptions{}, controls);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_NE(result.status.ErrorMessage().find("defensive pass cap"), std::string::npos);
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), original_node_count);
  AssertResolved(graph);
}

TEST_F(FunctionExtractorTest, ReportsAppliedCountOnResolveFailure) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_STATUS_OK(graph.Resolve());

  using namespace function_extractor_internal;
  const NormalizedFunctionPattern normalized =
      NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
  ASSERT_STATUS_OK(normalized.construction_status);
  ExtractionControls controls;
  controls.resolve_graph = FailGraphResolve;

  const FunctionExtractionResult result =
      ExtractGraph(graph, normalized, FunctionExtractorOptions{}, controls);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_NE(result.status.ErrorMessage().find("injected resolve failure"), std::string::npos);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_TRUE(graph.GraphResolveNeeded());
  EXPECT_EQ(graph.NumberOfNodes(), 1u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 1u);
}

TEST_F(FunctionExtractorTest, StrictlyDecreasesNodeCountPerReplacement) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_STATUS_OK(graph.Resolve());
  const size_t original_node_count = graph.NumberOfNodes();

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  EXPECT_LT(graph.NumberOfNodes(), original_node_count);
  EXPECT_EQ(graph.NumberOfNodes(), 1u);
}

TEST_F(FunctionExtractorTest, PreservesOutputIdentityAndFanout) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* pattern_output = builder.MakeIntermediate<float>({2});
  NodeArg* output_a = builder.MakeOutput<float>({2});
  NodeArg* output_b = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {pattern_output});
  builder.AddNode("Identity", {pattern_output}, {output_a});
  builder.AddNode("Neg", {pattern_output}, {output_b});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  const std::string pattern_output_name = pattern_output->Name();
  const ONNX_NAMESPACE::TypeProto pattern_output_type = *pattern_output->TypeAsProto();

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  ASSERT_EQ(call.OutputDefs()[0]->Name(), pattern_output_name);
  EXPECT_EQ(call.OutputDefs()[0]->TypeAsProto()->SerializeAsString(),
            pattern_output_type.SerializeAsString());
  EXPECT_EQ(graph.GetOutputs()[0]->Name(), output_a->Name());
  EXPECT_EQ(graph.GetOutputs()[1]->Name(), output_b->Name());
  EXPECT_EQ(graph.GetConsumerNodes(pattern_output_name).size(), 2u);
}

TEST_F(FunctionExtractorTest, ReturnsResolvedGraphAtFixpoint) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult first = extractor.Extract(graph);
  ASSERT_STATUS_OK(first.status);
  EXPECT_EQ(first.replacements_applied, 1u);
  AssertResolved(graph);

  const FunctionExtractionResult second = extractor.Extract(graph);
  ASSERT_STATUS_OK(second.status);
  EXPECT_EQ(second.replacements_applied, 0u);
  AssertResolved(graph);
}

TEST_F(FunctionExtractorTest, PersistsRegisteredCallAfterSerializeReload) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);

  std::shared_ptr<Model> reloaded_model = SerializeAndReload(*model);
  ASSERT_NE(reloaded_model, nullptr);
  AssertResolved(reloaded_model->MainGraph());
  EXPECT_EQ(CountOp(reloaded_model->MainGraph(), kFunctionDomain, function_proto.name()), 1u);
  EXPECT_EQ(reloaded_model->ToProto().functions_size(), 1);
}

TEST_F(FunctionExtractorTest, RoundTripsThroughInlineFunction) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  ASSERT_STATUS_OK(graph.InlineFunction(call));
  ASSERT_STATUS_OK(graph.Resolve());
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 0u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Add"), 1u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Relu"), 1u);
  EXPECT_EQ(graph.GetOutputs()[0]->Name(), output->Name());
  AssertResolved(graph);
}

}  // namespace test
}  // namespace onnxruntime
