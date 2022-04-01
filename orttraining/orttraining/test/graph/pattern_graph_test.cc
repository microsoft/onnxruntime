// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/graph/graph.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "core/optimizer/utils.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/common/safeint.h"
#include "core/optimizer/initializer.h"
#include "orttraining/core/graph/graph_augmenter.h"
#include "orttraining/core/graph/pattern_graph/pattern_graph.h"

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

using onnxruntime::test::CountOpsInGraph;
using PatternTypeCategory = onnxruntime::training::PatternType::PatternTypeCategory;

namespace onnxruntime {
namespace training {
namespace test {

// TODO: move this into pattern_graph?
Status TryReplace(Graph& graph, PatternGraph& pattern, const PNode& alternative,
                  std::vector<std::pair<std::string, int>> fusion_inputs,
                  std::vector<std::pair<std::string, int>> fusion_outputs,
                  const std::string& start_node_name, const std::string& end_node_name) {
  PatternMatchResult match_results(graph);
  ORT_RETURN_IF_ERROR(pattern.TryMatch(graph, match_results));
  InlinedVector<std::reference_wrapper<Node>> matched_nodes;
  matched_nodes.push_back(*match_results.GetTargetNodeWithName(start_node_name));
  auto mapping = match_results.GetMatchMap();
  for (auto iter = mapping.rbegin(); iter != mapping.rend(); iter++) {
    if (iter->first != start_node_name && iter->first != end_node_name) {
      auto node = graph.GetNode(iter->second.first);
      matched_nodes.push_back(*node);
    }
  }
  matched_nodes.push_back(*match_results.GetTargetNodeWithName(end_node_name));

  auto add_node_args_with_name = [&match_results](std::string name, int idx, std::vector<NodeArg*>& args) {
    auto target_node = match_results.GetTargetNodeWithName(name);
    args.push_back(target_node->MutableInputDefs()[idx]);
  };

  std::vector<NodeArg*> input_args, output_args;
  for (auto item : fusion_inputs) {
    add_node_args_with_name(item.first, item.second, input_args);
  }
  for (auto item : fusion_outputs) {
    add_node_args_with_name(item.first, item.second, output_args);
  }
  auto node_def = alternative.GetNodeDef();
  Node& replace_node = graph.AddNode(node_def.name, node_def.op_type, "",
                                     input_args, output_args, {}, node_def.domain.c_str());
  for (auto& attr : node_def.attributes) {
    replace_node.AddAttributeProto(attr.second);
  }
  graph_utils::FinalizeNodeFusion(graph, matched_nodes, replace_node);
  return Status::OK();
}

// This method is used to replace node for EmbedLayerNorm case. It could be removed before formal PR.
template <typename T>
bool CheckEmbeddingData(const T* data, int64_t batch_size, int64_t element_count) {
  // check that all batches has same data.
  size_t data_length = SafeInt<size_t>(batch_size) * element_count;
  for (size_t i = gsl::narrow<size_t>(element_count); i < data_length; i++) {
    if (data[i] != data[i % element_count]) {
      return false;
    }
  }
  return true;
}

// This method is used to replace node for EmbedLayerNorm case. It could be removed before formal PR.
static NodeArg* ExtractEmbedding(Graph& graph,
                                 int64_t batch_size,
                                 int64_t sequence_length,
                                 int64_t hidden_size,
                                 const ONNX_NAMESPACE::TensorProto* tensor) {
  assert(nullptr != tensor);
  assert(batch_size > 0);
  assert(sequence_length > 0);
  assert(hidden_size > 0);

  Initializer old_initializer{*tensor, graph.ModelPath()};
  auto data_type = tensor->data_type();

  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name(graph.GenerateNodeArgName("position_embeddings"));
  initializer.add_dims(sequence_length);
  initializer.add_dims(hidden_size);
  initializer.set_data_type(data_type);
  const int64_t element_count = sequence_length * hidden_size;

  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const float* data = old_initializer.data<float>();
    if (!CheckEmbeddingData(data, batch_size, element_count)) {
      return nullptr;
    }

    initializer.set_raw_data(data, gsl::narrow<size_t>(element_count) * sizeof(float));
  } else {  // data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
    const MLFloat16* data = old_initializer.data<MLFloat16>();
    if (!CheckEmbeddingData(data, batch_size, element_count)) {
      return nullptr;
    }

    initializer.set_raw_data(data, gsl::narrow<size_t>(element_count) * sizeof(MLFloat16));
  }

  NodeArg& node_arg = graph_utils::AddInitializer(graph, initializer);
  return &node_arg;
}
// This method is used to replace node for EmbedLayerNorm case. It could be removed before formal PR.
static NodeArg* CastToInt32(Graph& graph, NodeArg* input, ProviderType provider_type) {
  auto data_type = input->TypeAsProto()->tensor_type().elem_type();
  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    return input;
  }
  const ONNX_NAMESPACE::TensorShapeProto* input_shape = input->Shape();
  TypeProto input_int32;
  input_int32.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  auto dim0 = input_int32.mutable_tensor_type()->mutable_shape()->add_dim();
  *dim0 = input_shape->dim(0);
  auto dim1 = input_int32.mutable_tensor_type()->mutable_shape()->add_dim();
  *dim1 = input_shape->dim(1);
  auto& cast32 = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(input->Name() + "_Int32"), &input_int32);

  Node& node = graph.AddNode(graph.GenerateNodeName(input->Name() + "_Cast"),
                             "Cast",
                             "Cast Input from int64 to int32",
                             std::array{input},
                             std::array{&cast32},
                             nullptr,
                             kOnnxDomain);

  // Add attribute: "to" = 6
  ONNX_NAMESPACE::AttributeProto to;
  to.set_name("to");
  to.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  to.set_i(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32));
  node.AddAttributeProto(std::move(to));

  node.SetExecutionProviderType(provider_type);
  return &cast32;
}
// This method is used to replace node for EmbedLayerNorm case. It could be removed before formal PR.
static void CreateEmbedLayernormNode(Graph& graph,
                                     NodeArg* input_ids,
                                     NodeArg* segment_ids,
                                     NodeArg* word_embedding,
                                     NodeArg* position_embedding,
                                     NodeArg* segment_embedding,
                                     Node& layer_norm_node,
                                     Node* embed_layer_norm_node) {
  // Cast input_ids and segment_ids to int32 if needed.
  input_ids = CastToInt32(graph, input_ids, layer_norm_node.GetExecutionProviderType());
  if (segment_ids != nullptr && segment_embedding != nullptr) {
    segment_ids = CastToInt32(graph, segment_ids, layer_norm_node.GetExecutionProviderType());
  }

  NodeArg place_holder("", nullptr);
  if (segment_ids == nullptr && segment_embedding == nullptr) {
    segment_ids = &place_holder;
    segment_embedding = &place_holder;
  }

  const std::vector<NodeArg*> embed_layer_norm_input_defs{
      input_ids,
      segment_ids,
      word_embedding,
      position_embedding,
      segment_embedding,
      layer_norm_node.MutableInputDefs()[1],
      layer_norm_node.MutableInputDefs()[2]};

  auto& mask_index = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("mask_index"), nullptr);

  auto& node = graph.AddNode(graph.GenerateNodeName("EmbedLayerNormalization"),
                             "EmbedLayerNormalization",
                             "fused EmbedLayerNorm subgraphs ",
                             embed_layer_norm_input_defs,
                             std::array{layer_norm_node.MutableOutputDefs()[0], &mask_index},
                             {}, kMSDomain);
  embed_layer_norm_node = &node;

  // Get attribute "epsilon" from "LayerNormalization" node if available. Else, default value
  // will be used.
  NodeAttributes ln_attrs = layer_norm_node.GetAttributes();
  NodeAttributes::const_iterator epsilon = ln_attrs.find("epsilon");
  if (epsilon != ln_attrs.end()) {
    embed_layer_norm_node->AddAttributeProto(epsilon->second);
  } else {
    embed_layer_norm_node->AddAttribute("epsilon", contrib::kDefaultEmbedLayerNormEpsilon);
  }

  // Assign provider to this new node. Provider should be same as the provider for old node.
  embed_layer_norm_node->SetExecutionProviderType(layer_norm_node.GetExecutionProviderType());
}

TEST(GraphParser, SimpleMatch1) {
  PatternType float_type(PatternTypeCategory::Float);
  PatternGraph target(
      {IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("X", float_type)},
      {PNode("Exp", {"X"}, {"Y"}, "Exp"),
       PNode("Add", {"Y", "C1"}, {"Z1"}, "gAdd"),
       PNode("Add", {"Y", "C2"}, {"Z2"}, "customize"),
       PNode("Sub", {"Z1", "Z2"}, {"W"}, "Sub")});
  PatternGraph pattern(
      {IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("X", float_type)},
      {PNode("Exp", {"X"}, {"Y"}, "Exp"),
       PNode("Add", {"Y", "C1"}, {"Z"}, "customize"),
       PNode("Sub", {"Z", "C2"}, {"W"}, "Sub")});

  class CustomNodeCompareFunc : public NodeCompareFunc {
   public:
    bool operator()(const Node* g, const PNode* p, const Graph& /*target*/,
                    const PatternGraph& /*pattern*/) const override {
      return p->NameEquals(g->Name());
    }
  };

  pattern.SetCustomConstraint(std::make_unique<CustomNodeCompareFunc>());
  Graph& target_graph = target.GetGraph();

  PatternMatchResult res(target_graph);
  ASSERT_TRUE(pattern.TryMatch(target_graph, res, "Exp").IsOK());
}

TEST(GraphParser, SimpleMatch2) {
  PatternType float_type(PatternTypeCategory::Float);
  PatternGraph target(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("C3", float_type),
       IArg("C4", float_type),
       IArg("X", float_type)},
      {PNode("ReduceMean", {"X"}, {"W1"}, "ex_rm1"),
       PNode("Sub", {"W1", "C4"}, {"W2"}, "ex_sub1"),

       PNode("ReduceMean", {"W2"}, {"Y"}, "g_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "g_sub1"),
       PNode("Sub", {"X", "Y"}, {"Sub2"}, "g_sub2"),
       PNode("Pow", {"Sub2", "C0"}, {"Pow"}, "g_pow"),

       PNode("Exp", {"Pow"}, {"ex_exp"}, "ex_exp"),
       PNode("Sqrt", {"ex_exp"}, {"ex_sqrt1"}, "ex_sqrt1"),

       PNode("ReduceMean", {"Pow"}, {"Z"}, "g_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "g_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "g_sqrt"),

       PNode("Sqrt", {"Sqrt"}, {"ex_sqrt2"}, "ex_sqrt2"),
       PNode("Add", {"ex_sqrt2", "ex_sqrt1"}, {"ex_add1"}, "ex_add1"),
       PNode("Sub", {"ex_add1", "ex_exp"}, {"ex_sub2"}, "ex_sub2"),

       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "g_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "g_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "g_final")});

  // The second pattern of layer norm fusion
  PatternGraph pattern(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("C3", float_type),
       IArg("X", float_type)},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Sub", {"X", "Y"}, {"Sub2"}, "p_sub2"),
       PNode("Pow", {"Sub2", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")});

  auto& target_graph = target.GetGraph();
  PatternMatchResult res(target_graph);
  ASSERT_TRUE(pattern.TryMatch(target_graph, res, "p_rm1").IsOK());
}

TEST(GraphParser, SimpleMatch3) {
  PatternType float_type(PatternTypeCategory::Float);
  PatternType integer_type(PatternTypeCategory::Integer);
  PatternGraph target(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("C3", integer_type),
       IArg("C4", integer_type),
       IArg("X", float_type)},
      {PNode("Shape", {"X"}, {"Y1"}, "g_shape"),
       PNode("Exp", {"X"}, {"P1"}, "ex_p1"),
       PNode("Sqrt", {"P1"}, {"P2"}, "ex_p2"),
       PNode("Log", {"P2"}, {"P3"}, "ex_p3"),
       PNode("Add", {"P2", "P3"}, {"P4"}, "ex_p4"),
       PNode("Gather", {"Y1", "C3"}, {"Y2"}, "g_gather"),
       PNode("Unsqueeze", {"Y2", "C4"}, {"Y3"}, "g_unsqueeze"),
       PNode("Concat", {"Y3"}, {"Y4"}, "g_concat", {""}, {}, {ONNX_NAMESPACE::MakeAttribute("axis", int64_t(0))}),
       PNode("MatMul", {"X", "C0"}, {"Z1"}, "g_matmul"),
       PNode("Add", {"Z1", "C1"}, {"Z2"}, "g_add"),
       PNode("Reshape", {"Z2", "Y4"}, {"oup"}, "g_reshape")});

  // The second pattern of layer norm fusion
  PatternGraph pattern(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", integer_type),
       IArg("C3", integer_type),
       IArg("X", float_type)},
      {PNode("Shape", {"X"}, {"Y1"}, "p_shape"),
       PNode("Gather", {"Y1", "C2"}, {"Y2"}, "p_gather"),
       PNode("Unsqueeze", {"Y2", "C3"}, {"Y3"}, "p_unsqueeze"),
       PNode("Concat", {"Y3"}, {"Y4"}, "p_concat", {""}, {}, {ONNX_NAMESPACE::MakeAttribute("axis", int64_t(0))}),
       PNode("MatMul", {"X", "C0"}, {"Z1"}, "p_matmul"),
       PNode("Add", {"Z1", "C1"}, {"Z2"}, "p_add"),
       PNode("Reshape", {"Z2", "Y4"}, {"oup"}, "p_reshape")});

  auto& target_graph = target.GetGraph();
  PatternMatchResult res(target_graph);
  ASSERT_TRUE(pattern.TryMatch(target_graph, res).IsOK());
}

TEST(GraphParser, SimpleReplace1) {
  PatternType float_type(PatternTypeCategory::Float);
  PatternGraph target(
      {IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("X", float_type)},
      {PNode("Exp", {"X"}, {"Y"}, "gExp"),
       PNode("Add", {"Y", "C1"}, {"Z1"}, "gAdd"),
       PNode("Mul", {"Y", "C2"}, {"Z2"}, "gMul"),
       PNode("Sub", {"Z1", "Z2"}, {"W"}, "gSub")});
  PatternGraph pattern(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("X2", float_type),
       IArg("X1", float_type)},
      {PNode("Mul", {"X1", "C0"}, {"Y1"}, "pMul"),
       PNode("Add", {"X2", "C1"}, {"Y2"}, "pAdd"),
       PNode("Sub", {"Y1", "Y2"}, {"Z"}, "pSub")});

  Graph& graph = target.GetGraph();
  ASSERT_TRUE(TryReplace(graph, pattern, PNode("Sqrt", {}, {}, "test123"), {{"pAdd", 0}}, {}, "pMul", "pSub").IsOK());
}

TEST(GraphParser, SimpleReplace2) {
  PatternType float_type(PatternTypeCategory::Float);
  PatternGraph target(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("C3", float_type),
       IArg("C4", float_type),
       IArg("X", float_type)},
      {PNode("ReduceMean", {"X"}, {"W1"}, "ex_rm1"),
       PNode("Sub", {"W1", "C4"}, {"W2"}, "ex_sub1"),

       PNode("ReduceMean", {"W2"}, {"Y"}, "g_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "g_sub1"),
       PNode("Sub", {"X", "Y"}, {"Sub2"}, "g_sub2"),
       PNode("Pow", {"Sub2", "C0"}, {"Pow"}, "g_pow"),

       PNode("Exp", {"Pow"}, {"ex_exp"}, "ex_exp"),
       PNode("Sqrt", {"ex_exp"}, {"ex_sqrt1"}, "ex_sqrt1"),

       PNode("ReduceMean", {"Pow"}, {"Z"}, "g_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "g_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "g_sqrt"),

       PNode("Sqrt", {"Sqrt"}, {"ex_sqrt2"}, "ex_sqrt2"),
       PNode("Add", {"ex_sqrt2", "ex_sqrt1"}, {"ex_add1"}, "ex_add1"),
       PNode("Sub", {"ex_add1", "ex_exp"}, {"ex_sub2"}, "ex_sub2"),

       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "g_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "g_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "g_final")});

  // The second pattern of layer norm fusion
  PatternGraph pattern(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("C3", float_type),
       IArg("X", float_type)},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Sub", {"X", "Y"}, {"Sub2"}, "p_sub2"),
       PNode("Pow", {"Sub2", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")});

  auto& target_graph = target.GetGraph();
  ASSERT_TRUE(TryReplace(target_graph, pattern, PNode("Sqrt", {}, {}, "test123"), {{"p_rm1", 0}}, {}, "p_rm1", "p_final").IsOK());
}

TEST(GraphParser, LayerNormFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/layer_norm.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternType float_type({ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE});
  PatternGraph pattern1(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("C3", float_type),
       IArg("X", float_type)},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Sub", {"X", "Y"}, {"Sub2"}, "p_sub2"),
       PNode("Pow", {"Sub2", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")});

  PatternGraph pattern2(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("C3", float_type),
       IArg("X", float_type)},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Pow", {"Sub1", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")});

  PatternGraph pattern3(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("C3", float_type),
       IArg("X", float_type)},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Cast", {"Sub1"}, {"Cast"}, "p_cast", {""}, {}, {ONNX_NAMESPACE::MakeAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE})}),
       PNode("Pow", {"Cast", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")});

  PatternGraph pattern4(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type, 1, false),
       IArg("C3", float_type, 1, false),
       IArg("X", float_type)},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Cast", {"C0"}, {"Cast"}, "p_cast", {""}, {}, {ONNX_NAMESPACE::MakeAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE})}),
       PNode("Pow", {"Cast", "Sub1"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")});

  class CustomArgCompareFunc : public ArgCompareFunc {
   public:
    bool operator()(const NodeArg* g, const IArg*, const Graph& target,
                    const PatternGraph& /*pattern*/) const override {
      return graph_utils::NodeArgIsConstant(target, *g) || graph_utils::IsGraphInput(target, g);
    }
  };

  pattern2.SetCustomConstraint(std::make_unique<CustomArgCompareFunc>(), "C2").SetCustomConstraint(std::make_unique<CustomArgCompareFunc>(), "C3");
  bool match = false;
  if (!match) {
    match = TryReplace(graph, pattern1, PNode("LayerNormalization", {}, {}, "replaced_node"), {{"p_rm1", 0}, {"p_mul", 1}, {"p_final", 1}}, {}, "p_rm1", "p_final").IsOK();
  }
  if (!match) {
    match = TryReplace(graph, pattern2, PNode("LayerNormalization", {}, {}, "replaced_node"), {{"p_rm1", 0}, {"p_mul", 0}, {"p_final", 1}}, {}, "p_rm1", "p_final").IsOK();
  }
  if (!match) {
    match = TryReplace(graph, pattern3, PNode("LayerNormalization", {}, {}, "replaced_node"), {{"p_rm1", 0}, {"p_mul", 1}, {"p_final", 1}}, {}, "p_rm1", "p_final").IsOK();
  }
  if (!match) {
    match = TryReplace(graph, pattern4, PNode("LayerNormalization", {}, {}, "replaced_node"), {{"p_rm1", 0}, {"p_mul", 1}, {"p_final", 1}}, {}, "p_rm1", "p_final").IsOK();
  }
  ASSERT_TRUE(match);

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Div"], 0);
  ASSERT_EQ(op_to_count["Add"], 0);
  ASSERT_EQ(op_to_count["Sub"], 0);
  ASSERT_EQ(op_to_count["ReduceMean"], 0);
  ASSERT_EQ(op_to_count["Pow"], 0);
  ASSERT_EQ(op_to_count["Sqrt"], 0);
  ASSERT_EQ(op_to_count["LayerNormalization"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "LayerNormalization") {
      // LayerNormalization should have three inputs.
      EXPECT_EQ(node.InputDefs().size(), 3u) << "LayerNormalization number of inputs does not equal to 3. Got:" << node.InputDefs().size();
      // LayerNormalization input "scale" and "bias" should have the same dimension.
      const ONNX_NAMESPACE::TensorShapeProto* scale_shape = node.InputDefs()[1]->Shape();
      const ONNX_NAMESPACE::TensorShapeProto* bias_shape = node.InputDefs()[2]->Shape();
      EXPECT_EQ(scale_shape->dim_size(), 1) << "LayerNormalization scale should be 1D. Got: " << scale_shape->dim_size();
      EXPECT_EQ(bias_shape->dim_size(), 1) << "LayerNormalization bias should be 1D. Got: " << bias_shape->dim_size();
      EXPECT_EQ(scale_shape->dim(0).dim_value(), bias_shape->dim(0).dim_value());
    } else {
      EXPECT_TRUE(false) << "Unexpected node " << node.Name();
    }
  }
}

TEST(GraphParser, LayerNormWithCastFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/layer_norm_with_cast.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternType float_type({ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE});
  PatternGraph pattern1(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("C3", float_type),
       IArg("X", float_type)},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Cast", {"Sub1"}, {"Cast"}, "p_cast", {""}, {}, {ONNX_NAMESPACE::MakeAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE})}),
       PNode("Pow", {"Cast", "C0"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")});

  PatternGraph pattern2(
      {IArg("C0", float_type),
       IArg("C1", float_type),
       IArg("C2", float_type),
       IArg("C3", float_type),
       IArg("C4", float_type),
       IArg("X", float_type)},
      {PNode("ReduceMean", {"X"}, {"Y"}, "p_rm1"),
       PNode("Sub", {"X", "Y"}, {"Sub1"}, "p_sub1"),
       PNode("Cast", {"C4"}, {"Cast"}, "p_cast", {""}, {}, {ONNX_NAMESPACE::MakeAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE})}),
       PNode("Pow", {"Cast", "Sub1"}, {"Pow"}, "p_pow"),
       PNode("ReduceMean", {"Pow"}, {"Z"}, "p_rm2"),
       PNode("Add", {"C1", "Z"}, {"Add1"}, "p_add1"),
       PNode("Sqrt", {"Add1"}, {"Sqrt"}, "p_sqrt"),
       PNode("Div", {"Sub1", "Sqrt"}, {"Div"}, "p_div"),
       PNode("Mul", {"Div", "C2"}, {"Mul"}, "p_mul"),
       PNode("Add", {"Mul", "C3"}, {"Final"}, "p_final")});

  bool match = false;
  if (!match) {
    std::vector<PatternMatchPair> res;
    match = TryReplace(graph, pattern1, PNode("LayerNormalization", {}, {}, "replaced_node"), {{"p_rm1", 0}, {"p_mul", 1}, {"p_final", 1}}, {}, "p_rm1", "p_final").IsOK();
  }
  if (!match) {
    std::vector<PatternMatchPair> res;
    match = TryReplace(graph, pattern2, PNode("LayerNormalization", {}, {}, "replaced_node"), {{"p_rm1", 0}, {"p_mul", 0}, {"p_final", 1}}, {}, "p_rm1", "p_final").IsOK();
  }
  ASSERT_TRUE(match);

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 1);
}

TEST(GraphParser, NoopEliminationTest) {
  auto model_uri = MODEL_FOLDER "noop-add.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternType float_type(PatternTypeCategory::Float);
  PatternGraph pattern1(
      {IArg("C0", float_type),
       IArg("X", float_type)},
      {PNode("Add", {"X", "C0"}, {"Y"}, "p_add")});

  class CustomNodeCompareFunc : public NodeCompareFunc {
   public:
    bool operator()(const Node* g, const PNode*, const Graph& target,
                    const PatternGraph& /*pattern*/) const override {
      if (g->OpType() != "Add") return false;
      bool input0_is_initializer = graph_utils::IsConstantInitializer(target, g->InputDefs()[0]->Name());
      bool input1_is_initializer = graph_utils::IsConstantInitializer(target, g->InputDefs()[1]->Name());

      // reject if both or neither inputs are initializers for now
      if (input0_is_initializer == input1_is_initializer) {
        return false;
      }

      const auto* initializer = graph_utils::GetConstantInitializer(target, g->InputDefs()[input0_is_initializer ? 0 : 1]->Name());

      // if initializer_rank is bigger, the output is expected to be initializer_rank per broadcasting rule,
      // but it won't happen if the case is accepted, thus reject it
      auto initializer_rank = initializer->dims().size();
      const auto* other_input_shape = g->InputDefs()[input0_is_initializer ? 1 : 0]->Shape();
      if (other_input_shape == nullptr || initializer_rank > other_input_shape->dim_size()) {
        return false;
      }

      int32_t data_type = initializer->data_type();
      Initializer add_init(*initializer, target.ModelPath());
      if (add_init.size() > 1) {
        return false;
      }
      // handle edge case where the total size of the initializer is 0
      if (add_init.size() == 0) {
        return true;
      }
      switch (data_type) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          if (*add_init.data<float>() != 0.f) {
            return false;
          }
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
          if (math::halfToFloat(add_init.data<MLFloat16>()->val) != 0.f) {
            return false;
          }
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
          if (*add_init.data<double>() != static_cast<double>(0.f)) {
            return false;
          }
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_INT32:
          if (*add_init.data<int32_t>() != static_cast<int32_t>(0)) {
            return false;
          }
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_INT64:
          if (*add_init.data<int64_t>() != static_cast<int64_t>(0)) {
            return false;
          }
          break;
        default:
          return false;
      }

      // reject node output is graph output for now
      if (!graph_utils::CanRemoveNode(target, *g, logging::LoggingManager::DefaultLogger())) {
        return false;
      }

      return true;
    }
  };

  pattern1.SetCustomConstraint(std::make_unique<CustomNodeCompareFunc>(), "p_add");
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 5);

  bool match = true;
  while (match) {
    PatternMatchResult res(graph);
    auto st = pattern1.TryMatch(graph, res);
    match = st.IsOK();
    std::cout << st.ToString() << std::endl;
    for (auto [name, pair] : res.GetMatchMap()) {
      auto gnode = graph.GetNode(pair.first);
      if (graph_utils::CanRemoveNode(graph, *gnode, logging::LoggingManager::DefaultLogger())) {
        ASSERT_TRUE(graph_utils::RemoveNode(graph, *gnode));
      }
    }
  }

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
}

TEST(GraphParser, ReshapeFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternType float_type(PatternTypeCategory::Float);
  PatternType integer_type(PatternTypeCategory::Integer);
  PatternGraph pattern(
      {IArg("X", float_type),
       IArg("C0", integer_type),
       IArg("C1", float_type),
       IArg("C2", integer_type),
       IArg("C3", integer_type),
       IArg("C4", integer_type)},
      {PNode("Shape", {"X"}, {"Shape1"}, "p_shape1"),
       PNode("Gather", {"Shape1", "C0"}, {"Gather1"}, "p_gather1"),
       PNode("Unsqueeze", {"Gather1", "C3"}, {"Unsqueeze1"}, "p_unsqueeze1"),
       PNode("Shape", {"C1"}, {"Shape2"}, "p_shape2"),
       PNode("Gather", {"Shape2", "C2"}, {"Gather2"}, "p_gather2"),
       PNode("Unsqueeze", {"Gather2", "C4"}, {"Unsqueeze2"}, "p_unsqueeze2"),
       PNode("Concat", {"Unsqueeze1", "Unsqueeze2"}, {"Concat"}, "p_concat", {""}, {}, {ONNX_NAMESPACE::MakeAttribute("axis", int64_t(0))}),
       PNode("Reshape", {"X", "Concat"}, {"Y"}, "p_reshape")});

  class CustomNodeCompareFunc : public NodeCompareFunc {
   public:
    bool operator()(const Node* g, const PNode* p, const Graph& target,
                    const PatternGraph& /*pattern*/) const override {
      auto GetConstantInitializerCount = [](const Graph& graph, const Node* node) {
        int res = 0;
        for (size_t i = 0; i < node->InputDefs().size(); i++) {
          res += graph_utils::IsConstantInitializer(graph, node->InputDefs()[i]->Name());
        }
        return res;
      };

      if (!p->OpTypeEquals(g->OpType())) return false;
      if (p->NameEquals("p_concat")) {
        return GetConstantInitializerCount(target, g) == 2;
      } else if (p->NameEquals("p_gather1") && !optimizer_utils::IsInitializerWithExpectedValue(target, *(g->InputDefs()[1]), int64_t(0), false)) {
        return false;
      } else if (p->NameEquals("p_gather2") && !optimizer_utils::IsInitializerWithExpectedValue(target, *(g->InputDefs()[1]), int64_t(1), false)) {
        return false;
      } else if (p->OpTypeEquals("Unqueeze")) {
        InlinedVector<int64_t> axes;
        if (!(graph_utils::GetRepeatedNodeAttributeValues(*g, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
          return false;
        }
      }
      return true;
    }
  };

  pattern.SetCustomConstraint(std::make_unique<CustomNodeCompareFunc>());

  PatternMatchResult res(graph);
  ASSERT_TRUE(pattern.TryMatch(graph, res).IsOK());

  // replace
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
  auto x = res.GetTargetNodeWithName("p_reshape")->MutableInputDefs()[0];
  std::vector<NodeArg*> input_args({x});
  InlinedVector<int64_t> shape_value;
  auto concat_node = res.GetTargetNodeWithName("p_concat");
  const auto* shape_def = concat_node->OutputDefs()[0];
  for (auto def : concat_node->InputDefs()) {
    if (graph_utils::IsInitializer(graph, def->Name(), false)) {
      optimizer_utils::AppendTensorFromInitializer(graph, *def, shape_value, true);
    } else {
      shape_value.push_back(0);
    }
  }
  ONNX_NAMESPACE::TensorProto shape_initializer_proto;
  shape_initializer_proto.set_name(shape_def->Name());
  shape_initializer_proto.add_dims(static_cast<int64_t>(shape_value.size()));
  shape_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  shape_initializer_proto.set_raw_data(shape_value.data(), shape_value.size() * sizeof(int64_t));
  auto& new_node_arg = graph_utils::AddInitializer(graph, shape_initializer_proto);
  input_args.push_back(&new_node_arg);
  nodes_to_remove.push_back(*res.GetTargetNodeWithName("p_shape1"));
  auto mapping = res.GetMatchMap();
  for (auto iter = mapping.begin(); iter != mapping.end(); iter++) {
    if (iter->first != "p_shape1" && iter->first != "p_reshape") {
      nodes_to_remove.push_back(*graph.GetNode(iter->second.first));
    }
  }
  nodes_to_remove.push_back(*res.GetTargetNodeWithName("p_reshape"));
  Node& replace_node = graph.AddNode("replace", "Reshape", "", input_args, {}, {}, "");
  graph_utils::FinalizeNodeFusion(graph, nodes_to_remove, replace_node);

  // verify
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  ASSERT_TRUE(op_to_count["Concat"] == 0);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 4);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], 12);
      EXPECT_EQ(val[3], 64);
    }
  }
}

bool IsSupportedOptypeVersionAndDomain(const Node& node, const std::string& op_type,
                                       std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> versions,
                                       std::string_view domain) {
  return (node.OpType() == op_type && graph_utils::MatchesOpSinceVersion(node, versions) &&
          graph_utils::MatchesOpSetDomain(node, domain));
}
bool IsFusableActivation(const Node& node) {
  return IsSupportedOptypeVersionAndDomain(node, "Elu", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "HardSigmoid", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "LeakyRelu", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Relu", {6, 13, 14}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Selu", {6}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Sigmoid", {6, 13}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Softplus", {1}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Softsign", {1}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "Tanh", {6, 13}, kOnnxDomain) ||
#ifndef DISABLE_CONTRIB_OPS
         IsSupportedOptypeVersionAndDomain(node, "ScaledTanh", {1}, kOnnxDomain) ||
         IsSupportedOptypeVersionAndDomain(node, "ParametricSoftplus", {1}, kOnnxDomain) ||
#endif
         IsSupportedOptypeVersionAndDomain(node, "ThresholdedRelu", {1, 10}, kOnnxDomain);
}

TEST(GraphParser, GemmActivationFusionTest1) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/gemm_relu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternType float_type(PatternTypeCategory::Float);
  PatternGraph pattern(
      {IArg("X", float_type, {1, 1}),
       IArg("C0", float_type, {1, 1}),
       IArg("C1", float_type, {1, 1})},
      {PNode("Gemm", {"X", "C0", "C1"}, {"Y"}, "p_gemm"),
       PNode("Sqrt", {"Y"}, {"Z"}, "p_active")});

  class CustomNodeCompareFunc : public NodeCompareFunc {
   public:
    bool operator()(const Node* g, const PNode*, const Graph& /*terget*/,
                    const PatternGraph& /*pattern*/) const override {
      return IsFusableActivation(*g);
    }
  };
  pattern.SetCustomConstraint(std::make_unique<CustomNodeCompareFunc>(), "p_active");

  PatternMatchResult res(graph);
  ASSERT_TRUE(pattern.TryMatch(graph, res).IsOK());

  // replace
  ASSERT_EQ(TryReplace(graph, pattern, PNode("FusedGemm", {}, {}, "replaced_node"), {{"p_gemm", 0}, {"p_gemm", 1}, {"p_gemm", 2}}, {}, "p_gemm", "p_active").ToString(), "OK");

  // verify
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 0);
}

TEST(GraphParser, GemmActivationFusionTest2) {
  auto model_uri = MODEL_FOLDER "gemm_activation_fusion/gemm_activation_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternType float_type(PatternTypeCategory::Float);
  PatternGraph pattern(
      {IArg("X", float_type, {1, 1}),
       IArg("C0", float_type, {1, 1}),
       IArg("C1", float_type, {1, 1})},
      {PNode("Gemm", {"X", "C0", "C1"}, {"Y"}, "p_gemm"),
       PNode("Sqrt", {"Y"}, {"Z"}, "p_active")});

  class CustomNodeCompareFunc : public NodeCompareFunc {
   public:
    bool operator()(const Node* g, const PNode*, const Graph& /*target*/,
                    const PatternGraph& /*pattern*/) const override {
      return IsFusableActivation(*g);
    }
  };
  pattern.SetCustomConstraint(std::make_unique<CustomNodeCompareFunc>(), "p_active");

  PatternMatchResult res(graph);
  ASSERT_TRUE(pattern.TryMatch(graph, res).IsOK());

  // replace
  ASSERT_EQ(TryReplace(graph, pattern, PNode("FusedGemm", {}, {}, "replaced_node", {"com.microsoft"}), {{"p_gemm", 0}, {"p_gemm", 1}, {"p_gemm", 2}}, {}, "p_gemm", "p_active").ToString(), "OK");

  // verify
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["LeakyRelu"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedGemm"] == 1);
}

TEST(GraphParser, EmbedLayerNormFusionTest1) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternType float_type(PatternTypeCategory::Float);
  PatternType integer_type(PatternTypeCategory::Integer);
  PatternGraph pattern(
      {IArg("X", float_type, 3),
       IArg("C0", integer_type),
       IArg("C1", integer_type),
       IArg("C2", integer_type),
       IArg("C3", float_type),
       IArg("C4", integer_type),
       IArg("C5", float_type)},
      {PNode("Gather", {"X", "C0"}, {"Gather1"}, "p_gather1"),
       PNode("Gather", {"X", "C1"}, {"Gather2"}, "p_gather2"),
       PNode("Add", {"Gather1", "Gather2"}, {"Add1"}, "p_add1"),
       PNode("Gather", {"X", "C2"}, {"Gather3"}, "p_gather3"),
       PNode("Add", {"Add1", "Gather3"}, {"Add2"}, "p_add2"),
       PNode("LayerNormalization", {"Add2", "C3"}, {"LayerNorm"}, "p_layernorm"),
       PNode("ReduceSum", {"X"}, {"ReduceSum"}, "p_reducesum"),
       PNode("Attention", {"ReduceSum", "LayerNorm", "C5"}, {"Y"}, "p_attention", {"com.microsoft"}, {1}, {ONNX_NAMESPACE::MakeAttribute("num_heads", int64_t{1})})});
  PatternMatchResult res(graph);
  ASSERT_TRUE(pattern.TryMatch(graph, res).IsOK());

  // replace
  std::vector<NodeIndex> nodes_to_remove;
  auto layer_norm_node = res.GetTargetNodeWithName("p_layernorm");
  auto layer_norm_add_node = res.GetTargetNodeWithName("p_add2");
  auto segment_gather_node = res.GetTargetNodeWithName("p_gather3");
  NodeArg* segment_embedding = segment_gather_node->MutableInputDefs()[0];
  auto add_node = res.GetTargetNodeWithName("p_add1");
  auto word_gather_node = res.GetTargetNodeWithName("p_gather1");
  NodeArg* word_embedding = word_gather_node->MutableInputDefs()[0];
  NodeArg* input_ids = word_gather_node->MutableInputDefs()[1];
  NodeArg* segment_ids = segment_gather_node->MutableInputDefs()[1];
  auto input_shape = input_ids->Shape();
  ASSERT_FALSE(input_shape->dim_size() != 2 ||
               !utils::HasDimValue(input_shape->dim()[0]) ||
               !utils::HasDimValue(input_shape->dim()[1]));
  int64_t batch_size = input_shape->dim()[0].dim_value();
  int64_t sequence_length = input_shape->dim()[1].dim_value();
  ASSERT_FALSE(batch_size <= 0 || sequence_length <= 0);
  auto sg_shape = segment_embedding->Shape();
  ASSERT_FALSE(sg_shape == nullptr || sg_shape->dim_size() != 2 ||
               !utils::HasDimValue(sg_shape->dim()[1]) ||
               sg_shape->dim()[1].dim_value() <= 0);
  auto hidden_size = sg_shape->dim()[1].dim_value();
  auto add_input_name = add_node->MutableInputDefs()[1]->Name();
  const ONNX_NAMESPACE::TensorProto* position_embed_tensor;
  NodeArg* position_embedding = nullptr;
  if (graph_utils::IsConstantInitializer(graph, add_input_name)) {
    ASSERT_TRUE(graph.GetInitializedTensor(add_input_name, position_embed_tensor));
    position_embedding = ExtractEmbedding(graph, batch_size, sequence_length, hidden_size, position_embed_tensor);
  } else {
    auto position_gather_node = res.GetTargetNodeWithName("p_gather2");
    position_embedding = position_gather_node->MutableInputDefs()[0];
    nodes_to_remove.push_back(position_gather_node->Index());
  }
  Node* replace_node = nullptr;
  CreateEmbedLayernormNode(graph, input_ids, segment_ids, word_embedding, position_embedding, segment_embedding,
                           *layer_norm_node, replace_node);
  if (!nodes_to_remove.empty()) {
    graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *graph.GetNode(nodes_to_remove[0]));
  }

  nodes_to_remove.clear();

  nodes_to_remove.push_back(word_gather_node->Index());
  nodes_to_remove.push_back(segment_gather_node->Index());
  nodes_to_remove.push_back(add_node->Index());
  nodes_to_remove.push_back(layer_norm_add_node->Index());
  nodes_to_remove.push_back(layer_norm_node->Index());

  for (const NodeIndex index : nodes_to_remove) {
    Node* node = graph.GetNode(index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["ReduceSum"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.Attention"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.EmbedLayerNormalization"] == 1);
}

TEST(GraphParser, EmbedLayerNormFusionTest2) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format2.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternType float_type(PatternTypeCategory::Float);
  PatternType integer_type(PatternTypeCategory::Integer);
  PatternGraph pattern(
      {IArg("X", float_type, 3),
       IArg("C0", integer_type),
       IArg("C1", float_type)},
      {PNode("Shape", {"X"}, {"Shape1"}, "p_shape1", {}, {1}),
       PNode("Gather", {"Shape1", "C0"}, {"Gather1"}, "p_gather1"),
       PNode("Unsqueeze", {"Gather1", "C0"}, {"Unsqueeze1"}, "p_unsqueeze1", {}, {1}),
       PNode("Shape", {"Unsqueeze1"}, {"COS"}, "p_cos"),
       PNode("NonZero", {"COS"}, {"NonZero"}, "p_nonzero"),
       PNode("Transpose", {"NonZero"}, {"Transpose"}, "p_transpose", {}, {1}),
       PNode("Squeeze", {"Transpose", "C0"}, {"Squeeze"}, "p_squeeze", {}, {1}),
       PNode("Cast", {"Squeeze"}, {"Cast1"}, "p_cast1", {""}, {}, {ONNX_NAMESPACE::MakeAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT})}),
       PNode("Unsqueeze", {"Cast1", "C0"}, {"Unsqueeze2"}, "p_unsqueeze2", {}, {1}),
       PNode("Shape", {"X"}, {"Shape2"}, "p_shape2", {}, {1}),
       PNode("Expand", {"Unsqueeze2", "Shape2"}, {"Expand"}, "p_expand", {}, {8}),
       PNode("Gather", {"Expand", "C0"}, {"Gather2"}, "p_gather2"),
       PNode("Gather", {"X", "C0"}, {"Gather3"}, "p_gather3"),
       PNode("Add", {"Gather2", "Gather3"}, {"Add1"}, "p_add1"),
       PNode("Gather", {"X", "C0"}, {"Gather4"}, "p_gather4"),
       PNode("Add", {"Add1", "Gather4"}, {"Add2"}, "p_add2"),
       PNode("LayerNormalization", {"Add2", "C1"}, {"LayerNorm"}, "p_layernorm"),
       PNode("ReduceSum", {"X"}, {"ReduceSum"}, "p_reducesum", {}, {1}),
       PNode("Attention", {"ReduceSum", "LayerNorm", "C1"}, {"Y"}, "p_attention", {"com.microsoft"}, {1}, {ONNX_NAMESPACE::MakeAttribute("num_heads", int64_t{1})})});

  class CustomNodeCompareFunc : public NodeCompareFunc {
   public:
    bool operator()(const Node* g, const PNode*, const Graph& /*target*/,
                    const PatternGraph& /*pattern*/) const override {
      return g->OpType() == "ConstantOfShape";
    }
  };
  pattern.SetCustomConstraint(std::make_unique<CustomNodeCompareFunc>(), "p_cos");

  PatternMatchResult res(graph);
  ASSERT_TRUE(pattern.TryMatch(graph, res).IsOK());

  // replace
  std::vector<NodeIndex> nodes_to_remove;
  auto layer_norm_node = res.GetTargetNodeWithName("p_layernorm");
  auto layer_norm_add_node = res.GetTargetNodeWithName("p_add2");
  auto segment_gather_node = res.GetTargetNodeWithName("p_gather4");
  NodeArg* segment_embedding = segment_gather_node->MutableInputDefs()[0];
  auto add_node = res.GetTargetNodeWithName("p_add1");
  auto word_gather_node = res.GetTargetNodeWithName("p_gather3");
  NodeArg* word_embedding = word_gather_node->MutableInputDefs()[0];
  NodeArg* input_ids = word_gather_node->MutableInputDefs()[1];
  NodeArg* segment_ids = segment_gather_node->MutableInputDefs()[1];
  auto input_shape = input_ids->Shape();
  ASSERT_FALSE(input_shape->dim_size() != 2 ||
               !utils::HasDimValue(input_shape->dim()[0]) ||
               !utils::HasDimValue(input_shape->dim()[1]));
  int64_t batch_size = input_shape->dim()[0].dim_value();
  int64_t sequence_length = input_shape->dim()[1].dim_value();
  ASSERT_FALSE(batch_size <= 0 || sequence_length <= 0);
  auto sg_shape = segment_embedding->Shape();
  ASSERT_FALSE(sg_shape == nullptr || sg_shape->dim_size() != 2 ||
               !utils::HasDimValue(sg_shape->dim()[1]) ||
               sg_shape->dim()[1].dim_value() <= 0);
  auto hidden_size = sg_shape->dim()[1].dim_value();
  auto add_input_name = add_node->MutableInputDefs()[1]->Name();
  const ONNX_NAMESPACE::TensorProto* position_embed_tensor;
  NodeArg* position_embedding = nullptr;
  if (graph_utils::IsConstantInitializer(graph, add_input_name)) {
    ASSERT_TRUE(graph.GetInitializedTensor(add_input_name, position_embed_tensor));
    position_embedding = ExtractEmbedding(graph, batch_size, sequence_length, hidden_size, position_embed_tensor);
  } else {
    auto position_gather_node = res.GetTargetNodeWithName("p_gather2");
    position_embedding = position_gather_node->MutableInputDefs()[0];
    nodes_to_remove.push_back(position_gather_node->Index());
  }
  Node* replace_node = nullptr;
  CreateEmbedLayernormNode(graph, input_ids, segment_ids, word_embedding, position_embedding, segment_embedding,
                           *layer_norm_node, replace_node);
  if (!nodes_to_remove.empty()) {
    graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *graph.GetNode(nodes_to_remove[0]));
  }

  nodes_to_remove.clear();

  nodes_to_remove.push_back(word_gather_node->Index());
  nodes_to_remove.push_back(segment_gather_node->Index());
  nodes_to_remove.push_back(add_node->Index());
  nodes_to_remove.push_back(layer_norm_add_node->Index());
  nodes_to_remove.push_back(layer_norm_node->Index());

  for (const NodeIndex index : nodes_to_remove) {
    Node* node = graph.GetNode(index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["ReduceSum"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.Attention"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.EmbedLayerNormalization"] == 1);
}

TEST(GraphParser, EmbedLayerNormFusionTest3) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format3.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternGraph pattern(
      {IArg("X", PatternType(PatternTypeCategory::Float), 3),
       IArg("C0", PatternType(PatternTypeCategory::Integer)),
       IArg("C1", PatternType(PatternTypeCategory::Float))},
      {PNode("Shape", {"X"}, {"Shape1"}, "p_shape1"),
       PNode("Gather", {"Shape1", "C0"}, {"Gather1"}, "p_gather1"),
       PNode("Cast", {"Gather1"}, {"Cast1"}, "p_cast1", {""}, {}, {ONNX_NAMESPACE::MakeAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT})}),
       PNode("Range", {"Cast1", "Cast1", "Cast1"}, {"Range"}, "p_range"),
       PNode("Unsqueeze", {"Range", "C0"}, {"Unsqueeze1"}, "p_unsqueeze1"),
       PNode("Add", {"Gather1", "C0"}, {"Unsqueeze2"}, "p_unsqueeze2"),
       PNode("Shape", {"X"}, {"Shape2"}, "p_shape2"),
       PNode("Gather", {"Shape2", "C0"}, {"Gather2"}, "p_gather2"),
       PNode("Add", {"Gather2", "C0"}, {"Unsqueeze3"}, "p_unsqueeze3"),
       PNode("Concat", {"Unsqueeze2", "Unsqueeze3"}, {"Concat"}, "p_concat", {}, {}, {ONNX_NAMESPACE::MakeAttribute("axis", int64_t(0))}),
       PNode("Expand", {"Unsqueeze1", "Concat"}, {"Expand"}, "p_expand"),
       PNode("Gather", {"Expand", "C0"}, {"Gather3"}, "p_gather3"),
       PNode("Gather", {"X", "C0"}, {"Gather4"}, "p_gather4"),
       PNode("Add", {"Gather3", "Gather4"}, {"Add1"}, "p_add1"),
       PNode("Gather", {"X", "C0"}, {"Gather5"}, "p_gather5"),
       PNode("Add", {"Add1", "Gather5"}, {"Add2"}, "p_add2"),
       PNode("LayerNormalization", {"Add2", "C1"}, {"LayerNorm"}, "p_layernorm"),
       PNode("ReduceSum", {"X"}, {"ReduceSum"}, "p_reducesum"),
       PNode("Attention", {"ReduceSum", "LayerNorm", "C1"}, {"Attention"}, "p_attention", {"com.microsoft"}, {}, {ONNX_NAMESPACE::MakeAttribute("num_heads", int64_t{1})})});

  class CustomNodeCompareFunc : public NodeCompareFunc {
   public:
    bool operator()(const Node* g, const PNode*, const Graph& /*target*/,
                    const PatternGraph& /*pattern*/) const override {
      return g->OpType() == "Unsqueeze";
    }
  };
  pattern.SetCustomConstraint(std::make_unique<CustomNodeCompareFunc>(), "p_unsqueeze2");
  pattern.SetCustomConstraint(std::make_unique<CustomNodeCompareFunc>(), "p_unsqueeze3");

  PatternMatchResult res(graph);
  ASSERT_TRUE(pattern.TryMatch(graph, res).IsOK());
}

TEST(GraphParser, AttentionFusionInt32Test) {
  auto model_uri = MODEL_FOLDER "fusion/attention_int32_mask.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternGraph pattern(
      {IArg("X", PatternType(PatternTypeCategory::Float)),
       IArg("C0", PatternType(PatternTypeCategory::Integer)),
       IArg("C1", PatternType(PatternTypeCategory::Float))},
      {PNode("LayerNormalization", {"X", "C1", "C1"}, {"LayerNorm"}, "p_layernorm"),
       PNode("MatMul", {"LayerNorm", "C1"}, {"Matmul_q"}, "p_matmul_q"),
       PNode("MatMul", {"LayerNorm", "C1"}, {"Matmul_k"}, "p_matmul_k"),
       PNode("MatMul", {"LayerNorm", "C1"}, {"Matmul_v"}, "p_matmul_v"),
       PNode("Add", {"Matmul_q", "C1"}, {"Add_q"}, "p_add_q"),
       PNode("Add", {"Matmul_k", "C1"}, {"Add_k"}, "p_add_k"),
       PNode("Add", {"Matmul_v", "C1"}, {"Add_v"}, "p_add_v"),
       PNode("Reshape", {"Add_q", "C0"}, {"Reshape_q"}, "p_reshape_q"),
       PNode("Reshape", {"Add_k", "C0"}, {"Reshape_k"}, "p_reshape_k"),
       PNode("Reshape", {"Add_v", "C0"}, {"Reshape_v"}, "p_reshape_v"),
       PNode("Transpose", {"Reshape_q"}, {"Transpose_q"}, "p_transpose_q"),
       PNode("Transpose", {"Reshape_k"}, {"Transpose_k"}, "p_transpose_k"),
       PNode("Transpose", {"Reshape_v"}, {"Transpose_v"}, "p_transpose_v"),
       PNode("MatMul", {"Transpose_q", "Transpose_k"}, {"Matmul_qk"}, "p_matmul_qk"),
       PNode("Div", {"Matmul_qk", "C1"}, {"Div_qk"}, "p_div_qk"),
       PNode("Add", {"Div_qk", "C1"}, {"Mask_add"}, "p_mask_add"),
       PNode("Softmax", {"Mask_add"}, {"Softmax"}, "p_softmax"),
       PNode("MatMul", {"Softmax", "Transpose_v"}, {"Matmul_qkv"}, "p_matmul_qkv"),
       PNode("Transpose", {"Matmul_qkv"}, {"Transpsoe"}, "p_transpose"),
       PNode("Reshape", {"Transpsoe", "C0"}, {"Reshape"}, "p_reshape"),
       PNode("MatMul", {"Reshape", "C1"}, {"Matmul"}, "p_matmul"),
       PNode("Add", {"Matmul", "C1"}, {"Add1"}, "p_add1"),
       PNode("Add", {"Add1", "LayerNorm"}, {"final"}, "p_add2")});

  PatternMatchResult res(graph);
  ASSERT_TRUE(pattern.TryMatch(graph, res).IsOK());
}

TEST(GraphParser, FastGeluWithBiasFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu_with_bias.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternGraph pattern(
      {IArg("X", PatternType(PatternTypeCategory::Float)),
       IArg("C0", PatternType(PatternTypeCategory::Integer)),
       IArg("C1", PatternType(PatternTypeCategory::Float))},
      {PNode("Identity", {"X"}, {"Identity1"}, "p_identity1"),
       PNode("Add", {"Identity1", "C1"}, {"Add1"}, "p_add1"),
       PNode("Mul", {"Add1", "C1"}, {"Mul1"}, "p_mul1"),
       PNode("Mul", {"Add1", "Mul1"}, {"Mul2"}, "p_mul2"),
       PNode("Add", {"Mul2", "C1"}, {"Add2"}, "p_add2"),
       PNode("Mul", {"Add1", "C1"}, {"Mul3"}, "p_mul3"),
       PNode("Mul", {"Add2", "Mul3"}, {"Mul4"}, "p_mul4"),
       PNode("Tanh", {"Mul4"}, {"Tanh"}, "p_tanh"),
       PNode("Add", {"Tanh", "C1"}, {"Add3"}, "p_add3"),
       PNode("Mul", {"Add1", "C1"}, {"Mul5"}, "p_mul5"),
       PNode("Mul", {"Mul5", "Add3"}, {"Mul6"}, "p_mul6"),
       PNode("Identity", {"Mul6"}, {"Identity2"}, "p_identity2")});

  PatternMatchResult res(graph);
  ASSERT_TRUE(pattern.TryMatch(graph, res).IsOK());
}

TEST(GraphParser, FastGeluWithBiasFusionTestPart1) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu_with_bias.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logging::LoggingManager::DefaultLogger()));
  Graph& graph = p_model->MainGraph();

  PatternGraph pattern(
      {IArg("X", PatternType(PatternTypeCategory::Float)),
       IArg("C0", PatternType(PatternTypeCategory::Integer)),
       IArg("C1", PatternType(PatternTypeCategory::Float))},
      {PNode("Add", {"C1", "C1"}, {"Add1"}, "p_add1"),
       PNode("Mul", {"Add1", "C1"}, {"Mul1"}, "p_mul1"),
       PNode("Mul", {"Add1", "Mul1"}, {"Mul2"}, "p_mul2"),
       PNode("Add", {"Mul2", "C1"}, {"Add2"}, "p_add2"),
       PNode("Mul", {"Add2", "C1"}, {"Mul4"}, "p_mul4"),
       PNode("Tanh", {"Mul4"}, {"Tanh"}, "p_tanh"),
       PNode("Add", {"Tanh", "C1"}, {"Add3"}, "p_add3"),
       PNode("Mul", {"Add1", "C1"}, {"Mul5"}, "p_mul5"),
       PNode("Mul", {"Mul5", "Add3"}, {"Mul6"}, "p_mul6")});

  PatternMatchResult res(graph);
  ASSERT_TRUE(pattern.TryMatch(graph, res).IsOK());
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
