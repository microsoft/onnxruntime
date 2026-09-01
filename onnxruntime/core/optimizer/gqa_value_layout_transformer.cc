// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gqa_value_layout_transformer.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/graph/graph_utils.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

namespace {

// GroupQueryAttention operand positions. See docs/ContribOperators.md#com.microsoft.GroupQueryAttention.
constexpr size_t kPastValueInputIndex = 4;
constexpr size_t kPresentValueOutputIndex = 2;

// Swaps the last two dimensions of a rank-4 tensor, i.e. BNSH <-> BNHS.
constexpr std::array<int64_t, 4> kValueLayoutPerm{0, 1, 3, 2};

bool HasInput(const Node& node, size_t index) {
  return index < node.InputDefs().size() && node.InputDefs()[index] != nullptr &&
         node.InputDefs()[index]->Exists();
}

bool HasOutput(const Node& node, size_t index) {
  return index < node.OutputDefs().size() && node.OutputDefs()[index] != nullptr &&
         node.OutputDefs()[index]->Exists();
}

std::string GetStringAttr(const Node& node, const std::string& attr_name, const std::string& default_value) {
  const auto* attr = graph_utils::GetNodeAttribute(node, attr_name);
  return (attr != nullptr && attr->has_s()) ? attr->s() : default_value;
}

int64_t GetIntAttr(const Node& node, const std::string& attr_name, int64_t default_value) {
  const auto* attr = graph_utils::GetNodeAttribute(node, attr_name);
  return (attr != nullptr && attr->has_i()) ? attr->i() : default_value;
}

// A name to use in log and error messages. Node names are optional in ONNX.
std::string DescribeNode(const Node& node) {
  return node.Name().empty() ? ("GroupQueryAttention#" + std::to_string(node.Index())) : node.Name();
}

// Is this a Transpose node that swaps the last two dimensions of a rank-4 tensor?
bool IsValueLayoutTranspose(const Node& node) {
  if (node.OpType() != "Transpose" || node.Domain() != kOnnxDomain) {
    return false;
  }

  const auto* perm = graph_utils::GetNodeAttribute(node, "perm");
  if (perm == nullptr || static_cast<size_t>(perm->ints_size()) != kValueLayoutPerm.size()) {
    return false;
  }

  for (size_t i = 0; i < kValueLayoutPerm.size(); ++i) {
    if (perm->ints(static_cast<int>(i)) != kValueLayoutPerm[i]) {
      return false;
    }
  }

  return true;
}

// Whether one Value operand of a node is something this transformer can or should convert. The
// two operands are classified independently: a model may legitimately expose only one of them to
// the application, and converting just that one keeps the graph coherent because the GQA node
// itself stays BNSH on both sides either way.
enum class OperandStatus {
  kAbsent,       // the node does not have this operand
  kConverted,    // already routed through a value-layout Transpose to or from an application boundary
  kConvertible,  // sits at an application boundary and is not converted yet
  kOutOfScope,   // present, but not a boundary the application binds; it stays BNSH
};

// Graph inputs excluding initializers: the tensors an application must supply. An initializer that
// is not also a graph input is baked into the model and can never be bound, so it is out of scope.
bool IsApplicationInput(const Graph& graph, const NodeArg* arg) {
  if (arg == nullptr) {
    return false;
  }
  for (const auto* graph_input : graph.GetInputs()) {
    if (graph_input != nullptr && graph_input->Name() == arg->Name()) {
      return true;
    }
  }
  return false;
}

// An initializer that is also declared as a graph input, so a feed may override it at run time.
bool IsOverridableInitializer(const Graph& graph, const NodeArg* arg) {
  if (arg == nullptr) {
    return false;
  }
  for (const auto* initializer : graph.GetOverridableInitializers()) {
    if (initializer != nullptr && initializer->Name() == arg->Name()) {
      return true;
    }
  }
  return false;
}

Status ClassifyPastValue(const Graph& graph, const Node& node, OperandStatus& status) {
  status = OperandStatus::kAbsent;
  if (!HasInput(node, kPastValueInputIndex)) {
    return Status::OK();
  }

  const NodeArg* arg = node.InputDefs()[kPastValueInputIndex];

  // Converting again would insert a second Transpose and swap the boundary shape back to BNSH while
  // the application still supplies BNHS, so recognizing the converted form is a correctness
  // requirement rather than an optimization.
  const Node* producer = graph.GetProducerNode(arg->Name());
  if (producer != nullptr && IsValueLayoutTranspose(*producer) &&
      IsApplicationInput(graph, producer->InputDefs()[0])) {
    status = OperandStatus::kConverted;
    return Status::OK();
  }

  // An overridable initializer is bindable, so it is an application boundary, but its baked-in data
  // stays BNSH whatever we do to the declared shape. Swapping the shape alone would either fail
  // Graph::Resolve on the initializer/NodeArg mismatch or, when the feed is omitted, hand the
  // default BNSH buffer to a Transpose that reads it as BNHS.
  ORT_RETURN_IF(IsOverridableInitializer(graph, arg),
                "GroupQueryAttention node '", DescribeNode(node),
                "' reads past_value from an overridable "
                "initializer ('",
                arg->Name(), "'), which the '", kOrtSessionOptionsGqaValueLayout,
                "' option cannot convert: the initializer data would stay BNSH. Remove the initializer so the "
                "input is supplied by the application, or transpose it to BNHS when producing the model.");

  status = IsApplicationInput(graph, arg) ? OperandStatus::kConvertible : OperandStatus::kOutOfScope;
  return Status::OK();
}

OperandStatus ClassifyPresentValue(const Graph& graph, const Node& node) {
  if (!HasOutput(node, kPresentValueOutputIndex)) {
    return OperandStatus::kAbsent;
  }

  const NodeArg* arg = node.OutputDefs()[kPresentValueOutputIndex];

  const auto consumers = graph.GetConsumerNodes(arg->Name());
  if (consumers.size() == 1 && consumers[0] != nullptr && IsValueLayoutTranspose(*consumers[0]) &&
      graph.IsOutput(consumers[0]->OutputDefs()[0])) {
    return OperandStatus::kConverted;
  }

  return graph.IsOutput(arg) ? OperandStatus::kConvertible : OperandStatus::kOutOfScope;
}

// Which operands of one node this transformer will convert.
struct NodeConversionPlan {
  bool convert_past_value = false;
  bool convert_present_value = false;

  bool AnythingToDo() const { return convert_past_value || convert_present_value; }
};

Node& AddValueLayoutTranspose(Graph& graph,
                              const std::string& name,
                              const std::string& description,
                              NodeArg& input,
                              NodeArg& output) {
  Node& transpose = graph.AddNode(graph.GenerateNodeName(name), "Transpose", description,
                                  {&input}, {&output}, nullptr, kOnnxDomain);
  transpose.AddAttribute("perm", std::vector<int64_t>{kValueLayoutPerm.begin(), kValueLayoutPerm.end()});
  return transpose;
}

// A declared shape can only be reinterpreted between BNSH and BNHS if it is rank 4. An undeclared
// shape imposes no constraint and needs no update. Checked during validation so that the mutation
// below cannot fail.
Status ValidateSwappableShape(const NodeArg& arg) {
  const auto* shape = arg.Shape();
  if (shape == nullptr) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(shape->dim_size() == 4, "GQA Value cache tensor '", arg.Name(), "' must be rank 4 to use the ",
                    "BNHS layout, but it has rank ", shape->dim_size(), ".");
  return Status::OK();
}

// Rewrites a rank-4 declared shape from BNSH to BNHS (or back). Symbolic dimension parameters are
// carried across unchanged, so the transposed shape stays consistent with the rest of the graph.
// Infallible by construction: ValidateSwappableShape() has already established rank 4 or no shape.
void SwapLastTwoDims(NodeArg& arg) {
  const auto* shape = arg.Shape();
  if (shape == nullptr || shape->dim_size() != 4) {
    return;
  }

  ONNX_NAMESPACE::TensorShapeProto swapped = *shape;
  swapped.mutable_dim()->SwapElements(2, 3);
  arg.SetShape(swapped);
}

// Rejects Value cache formats that a Transpose pair cannot express, independently of how much of
// the layout the node already carries.
Status ValidateCacheFormat(const Node& node) {
  // A 4-bit Value cache is uint8 with two values packed into each byte along head_size. A byte-wise
  // Transpose moves whole bytes, so it cannot convert between BNHS and BNSH packing, and the
  // declared-shape swap would be wrong as well. Reject rather than silently producing bad results
  // on any EP that does not fuse the Transposes away.
  const bool value_cache_is_quantized = GetStringAttr(node, "v_quant_type", "NONE") != "NONE";
  const int64_t bit_width = GetIntAttr(node, "kv_cache_bit_width", 8);
  ORT_RETURN_IF(value_cache_is_quantized && bit_width == 4,
                "GroupQueryAttention node '", DescribeNode(node), "' uses a 4-bit quantized Value cache, which is ",
                "not supported with the BNHS Value layout ('", kOrtSessionOptionsGqaValueLayout,
                "'). Two 4-bit values are packed per byte along head_size and cannot be transposed byte-wise.");

  return Status::OK();
}

// Decides what to do with one node, without mutating the graph.
//
// Returns an error for a topology the application would observe as inconsistent: it asked for BNHS,
// so an application-bound boundary that stays BNSH means the buffers it binds are in the wrong
// layout. Failing at initialization is the only way to keep the option external contract honest.
//
// Leaves an operand out of the plan, with a warning, when it is not an application boundary. Such a
// cache stays BNSH by design; see the scope note on kOrtSessionOptionsGqaValueLayout. The two
// operands are judged separately, so a node with one bound and one internal cache still gets the
// bound side converted.
Status ClassifyNode(const Graph& graph, const Node& node, const logging::Logger& logger,
                    NodeConversionPlan& plan) {
  plan = NodeConversionPlan{};

  // Checked before the operand statuses, so that a model which already carries the Transposes is
  // rejected too. A 4-bit cache is unsupported whether this run would insert the Transposes or a
  // previous one already did: accepting an already converted node here would let the model
  // initialize and then run the invalid byte-wise transpose on any EP that does not fuse it.
  ORT_RETURN_IF_ERROR(ValidateCacheFormat(node));

  OperandStatus past_value_status = OperandStatus::kAbsent;
  ORT_RETURN_IF_ERROR(ClassifyPastValue(graph, node, past_value_status));
  const OperandStatus present_value_status = ClassifyPresentValue(graph, node);

  // One boundary converted while the other was equally convertible means the graph was edited by
  // hand or produced by a build that failed part way. The two boundaries no longer agree with each
  // other and converting the remainder cannot repair that. A converted operand paired with an
  // absent or out-of-scope one is a legitimate fully converted node, hence the narrow condition.
  if ((past_value_status == OperandStatus::kConverted && present_value_status == OperandStatus::kConvertible) ||
      (present_value_status == OperandStatus::kConverted && past_value_status == OperandStatus::kConvertible)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "GroupQueryAttention node '", DescribeNode(node), "' has the BNHS Value layout applied ",
                           "to only one of past_value / present_value. The graph is inconsistent, so the '",
                           kOrtSessionOptionsGqaValueLayout, "' option cannot be applied safely.");
  }

  if (past_value_status == OperandStatus::kOutOfScope) {
    LOGS(logger, WARNING) << "GroupQueryAttention node '" << DescribeNode(node) << "' has a past_value input ('"
                          << node.InputDefs()[kPastValueInputIndex]->Name() << "') that the application does not "
                          << "bind, so it is out of scope for the '" << kOrtSessionOptionsGqaValueLayout
                          << "' option and keeps the BNSH layout.";
  }

  if (present_value_status == OperandStatus::kOutOfScope) {
    LOGS(logger, WARNING) << "GroupQueryAttention node '" << DescribeNode(node) << "' has a present_value output ('"
                          << node.OutputDefs()[kPresentValueOutputIndex]->Name() << "') that the application does not "
                          << "read, so it is out of scope for the '" << kOrtSessionOptionsGqaValueLayout
                          << "' option and keeps the BNSH layout.";
  }

  plan.convert_past_value = past_value_status == OperandStatus::kConvertible;
  plan.convert_present_value = present_value_status == OperandStatus::kConvertible;

  if (!plan.AnythingToDo()) {
    if (past_value_status == OperandStatus::kConverted || present_value_status == OperandStatus::kConverted) {
      LOGS(logger, INFO) << "GroupQueryAttention node '" << DescribeNode(node)
                         << "' already uses the BNHS Value layout. Skipping.";
    }
    return Status::OK();
  }

  // Each boundary NodeArg is shared state: swapping its declared shape is visible to every node that
  // reads or writes it, but only this node gets rewired through a Transpose. If a boundary has any
  // other user, converting it would leave that user interpreting the tensor in the wrong layout
  // (and, for a shared past_value, would swap the declared shape a second time and undo it). These
  // boundaries are application visible, so the option cannot be honored and this is an error rather
  // than a silent skip.
  if (plan.convert_past_value) {
    const NodeArg* boundary_arg = node.InputDefs()[kPastValueInputIndex];
    const auto consumers = graph.GetConsumerNodes(boundary_arg->Name());
    if (consumers.size() != 1 || consumers[0] != &node) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "GroupQueryAttention node '", DescribeNode(node), "' reads a past_value graph input ('",
                             boundary_arg->Name(), "') that has ", consumers.size(), " consumer node(s); the '",
                             kOrtSessionOptionsGqaValueLayout, "' option requires this node to be its only consumer. ",
                             "A Value cache shared between nodes cannot be converted to BNHS.");
    }
    ORT_RETURN_IF_ERROR(ValidateSwappableShape(*boundary_arg));
  }

  if (plan.convert_present_value) {
    const NodeArg* boundary_arg = node.OutputDefs()[kPresentValueOutputIndex];
    const auto consumers = graph.GetConsumerNodes(boundary_arg->Name());
    if (!consumers.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "GroupQueryAttention node '", DescribeNode(node), "' writes a present_value graph output ('",
                             boundary_arg->Name(), "') that is also consumed by ", consumers.size(),
                             " node(s) inside the graph; the '", kOrtSessionOptionsGqaValueLayout,
                             "' option requires it to have no internal consumers, which would receive BNHS data where ",
                             "they expect BNSH.");
    }
    ORT_RETURN_IF_ERROR(ValidateSwappableShape(*boundary_arg));
  }

  return Status::OK();
}

// Rewires one validated node according to its plan. Has no failure modes: ClassifyNode() has already
// established every precondition, which is what lets the caller validate the whole graph before
// mutating any of it.
void TransformNode(Graph& graph, Node& node, const NodeConversionPlan& plan,
                   GqaValueLayoutBoundaries* converted_boundaries) {
  if (plan.convert_past_value) {
    // The graph input keeps its name and identity but now declares BNHS. A new NodeArg carries the
    // BNSH result of the Transpose into the GQA node, inheriting the original (BNSH) type/shape.
    NodeArg* boundary_arg = node.MutableInputDefs()[kPastValueInputIndex];
    NodeArg& bnsh_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(boundary_arg->Name() + "_bnsh"),
                                                 boundary_arg->TypeAsProto());

    AddValueLayoutTranspose(graph,
                            DescribeNode(node) + "/past_value_bnhs_to_bnsh",
                            "Converts the GQA past_value cache from BNHS to the BNSH layout the operator requires",
                            *boundary_arg,
                            bnsh_arg);

    graph_utils::ReplaceNodeInput(node, static_cast<int>(kPastValueInputIndex), bnsh_arg);
    SwapLastTwoDims(*boundary_arg);

    if (converted_boundaries != nullptr) {
      converted_boundaries->past_value_inputs.push_back(boundary_arg->Name());
    }
  }

  if (plan.convert_present_value) {
    // Symmetrically: the GQA node now writes BNSH into a new NodeArg, and the Transpose produces
    // the graph output, which keeps its name and identity but now declares BNHS.
    NodeArg* boundary_arg = node.MutableOutputDefs()[kPresentValueOutputIndex];
    NodeArg& bnsh_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(boundary_arg->Name() + "_bnsh"),
                                                 boundary_arg->TypeAsProto());

    // Retarget the GQA output before adding the Transpose so the graph never has two producers
    // for the boundary NodeArg.
    node.MutableOutputDefs()[kPresentValueOutputIndex] = &bnsh_arg;

    AddValueLayoutTranspose(graph,
                            DescribeNode(node) + "/present_value_bnsh_to_bnhs",
                            "Converts the GQA present_value cache from BNSH to the BNHS layout the application expects",
                            bnsh_arg,
                            *boundary_arg);

    SwapLastTwoDims(*boundary_arg);

    if (converted_boundaries != nullptr) {
      converted_boundaries->present_value_outputs.push_back(boundary_arg->Name());
    }
  }
}

}  // namespace

Status GqaValueLayoutTransformer::ApplyImpl(Graph& graph,
                                            bool& modified,
                                            int graph_level,
                                            const logging::Logger& logger) const {
  // Main graph only, so Recurse() is deliberately not called. The session option describes the
  // layout of the buffers the application binds to the session; a subgraph boundary (a BeamSearch
  // decoder body, a Loop carried value) is not that boundary.
  if (graph_level != 0) {
    return Status::OK();
  }

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // First pass: classify every GroupQueryAttention node without touching the graph. An
  // unconvertible topology therefore fails initialization with the graph exactly as it was loaded,
  // instead of leaving earlier nodes converted and the graph unresolved. It also means every node is
  // judged against the original graph, so the verdict does not depend on topological order or on
  // producer/consumer bookkeeping being up to date mid-rewrite.
  InlinedVector<std::pair<NodeIndex, NodeConversionPlan>> nodes_to_transform;

  for (auto node_index : node_topology_list) {
    const Node* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr) {
      continue;
    }
    const Node& node = *node_ptr;

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GroupQueryAttention", {1}, kMSDomain)) {
      continue;
    }

    NodeConversionPlan plan;
    ORT_RETURN_IF_ERROR(ClassifyNode(graph, node, logger, plan));
    if (plan.AnythingToDo()) {
      nodes_to_transform.emplace_back(node_index, plan);
    }
  }

  // Second pass: rewire. TransformNode() cannot fail, so the graph is either fully converted or
  // untouched.
  for (const auto& [node_index, plan] : nodes_to_transform) {
    Node* node_ptr = graph.GetNode(node_index);
    ORT_RETURN_IF(node_ptr == nullptr, "GroupQueryAttention node ", node_index,
                  " disappeared between validation and transformation.");

    TransformNode(graph, *node_ptr, plan, converted_boundaries_);
    modified = true;

    LOGS(logger, INFO) << "Applied the BNHS Value layout to GroupQueryAttention node '"
                       << DescribeNode(*node_ptr) << "'.";
  }

  return Status::OK();
}

InlinedVector<std::string> ReportUnfusedGqaValueLayoutTransposes(const Graph& graph,
                                                                 const GqaValueLayoutBoundaries& boundaries,
                                                                 const logging::Logger& logger) {
  InlinedVector<std::string> unfused;

  // Anchored on the boundary rather than on the GQA node: a compiling EP may fuse the whole
  // Transpose -> GQA -> Transpose sequence (in which case the boundary now connects straight to the
  // fused node and there is nothing to report), or claim only the GQA node and leave the Transposes
  // behind (in which case both full-cache copies still run and there is no GQA node to search from).
  const auto report = [&](const std::string& boundary_name, const Node* transpose, const char* operand) {
    if (transpose == nullptr || !IsValueLayoutTranspose(*transpose)) {
      return;  // absorbed by the provider, or never a Transpose to begin with
    }

    const std::string& ep = transpose->GetExecutionProviderType();
    LOGS(logger, WARNING) << "The Value-layout Transpose for the " << operand << " boundary '" << boundary_name
                          << "' was not fused by EP '" << (ep.empty() ? "<unassigned>" : ep) << "'. The BNHS Value "
                          << "cache will be transposed at runtime: expect a full copy of the cache per step and no "
                          << "past/present buffer sharing. Either select the '" << kGqaValueLayoutBNSH
                          << "' layout for '" << kOrtSessionOptionsGqaValueLayout << "', or use an EP that reports '"
                          << kGqaValueLayoutBNHS << "' for '" << kOrtEpDevice_EpMetadataKey_GqaPreferredValueLayout
                          << "'.";
    unfused.push_back(boundary_name);
  };

  for (const auto& boundary_name : boundaries.past_value_inputs) {
    // The graph input feeds the Transpose, so look at what consumes it.
    const auto consumers = graph.GetConsumerNodes(boundary_name);
    report(boundary_name, consumers.size() == 1 ? consumers[0] : nullptr, "past_value");
  }

  for (const auto& boundary_name : boundaries.present_value_outputs) {
    // The Transpose produces the graph output.
    report(boundary_name, graph.GetProducerNode(boundary_name), "present_value");
  }

  return unfused;
}

}  // namespace onnxruntime
