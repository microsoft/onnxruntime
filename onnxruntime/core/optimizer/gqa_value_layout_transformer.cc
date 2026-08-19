// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gqa_value_layout_transformer.h"

#include <array>
#include <string>
#include <vector>

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

// True if this GQA node has already been adapted, either by a previous run of this transformer or
// because the model was saved after the transform was applied (session.optimized_model_filepath).
// Running twice would insert a second pair of Transposes and swap the boundary shapes back to BNSH
// while the application still supplies BNHS, so this check is required for correctness rather than
// being a mere optimization.
bool AlreadyTransformed(const Graph& graph, const Node& node, const logging::Logger& logger) {
  const bool has_past_value = HasInput(node, kPastValueInputIndex);
  const bool has_present_value = HasOutput(node, kPresentValueOutputIndex);

  bool past_value_transformed = false;
  if (has_past_value) {
    const Node* producer = graph.GetProducerNode(node.InputDefs()[kPastValueInputIndex]->Name());
    past_value_transformed = producer != nullptr && IsValueLayoutTranspose(*producer) &&
                             graph.IsInputsIncludingInitializers(producer->InputDefs()[0]);
  }

  bool present_value_transformed = false;
  if (has_present_value) {
    const auto consumers = graph.GetConsumerNodes(node.OutputDefs()[kPresentValueOutputIndex]->Name());
    present_value_transformed = consumers.size() == 1 && consumers[0] != nullptr &&
                                IsValueLayoutTranspose(*consumers[0]) &&
                                graph.IsOutput(consumers[0]->OutputDefs()[0]);
  }

  // When a node has both operands, this transformer always converts them together, so a mismatch
  // means the graph was hand-edited or a previous run stopped part way. Report it, and still treat
  // the node as transformed so we do not compound the inconsistency. A node with only one of the
  // two operands legitimately has only that side converted, hence the two guards.
  if (has_past_value && has_present_value && past_value_transformed != present_value_transformed) {
    LOGS(logger, ERROR) << "GroupQueryAttention node '" << DescribeNode(node) << "' has the BNHS Value layout "
                        << "applied to only one of past_value / present_value. Leaving the node unchanged.";
  }

  return past_value_transformed || present_value_transformed;
}

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

// Rewrites a rank-4 declared shape from BNSH to BNHS (or back). Symbolic dimension parameters are
// carried across unchanged, so the transposed shape stays consistent with the rest of the graph.
Status SwapLastTwoDims(NodeArg& arg) {
  const auto* shape = arg.Shape();
  if (shape == nullptr) {
    // No declared shape means no constraint to update: any shape is accepted at this boundary.
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(shape->dim_size() == 4, "GQA Value cache tensor '", arg.Name(), "' must be rank 4 to use the ",
                    "BNHS layout, but it has rank ", shape->dim_size(), ".");

  ONNX_NAMESPACE::TensorShapeProto swapped = *shape;
  swapped.mutable_dim()->SwapElements(2, 3);
  arg.SetShape(swapped);

  return Status::OK();
}

// Rejects configurations the Transpose pair cannot express. Returns an error for those; the caller
// skips (with a warning) for cases that are merely outside this transformer's scope.
Status ValidateNode(const Node& node) {
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

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (node_ptr == nullptr) {
      continue;
    }
    Node& node = *node_ptr;

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GroupQueryAttention", {1}, kMSDomain)) {
      continue;
    }

    if (AlreadyTransformed(graph, node, logger)) {
      LOGS(logger, INFO) << "GroupQueryAttention node '" << DescribeNode(node)
                         << "' already uses the BNHS Value layout. Skipping.";
      continue;
    }

    ORT_RETURN_IF_ERROR(ValidateNode(node));

    const bool has_past_value = HasInput(node, kPastValueInputIndex);
    const bool has_present_value = HasOutput(node, kPresentValueOutputIndex);
    if (!has_past_value && !has_present_value) {
      continue;
    }

    // The layout only means something if the tensor is what the application binds to the session.
    // Anything else is outside this transformer's contract, so leave the node alone rather than
    // rewriting a layout the producer or consumer does not expect.
    if (has_past_value && !graph.IsInputsIncludingInitializers(node.InputDefs()[kPastValueInputIndex])) {
      LOGS(logger, WARNING) << "GroupQueryAttention node '" << DescribeNode(node) << "' has a past_value input ('"
                            << node.InputDefs()[kPastValueInputIndex]->Name() << "') that is not a graph input. "
                            << "The BNHS Value layout will not be applied to this node.";
      continue;
    }

    if (has_present_value && !graph.IsOutput(node.OutputDefs()[kPresentValueOutputIndex])) {
      LOGS(logger, WARNING) << "GroupQueryAttention node '" << DescribeNode(node) << "' has a present_value output ('"
                            << node.OutputDefs()[kPresentValueOutputIndex]->Name() << "') that is not a graph output. "
                            << "The BNHS Value layout will not be applied to this node.";
      continue;
    }

    // Each boundary NodeArg is shared: swapping its declared shape is visible to every other node
    // that reads or writes it, but only this node gets rewired through a Transpose. If a boundary
    // has any other user, transforming it would leave that user interpreting the tensor in the wrong
    // layout (and, for a shared past_value, would swap the declared shape a second time and undo
    // it). Until this transformer can rewire every BNSH user of a boundary, require sole ownership.
    if (has_past_value) {
      const NodeArg* boundary_arg = node.InputDefs()[kPastValueInputIndex];
      const auto consumers = graph.GetConsumerNodes(boundary_arg->Name());
      if (consumers.size() != 1 || consumers[0] != &node) {
        LOGS(logger, ERROR) << "GroupQueryAttention node '" << DescribeNode(node) << "' shares its past_value graph "
                              << "input ('" << boundary_arg->Name() << "') with " << (consumers.size() - 1)
                              << " other node(s). The BNHS Value layout will not be applied to this node.";
        continue;
      }
    }

    if (has_present_value) {
      const NodeArg* boundary_arg = node.OutputDefs()[kPresentValueOutputIndex];
      const auto consumers = graph.GetConsumerNodes(boundary_arg->Name());
      if (!consumers.empty()) {
        LOGS(logger, ERROR) << "GroupQueryAttention node '" << DescribeNode(node) << "' has a present_value graph "
                              << "output ('" << boundary_arg->Name() << "') that is also consumed by " << consumers.size()
                              << " node(s) inside the graph. The BNHS Value layout will not be applied to this node.";
        continue;
      }
    }

    if (has_past_value) {
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
      ORT_RETURN_IF_ERROR(SwapLastTwoDims(*boundary_arg));

      modified = true;
    }

    if (has_present_value) {
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

      ORT_RETURN_IF_ERROR(SwapLastTwoDims(*boundary_arg));

      modified = true;
    }

    LOGS(logger, INFO) << "Applied the BNHS Value layout to GroupQueryAttention node '" << DescribeNode(node) << "'.";
  }

  return Status::OK();
}

void LogUnfusedGqaValueLayoutTransposes(const Graph& graph, const logging::Logger& logger) {
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() != "GroupQueryAttention" || node.Domain() != kMSDomain) {
      continue;
    }

    // A provider that fused the sequence has already replaced these nodes, so anything still here
    // will run as a real transpose of the whole Value cache.
    bool past_value_transpose_remains = false;
    if (HasInput(node, kPastValueInputIndex)) {
      const Node* producer = graph.GetProducerNode(node.InputDefs()[kPastValueInputIndex]->Name());
      past_value_transpose_remains = producer != nullptr && IsValueLayoutTranspose(*producer);
    }

    bool present_value_transpose_remains = false;
    if (HasOutput(node, kPresentValueOutputIndex)) {
      const auto consumers = graph.GetConsumerNodes(node.OutputDefs()[kPresentValueOutputIndex]->Name());
      present_value_transpose_remains = consumers.size() == 1 && consumers[0] != nullptr &&
                                        IsValueLayoutTranspose(*consumers[0]);
    }

    if (!past_value_transpose_remains && !present_value_transpose_remains) {
      continue;
    }

    const std::string& ep = node.GetExecutionProviderType();
    LOGS(logger, WARNING) << "GroupQueryAttention node '" << DescribeNode(node) << "' was assigned to EP '"
                          << (ep.empty() ? "<unassigned>" : ep) << "', which did not fuse the Value-layout Transpose "
                          << "nodes. The BNHS Value cache will be transposed at runtime: expect a full copy of the "
                          << "cache per step and no past/present buffer sharing. Either select the '"
                          << kGqaValueLayoutBNSH << "' layout for '" << kOrtSessionOptionsGqaValueLayout
                          << "', or use an EP that reports '" << kGqaValueLayoutBNHS << "' for '"
                          << kOrtEpDevice_EpMetadataKey_GqaPreferredValueLayout << "'.";
  }
}

}  // namespace onnxruntime
