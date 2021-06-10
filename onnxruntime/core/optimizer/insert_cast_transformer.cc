// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/insert_cast_transformer.h"
#include "core/framework/data_types.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
bool InsertCastTransformer::NeedInsertCast(const onnxruntime::Node* node, const onnxruntime::NodeArg* input) const {
  // If the node's input is float16 and currently the node is not assigned to any EP
  // we need to insert a cast to float, and put the node on CPU for default behavior.
  // TODO: a better check is to check does the CPU kernel with float exist or not.
  return input->Type() != nullptr &&
         DataTypeImpl::TypeFromProto(*input->TypeAsProto()) == DataTypeImpl::GetTensorType<MLFloat16>() &&
         node->GetExecutionProviderType().empty();
}

onnxruntime::NodeArg* AddCastNode(onnxruntime::Graph& graph,
                                  onnxruntime::NodeArg* old_arg,
                                  TypeProto* new_type,
                                  bool new_on_input,
                                  int64_t to_type,
                                  onnxruntime::ProviderType providerType) {
  // insert cast op to cast input
  std::string node_name = graph.GenerateNodeName("InsertedCast_" + old_arg->Name());

  auto* new_arg = &graph.GetOrCreateNodeArg(node_name, new_type);

  std::vector<onnxruntime::NodeArg*> input_defs = {new_on_input ? new_arg : old_arg};
  std::vector<onnxruntime::NodeArg*> output_defs = {new_on_input ? old_arg : new_arg};

  auto& cast_node = graph.AddNode(node_name, "Cast", "cast node to cast from float16 to float32 on cpu", input_defs, output_defs);
  cast_node.AddAttribute("to", to_type);
  cast_node.SetExecutionProviderType(providerType);
  return new_arg;
}

static bool IsMLFloat16Tensor(const NodeArg& node_arg) {
  // Type() will return nullptr if node_arg.Exists() is true so don't need an additional check for that
  return node_arg.Type() != nullptr &&
         DataTypeImpl::TypeFromProto(*node_arg.TypeAsProto()) == DataTypeImpl::GetTensorType<MLFloat16>();
}

// check if the node has an fp16 input but was not able to be assigned an execution provider.
// we will need to add a casts to/from fp32 around the node for it to be executed
static bool NodeNeedsInputCastToFp32(const onnxruntime::Node& node) {
  bool not_assigned = node.GetExecutionProviderType().empty();

  if (not_assigned) {
    const auto& input_defs = node.InputDefs();
    bool has_fp16_input = std::any_of(input_defs.cbegin(), input_defs.cend(),
                                      [](const NodeArg* input_def) {
                                        return IsMLFloat16Tensor(*input_def);
                                      });
    return has_fp16_input;
  }

  return false;
}

// Detect an isolated node that is able to process fp16 data but is between other nodes that have fp16 inputs
// but will need a Cast inserted to enable them to run.
//
// Say we have 3 nodes in the middle of a graph that all have fp16 inputs.
//
// -> NodeA -> NodeB -> NodeC ->
//
// NodeA and NodeC have no kernel that can handle fp16 data (no execution provider assigned).
//   e.g. 'Add' does not have an fp16 kernel
// NodeB has a kernel that can process fp16 data (assigned to CPU EP).
//
// By default, we would insert Cast to/from fp32 around NodeA and NodeC as all operators have an fp32 kernel.
//
// i.e. -> CastToFp32 -> NodeA -> CastToFp16 -> NodeB -> CastToFp32 -> NodeC -> CastToFp16
//
// We can avoid the casts around NodeB if we also force that to run using fp32 data.
//
// Detect this scenario by checking the input and output edges of the node for fp16 values to that are coming from or
// going to a node that will need a Cast.
//
// Return true if all the fp16 inputs and outputs are connected to nodes that will be cast to fp32.
static bool IsIsolatedFp16Node(const onnxruntime::Node& node, onnxruntime::Graph& graph) {
  bool isolated_fp16_node = false;

  // if node has input coming from other nodes (only consuming graph inputs or initializers if it doesn't),
  //    does not have a subgraph (would have to alter subgraph inputs if we cast the input to this node),
  //    does not produce a graph output (node must produce fp16 output for the graph output),
  //    and is assigned to the CPU EP (we have fp32 implementations of all kernels so forcing to fp32 is safe),
  // we can check if it's an isolated fp16 node
  if (node.GetInputEdgesCount() > 0 &&
      !node.ContainsSubgraph() &&
      graph.GetNodeOutputsInGraphOutputs(node).empty() &&
      node.GetExecutionProviderType() == kCpuExecutionProvider) {
    do {
      // find the number of fp16 inputs as we need to make sure they're all coming from nodes that will be cast
      const auto& input_defs = node.InputDefs();
      size_t num_fp16_inputs = std::count_if(input_defs.cbegin(), input_defs.cend(),
                                             [](const NodeArg* input_def) {
                                               return IsMLFloat16Tensor(*input_def);
                                             });

      if (num_fp16_inputs == 0) {
        break;
      }

      size_t num_fp16_input_edges = 0;

      // check if all nodes providing our fp16 input need to be cast to fp32
      for (auto input_edge = node.InputEdgesBegin(), end = node.InputEdgesEnd(); input_edge != end; ++input_edge) {
        const NodeArg& input_def = *input_defs[input_edge->GetDstArgIndex()];

        if (IsMLFloat16Tensor(input_def)) {
          // if the node producing our fp16 input does not need its input cast to fp32 we should run in fp16
          if (!NodeNeedsInputCastToFp32(input_edge->GetNode())) {
            break;
          }

          ++num_fp16_input_edges;
        }
      }

      // one or more fp16 inputs are coming from a graph input or initializer
      if (num_fp16_inputs != num_fp16_input_edges) {
        break;
      }

      // if we got here all nodes providing our fp16 input/s will be cast to fp32.
      // check if the same applies to all nodes consuming our fp16 output.

      bool node_has_fp16_output = false;

      for (auto output_edge = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); output_edge != end; ++output_edge) {
        const NodeArg& output_def = *node.OutputDefs()[output_edge->GetSrcArgIndex()];
        if (IsMLFloat16Tensor(output_def)) {
          node_has_fp16_output = true;

          // if the node consuming our fp16 output does not need a cast, we should run in fp16
          if (!NodeNeedsInputCastToFp32(output_edge->GetNode())) {
            break;
          }
        }
      }

      if (node_has_fp16_output) {
        // all nodes providing our fp16 input/s will be cast to fp32, and
        // we produce one or more fp16 outputs, and all nodes consuming those outputs will be cast to fp32
        isolated_fp16_node = true;
      }
    } while (false);
  }

  return isolated_fp16_node;
}

Status ForceSingleNodeCPUFloat16ToFloat32(onnxruntime::Graph& graph) {
  // if graph only contain 1 compute node, don't force to float32
  if (graph.NumberOfNodes() <= 1) {
    return Status::OK();
  }

  for (auto& node : graph.Nodes()) {
    if (IsIsolatedFp16Node(node, graph)) {
      node.SetExecutionProviderType("");
    }
  }

  return Status::OK();
}

enum TypeGroup {
  Unknown = -1,
  Bool = 0,
  Integer = 1,
  Float = 2,
};

TypeGroup GetTypeGroup(DataType type) {
  if (*type == "tensor(bool)") {
    return Bool;
  }

  if (*type == "tensor(int16)" || *type == "tensor(int32)" || *type == "tensor(int64)" || *type == "tensor(int8)" ||
      *type == "tensor(uint16)" || *type == "tensor(uint32)" || *type == "tensor(uint64)" || *type == "tensor(uint8)") {
    return Integer;
  }

  if (*type == "tensor(bfloat16)" || *type == "tensor(double)" || *type == "tensor(float)" || *type == "tensor(float16)") {
    return Float;
  }

  return Unknown;
}

/** Transformer to remove duplicate Cast nodes. */
class RemoveDuplicateCastTransformer : public GraphTransformer {
 public:
  RemoveDuplicateCastTransformer() : GraphTransformer("RemoveDuplicateCastTransformer") {
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override {
    std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacement_defs;

    auto output_args = graph.GetOutputs();
    const std::unordered_set<const onnxruntime::NodeArg*> graph_outputs(output_args.begin(), output_args.end());

    for (auto& node : graph.Nodes()) {
      bool removed = false;
      if (node.OpType() == "Cast") {
        std::vector<std::reference_wrapper<Node>> nodes_to_remove;
        std::vector<std::reference_wrapper<Node>> cast_nodes_to_keep;

        // if cast's next node is also cast:
        //     - if the next cast's output type is equal to cast's input type, remove these two casts.
        //     - otherwise, remove the first cast.
        // Below are some exception cases for this optimization:
        //     - it's for non-numeric type casting.
        //     - if the casts are for (high precision -> low precision -> high precision),
        //       since there is actual loss of precision.
        // Other cases are OK for this optimization, including below two cases,
        // which are not actual loss of precision:
        //     - (low precision -> high precision ->low precision)
        //     - (high precision -> low precision -> lower precision)
        // It's possible that there are more than one casts following the first cast,
        // the first cast can be removed only when:
        //     - not providing graph output, and
        //     - all consumer nodes are cast nodes, and
        //     - for each consumer cast node, it meets above condition for this optimization.
        auto src_type = node.InputDefs()[0]->Type();
        auto dst_type = node.OutputDefs()[0]->Type();
        TypeGroup src_type_group = GetTypeGroup(src_type);
        TypeGroup dst_type_group = GetTypeGroup(dst_type);
        if (src_type_group == Unknown || dst_type_group == Unknown) {
          continue;
        }

        bool loss_precision_cast = false;
        if (src_type_group > dst_type_group) {
          loss_precision_cast = true;
        }

        size_t num_children = node.GetOutputEdgesCount();

        bool inconsistent_casts = false;
        for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
          const Node& output_node(*it);
          if (output_node.OpType() == "Cast") {
            auto src_type1 = output_node.InputDefs()[0]->Type();
            auto dst_type1 = output_node.OutputDefs()[0]->Type();
            TypeGroup src_type_group1 = GetTypeGroup(src_type1);
            TypeGroup dst_type_group1 = GetTypeGroup(dst_type1);
            if (src_type_group1 == Unknown || dst_type_group1 == Unknown ||
                (loss_precision_cast && dst_type_group1 > src_type_group1)) {
              inconsistent_casts = true;
              break;
            }

            // Cannot remove node if it's output is also an output of the graph
            if (graph_outputs.find(output_node.OutputDefs()[0]) == graph_outputs.end() &&
                src_type == dst_type1 && src_type1 == dst_type) {
              // get a mutable reference to the output node and save it
              nodes_to_remove.push_back(*graph.GetNode(output_node.Index()));
            } else {
              cast_nodes_to_keep.push_back(*graph.GetNode(output_node.Index()));
            }
          }
        }

        if (inconsistent_casts) {
          continue;
        }

        if (!nodes_to_remove.empty()) {
          if (node.GetInputEdgesCount() == 0) {
            // replacing with initializer or graph input so we just need the NodeArg for the input
            auto& input = *node.MutableInputDefs()[0];

            for (auto& n : nodes_to_remove) {
              Node& node_to_remove = n;
              NodeIndex node_idx = node_to_remove.Index();

              // copy the edges so we can remove as we iterate them
              std::vector<Node::EdgeEnd> edges(node_to_remove.OutputEdgesBegin(), node_to_remove.OutputEdgesEnd());

              for (auto edge = edges.cbegin(), end = edges.cend(); edge != end; ++edge) {
                int dst_idx = edge->GetDstArgIndex();
                graph.RemoveEdge(node_idx, edge->GetNode().Index(), edge->GetSrcArgIndex(), dst_idx);

                // replace the input of the downstream nodes with the initializer
                Node& mutable_target = *graph.GetNode(edge->GetNode().Index());
                graph_utils::ReplaceNodeInput(mutable_target, dst_idx, input);
              }

              graph.RemoveNode(node_idx);
            }
          } else {
            // replace the output from the second Cast node with the input to 'node'
            const Node::EdgeEnd& input_edge = *node.InputEdgesBegin();
            Node& mutable_src_node = *graph.GetNode(input_edge.GetNode().Index());
            int replacement_idx = input_edge.GetSrcArgIndex();

            for (auto& n : nodes_to_remove) {
              Node& node_to_remove = n;
              // replace output index 0 (Cast only produces one output)
              graph_utils::ReplaceDownstreamNodeInput(graph, node_to_remove, 0, mutable_src_node, replacement_idx);

              graph.RemoveNode(node_to_remove.Index());
            }
          }

          modified = true;
        }

        // If all the child nodes are either removed or another Cast node and we're not providing graph output,
        // we can remove this node. Connect those remaining child Cast nodes to current Cast node's input.
        if (num_children > 0 && nodes_to_remove.size() + cast_nodes_to_keep.size() == num_children &&
            graph_outputs.find(node.OutputDefs()[0]) == graph_outputs.end()) {
          for (auto& n : cast_nodes_to_keep) {
            Node& cast_node_to_keep = n;
            graph.SetNodeArgType(*cast_node_to_keep.MutableInputDefs()[0], *node.InputDefs()[0]->TypeAsProto());
          }

          removed = graph_utils::RemoveNode(graph, node);
          modified = true;
        }
      }

      if (!removed) {
        ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
      }
    }

    return Status::OK();
  }
};

Status InsertCastTransformer::ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level,
                                        const logging::Logger& logger) const {
  if (force_cpu_fp32_)
    ORT_RETURN_IF_ERROR(ForceSingleNodeCPUFloat16ToFloat32(graph));

  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();
  TypeProto float_16_tensor_proto;
  TypeProto float_tensor_proto;
  float_16_tensor_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  float_tensor_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  std::map<onnxruntime::NodeArg*, onnxruntime::NodeArg*> input_def_updates;

  for (onnxruntime::NodeIndex i : order) {
    auto node = graph.GetNode(i);
    if (!node)
      return Status(ONNXRUNTIME, INVALID_ARGUMENT);

    auto& inputs = node->MutableInputDefs();
    std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacement_defs;
    bool casted = false;
    for (auto input : inputs) {
      if (NeedInsertCast(node, input)) {
        auto src_arg = input;
        if (input_def_updates.count(src_arg)) {
          replacement_defs[src_arg] = input_def_updates[src_arg];
        } else {
          // insert cast op to cast input
          auto dst_arg = AddCastNode(graph,
                                     src_arg,
                                     &float_tensor_proto,
                                     false,
                                     static_cast<int64_t>(TensorProto_DataType_FLOAT),
                                     //right now we only cast for cpu cases.
                                     onnxruntime::kCpuExecutionProvider);
          replacement_defs[src_arg] = dst_arg;
          input_def_updates[src_arg] = dst_arg;
        }
        casted = true;
      }
    }

    if (casted) {
      // Set current node to run on the CPU execution provider
      // Keep in mind that the EP will be empty because NeedInsertCast() already insures that
      node->SetExecutionProviderType(kCpuExecutionProvider);

      // Some ONNX operators have an attribute `dtype` which define the output type for these operators
      // (mostly Generator ops like RandomNormal, RandomNormalLike, EyeLike, etc.).
      // Update that so that `dtype` is now Float. Otherwise there could be a mis-match between the actual
      // type of the NodeArg and the ONNX inferred type of the NodeArg and Graph Resolve() will complain.
      auto& attributes = node->GetMutableAttributes();
      auto dtype_attribute = attributes.find("dtype");

      if (dtype_attribute != attributes.end()) {
        // Simple sanity check
        ORT_ENFORCE(dtype_attribute->second.has_i(),
                    "InsertCastTransformer works on the assumption that `dtype` attribute holds an integer.");

        // Modify the dtype attribute (which defines the output type) to FLOAT if it is FLOAT16.
        if (dtype_attribute->second.i() == TensorProto_DataType_FLOAT16) {
          dtype_attribute->second.set_i(TensorProto_DataType_FLOAT);
        }
      }

      auto& outputs = node->MutableOutputDefs();
      for (auto output : outputs) {
        // TODO 1: Check if the kernel available
        // TODO 2: There is an inherent assumption that if we cast a cpu op's input from float16 to float
        // then this cpu op's output will be float (if it was inferred to be float16 previously).
        // Not sure if this is always true. Handle any corner case if it does exist.

        if (IsMLFloat16Tensor(*output)) {
          // insert cast op to cast output back to float16
          auto dst_arg = output;
          auto src_arg = AddCastNode(graph,
                                     dst_arg,
                                     &float_tensor_proto,
                                     true,
                                     static_cast<int64_t>(TensorProto_DataType_FLOAT16),
                                     onnxruntime::kCpuExecutionProvider);
          replacement_defs[dst_arg] = src_arg;
        }
      }

      node->ReplaceDefs(replacement_defs);
      modified = modified || casted;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));
  }

  auto status = Status::OK();

  // if this is the main graph we've recursed into all the subgraphs and added Cast nodes.
  // run the duplicate remover now, which will call Graph::Resolve from Apply(...) and handle the main and subgraphs.
  if (graph_level == 0) {
    if (modified) {
      ORT_RETURN_IF_ERROR(graph.Resolve());
    }

    // if we had multiple nodes in a row that were converted to fp32 we will have casts around every node.
    // Casts in between converted nodes cancel each other out and can be removed.
    // e.g.
    //      -> NodeA(fp16) -> NodeB(fp16) ->
    // After converting both to fp32
    //      -> CastToFp32 -> NodeA(fp32) -> CastToFp16 -> CastToFp32 -> NodeB(fp32) -> CastToFp16
    // After running duplicate cast removal
    //      -> CastToFp32 -> NodeA(fp32) -> NodeB(fp32) -> CastToFp16
    //
    RemoveDuplicateCastTransformer remover;
    status = remover.Apply(graph, modified, logger);
  }

  return status;
}
}  // namespace onnxruntime
