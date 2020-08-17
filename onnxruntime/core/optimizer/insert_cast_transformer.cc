// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/insert_cast_transformer.h"
#include "core/framework/data_types.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
class IdGenerator {
 public:
  int Next() {
    return id++;
  }

 private:
  int id = 0;
};

bool InsertCastTransformer::NeedInsertCast(const onnxruntime::Node* node, const onnxruntime::NodeArg* input) const {
  //If the node's input is float16 and currently the node is not assigned to any XP.
  //we need insert a cast to float, and put the node on CPU for default behavior.
  //TODO: a better check is to check does the CPU kernel with float exist or not.
  return input->Type() != nullptr &&
         DataTypeImpl::TypeFromProto(*input->TypeAsProto()) == DataTypeImpl::GetTensorType<MLFloat16>() &&
         node->GetExecutionProviderType().empty();
}

onnxruntime::NodeArg* AddCastNode(onnxruntime::Graph& graph,
                                  IdGenerator& id_generator,
                                  onnxruntime::NodeArg* old_arg,
                                  TypeProto* new_type,
                                  bool new_on_input,
                                  int64_t to_type,
                                  onnxruntime::ProviderType providerType) {
  //insert cast op to cast input
  int id = id_generator.Next();

  char str[32];
  snprintf(str, 32, "CastDef_%d", id);

  auto* new_arg = &graph.GetOrCreateNodeArg(str, new_type);

  std::vector<onnxruntime::NodeArg*> input_defs = {new_on_input ? new_arg : old_arg};
  std::vector<onnxruntime::NodeArg*> output_defs = {new_on_input ? old_arg : new_arg};

  auto& cast_node = graph.AddNode(str, "Cast", "cast node to cast from float16 to float32 on cpu", input_defs, output_defs);
  cast_node.AddAttribute("to", to_type);
  cast_node.SetExecutionProviderType(providerType);
  return new_arg;
}

static bool IsInputFloat16(const onnxruntime::Node& node) {
  for (auto input : node.InputDefs()) {
    if (input->Type() != nullptr &&
        DataTypeImpl::TypeFromProto(*input->TypeAsProto()) == DataTypeImpl::GetTensorType<MLFloat16>() &&
        !node.GetExecutionProviderType().empty()) {
      return true;
    }
  }
  return false;
}

static bool IsSingleInputNodeFloat16Node(const onnxruntime::Node& node) {
  if (IsInputFloat16(node) && node.GetExecutionProviderType() == kCpuExecutionProvider) {
    for (auto it = node.InputNodesBegin(); it != node.InputNodesEnd(); ++it) {
      if (IsInputFloat16(*it))
        return false;
    }
    for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
      if (IsInputFloat16(*it))
        return false;
    }
    return true;
  }
  return false;
}

Status ForceSingleNodeCPUFloat16ToFloat32(onnxruntime::Graph& graph) {
  // if graph only contain 1 compute node, don't force to float32
  if (graph.NumberOfNodes() <= 1) {
    return Status::OK();
  }

  for (auto& node : graph.Nodes()) {
    if (IsSingleInputNodeFloat16Node(node)) {
      node.SetExecutionProviderType("");
    }
  }

  return Status::OK();
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

        // if cast's next node is also cast and next cast's output type equal to cast's input type
        // remove those two cast.
        // boolean is an exception case for this optimization
        auto src_type = node.InputDefs()[0]->Type();
        auto dst_type = node.OutputDefs()[0]->Type();
        if (*src_type == "tensor(bool)" || *dst_type == "tensor(bool)")
          continue;

        size_t num_children = node.GetOutputEdgesCount();

        for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
          const Node& output_node(*it);
          if (output_node.OpType() == "Cast") {
            // Skip this child node if this child node's output is also an output of the graph
            if (graph_outputs.find(output_node.OutputDefs()[0]) != graph_outputs.end()) {
              continue;
            }

            auto src_type1 = output_node.InputDefs()[0]->Type();
            auto dst_type1 = output_node.OutputDefs()[0]->Type();
            if (src_type == dst_type1 && src_type1 == dst_type) {
              // get a mutable reference to the output node and save it
              nodes_to_remove.push_back(*graph.GetNode(output_node.Index()));
            }
          }
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

          // if we removed all the child nodes and we're not providing graph output we can remove this node
          if (num_children > 0 && nodes_to_remove.size() == num_children &&
              graph_outputs.find(node.OutputDefs()[0]) == graph_outputs.end()) {
            graph.RemoveNode(node.Index());
            removed = true;
          }
        }
      }

      if (!removed) {
        ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
      }
    }

    return Status::OK();
  }
};

Status InsertCastTransformer::ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  if (force_cpu_fp32_)
    ORT_RETURN_IF_ERROR(ForceSingleNodeCPUFloat16ToFloat32(graph));

  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();
  TypeProto float_16_tensor_proto;
  TypeProto float_tensor_proto;
  float_16_tensor_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  float_tensor_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  IdGenerator id_generator;
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
          //insert cast op to cast input
          auto dst_arg = AddCastNode(graph,
                                     id_generator,
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
    }

    auto& outputs = node->MutableOutputDefs();
    for (auto output : outputs) {
      // TODO 1: Check if the kernel available
      // TODO 2: There is an inherent assumption that if we cast a cpu op's input from float16 to float
      // then this cpu op's output will be float (if it was inferred to be float16 previously).
      // Not sure if this is always true. Handle any corner case if it does exist.

      if (output->Type() &&
          DataTypeImpl::TypeFromProto(*output->TypeAsProto()) == DataTypeImpl::GetTensorType<MLFloat16>() &&
          casted) {
        //insert cast op to cast output back to float16
        auto dst_arg = output;
        auto src_arg = AddCastNode(graph,
                                   id_generator,
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

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));
  }

  auto status = Status::OK();

  // if this is the main graph we've recursed into all the subgraphs and added Cast nodes.
  // run the duplicate remover now, which will call Graph::Resolve from Apply(...) and handle the main and subgraphs.
  if (graph_level == 0) {
    if (modified) {
      ORT_RETURN_IF_ERROR(graph.Resolve());
    }

    RemoveDuplicateCastTransformer remover;
    // RemoveDuplicateCastTransformer is a special transformer required for correctness.
    // It is provider agnostic so simply send an empty vector.
    status = remover.Apply(graph, modified, logger);
  }

  return status;
}
}  // namespace onnxruntime
