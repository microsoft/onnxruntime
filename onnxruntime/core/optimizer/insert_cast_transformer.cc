// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/insert_cast_transformer.h"
#include "core/framework/data_types.h"

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
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override {

    std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacement_defs;
    std::vector<onnxruntime::NodeIndex> removed_nodes;
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "Cast") {
        // if cast's next node is also cast and next cast's output type equal to cast's input type
        // remove those two cast.
        // boolean is an exception case for this optimization
        auto src_type = node.InputDefs()[0]->Type();
        auto dst_type = node.OutputDefs()[0]->Type();
        if (*src_type == "tensor(bool)" || *dst_type == "tensor(bool)") return Status::OK();
        auto input = node.MutableInputDefs()[0];
        int child_removed = 0;
        int num_child = 0;
        auto output_args = graph.GetOutputs();
        std::unordered_set<const onnxruntime::NodeArg*> graph_outputs(output_args.begin(), output_args.end());
        for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
          const Node& output_node{*it};
          if (output_node.OpType() == "Cast") {
            // Skip if the node's output is also the output of the graph
            if (graph_outputs.find(output_node.OutputDefs()[0]) != graph_outputs.end()) {
              break;
            }
            auto src_type1 = output_node.InputDefs()[0]->Type();
            auto dst_type1 = output_node.OutputDefs()[0]->Type();
            if (src_type == dst_type1 && src_type1 == dst_type) {
              //node *it's output's follower could be linked with node's input.
              replacement_defs.clear();
              replacement_defs[const_cast<onnxruntime::NodeArg*>(output_node.OutputDefs()[0])] = input;
              for (auto next_it = output_node.OutputNodesBegin(); next_it != output_node.OutputNodesEnd(); ++next_it) {
                const_cast<onnxruntime::Node*>(&(*next_it))->ReplaceDefs(replacement_defs);
              }
              removed_nodes.push_back(output_node.Index());
              child_removed++;
            }
          }
          num_child++;
        }

        if (child_removed == num_child && child_removed > 0) {
          removed_nodes.push_back(node.Index());
        }
      }

      ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));
    }

    for (auto i : removed_nodes) {
      graph.RemoveNode(i);
    }

    modified = modified || !removed_nodes.empty();
    return Status::OK();
  }
};

Status InsertCastTransformer::ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level) const {

  if (force_cpu_fp32_)
    ORT_RETURN_IF_ERROR(ForceSingleNodeCPUFloat16ToFloat32(graph));

  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();
  TypeProto float_16_tensor_proto, float_tensor_proto;
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

    if (casted && node->GetExecutionProviderType().empty()) {
      //set current node to CPU execution provider
      node->SetExecutionProviderType(kCpuExecutionProvider);
    }

    auto& outputs = node->MutableOutputDefs();
    for (auto output : outputs) {
      // todo: check is the kernel available
      // here is based on the assumption that if we cast a cpu op's input from float16 to float
      // then this cpu op's output will become float.
      // not sure is it always correct...
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

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level));
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
    status = remover.Apply(graph, modified);
  }

  return status;
}
}  // namespace onnxruntime
