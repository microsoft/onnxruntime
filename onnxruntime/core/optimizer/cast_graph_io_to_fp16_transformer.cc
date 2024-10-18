// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/cast_graph_io_to_fp16_transformer.h"
#include <cassert>
#include <string>

#include "core/common/span_utils.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {

static bool IsMLFloat32Tensor(const NodeArg& node_arg) {
  // Type() will return nullptr if node_arg.Exists() is true so don't need an additional check for that
  return node_arg.Type() != nullptr && DataTypeImpl::TypeFromProto(*node_arg.TypeAsProto()) == DataTypeImpl::GetTensorType<float>();
}

static onnxruntime::NodeArg* AddCastNode(onnxruntime::Graph& graph,
                                         onnxruntime::NodeArg* old_arg,
                                         ONNX_NAMESPACE::TypeProto* new_type,
                                         bool new_on_input,
                                         int64_t to_type,
                                         onnxruntime::ProviderType providerType) {
  // insert cast op to cast input
  std::string node_name = graph.GenerateNodeName("InsertedPrecisionFreeCast_" + old_arg->Name());

  auto* new_arg = &graph.GetOrCreateNodeArg(node_name, new_type);

  std::vector<onnxruntime::NodeArg*> input_defs = {new_on_input ? new_arg : old_arg};
  std::vector<onnxruntime::NodeArg*> output_defs = {new_on_input ? old_arg : new_arg};

  std::string input_arg_str = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*input_defs[0]->TypeAsProto()));
  std::string output_arg_str = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*output_defs[0]->TypeAsProto()));

  auto& cast_node = graph.AddNode(node_name, "Cast", "cast node to cast from " + input_arg_str + " to " + output_arg_str + " on " + providerType,
                                  input_defs, output_defs);
  cast_node.AddAttribute("to", to_type);
  cast_node.SetExecutionProviderType(providerType);
  return new_arg;
}

Status CastGraphIOToFp16Transformer::ApplyImpl(
    Graph& graph,
    bool& modified,
    int /*graph_level*/,
    const logging::Logger& /*logger*/) const {
  const GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  ONNX_NAMESPACE::TypeProto float_16_tensor_proto;
  float_16_tensor_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);

  ONNX_NAMESPACE::TypeProto float_32_tensor_proto;
  float_32_tensor_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  std::map<onnxruntime::NodeArg*, onnxruntime::NodeArg*> input_def_updates;
  std::map<onnxruntime::NodeArg*, onnxruntime::NodeArg*> output_def_updates;

  for (NodeIndex node_index : node_topology_list) {
    Node* node = graph.GetNode(node_index);

    if (node == nullptr || !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders())) {
      continue;
    }

    std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*> replacement_defs;

    bool has_graph_input = false;

    for (auto& input_def : node->MutableInputDefs()) {
      if (graph.IsInputsIncludingInitializers(input_def)) {
        has_graph_input = true;
        break;
      }
    }

    // We only need to convert the inputs that not non-overridable initializers currently, since it means that
    // it cannot be changed by DML without a cast. On the other hand, initializer inputs can be modified
    // within DML. We can make this pass more complete later-on and convert every single node to fp16, but
    // it is way more complex than simply handling the inputs.
    if (has_graph_input) {
      for (auto& input_def : node->MutableInputDefs()) {
        if (!IsMLFloat32Tensor(*input_def)) {
          continue;
        }

        // TODO (pavignol): Convert scale/zeropoints in-place
        if (input_def_updates.count(input_def)) {
          replacement_defs[input_def] = input_def_updates[input_def];
        } else {
          // Add an fp32->fp16 node running on the CPU
          auto cpu_cast_output_arg = AddCastNode(graph,
                                                 input_def,
                                                 &float_16_tensor_proto,
                                                 false,
                                                 static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16),
                                                 onnxruntime::kCpuExecutionProvider);

          // Add an fp16->fp32 node running on the EP
          auto ep_cast_output_arg = AddCastNode(graph,
                                                cpu_cast_output_arg,
                                                &float_32_tensor_proto,
                                                false,
                                                static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
                                                node->GetExecutionProviderType());

          replacement_defs[input_def] = ep_cast_output_arg;
          input_def_updates[input_def] = ep_cast_output_arg;
          modified = true;
        }
      }
    }

    for (auto& output_def : node->MutableOutputDefs()) {
      if (!IsMLFloat32Tensor(*output_def) || !graph.IsOutput(output_def)) {
        continue;
      }

      if (output_def_updates.count(output_def)) {
        replacement_defs[output_def] = output_def_updates[output_def];
      } else {
        // Add an fp16->fp32 node running on the CPU
        auto cpu_cast_output_arg = AddCastNode(graph,
                                               output_def,
                                               &float_16_tensor_proto,
                                               true,
                                               static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
                                               onnxruntime::kCpuExecutionProvider);

        // Add an fp32->fp16 node running on the EP
        auto ep_cast_output_arg = AddCastNode(graph,
                                              cpu_cast_output_arg,
                                              &float_32_tensor_proto,
                                              true,
                                              static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16),
                                              node->GetExecutionProviderType());

        replacement_defs[output_def] = ep_cast_output_arg;
        output_def_updates[output_def] = ep_cast_output_arg;
        modified = true;
      }
    }

    node->ReplaceDefs(replacement_defs);
  }

  return Status::OK();
}

}  // namespace onnxruntime
