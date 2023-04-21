// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/identical_children_consolidation.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
Status IdenticalChildrenConsolidation::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& /*logger*/) const {
  GraphViewer const graph_viewer(graph);
  for (auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    Node* node = graph.GetNode(node_index);
    if (!IsSupportedParentNode(node)) {
      continue;
    }
    for (auto supported_op : supported_ops.at(node->OpType())) {
      for (auto twin_group : DivideIdenticalChildrenIntoGroups(graph, node, supported_op)) {
        // If there is no twins in the group, skip it.
        if (twin_group.size() <= 1) {
          continue;
        }
        Node* first_twin = graph.GetNode(twin_group[0]);
        for (size_t i = 1; i < twin_group.size(); i++) {
          Node* other_twin = graph.GetNode(twin_group[i]);
          if (graph.NodeProducesGraphOutput(*other_twin)) {
            continue;
          }
          graph_utils::ReplaceDownstreamNodeInput(graph, *other_twin, 0, *first_twin, 0);
          graph_utils::RemoveNode(graph, *other_twin);
          modified = true;
        }
      }
    }
  }
  return Status::OK();
}

bool IdenticalChildrenConsolidation::IsSupportedParentNode(const Node* node) const {
  return node != nullptr && supported_ops.count(node->OpType()) != 0 && node->GetOutputEdgesCount() > 1;
}

std::vector<std::vector<NodeIndex>> IdenticalChildrenConsolidation::DivideIdenticalChildrenIntoGroups(
    const Graph& graph,
    Node* node,
    const string_view& op) const {
  unordered_map<string_view, std::vector<NodeIndex>> identical_children_map;
  for (auto i = node->OutputEdgesBegin(); i != node->OutputEdgesEnd(); ++i) {
    if (i->GetNode().OpType() == op) {
      identical_children_map[IdentityBuilder(graph, i->GetNode())].push_back(i->GetNode().Index());
    }
  }
  std::vector<std::vector<NodeIndex>> groups;
  for (auto& identical_children : identical_children_map) {
    if (identical_children.first != ignore_identity) {
      groups.push_back(std::move(identical_children.second));
    }
  }
  return groups;
}

string_view IdenticalChildrenConsolidation::IdentityBuilder(const Graph& graph, const Node& node) const {
  std::string identity;
  for (const auto* input_def : node.InputDefs()) {
    if (input_def->Exists() && !input_def->Name().empty()) {
      auto name = input_def->Name();
      if (graph_utils::NodeArgIsConstant(graph, *input_def)) {
        if (optimizer_utils::IsScalar(*input_def)) {
          const auto* data = graph_utils::GetConstantInitializer(graph, name);
          identity.append(constant_prefix);
          Initializer value{*data, graph.ModelPath()};
          switch (static_cast<TensorProto::DataType>(data->data_type())) {
            case TensorProto::DataType::TensorProto_DataType_INT8:
              identity.append(std::to_string(value.data<int8_t>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_INT16:
              identity.append(std::to_string(value.data<int16_t>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_INT32:
              identity.append(std::to_string(value.data<int32_t>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_UINT8:
              identity.append(std::to_string(value.data<uint8_t>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_UINT16:
              identity.append(std::to_string(value.data<uint16_t>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_BOOL:
              identity.append(std::to_string(value.data<bool>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_INT64:
              identity.append(std::to_string(value.data<int64_t>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_UINT32:
              identity.append(std::to_string(value.data<uint32_t>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_UINT64:
              identity.append(std::to_string(value.data<uint64_t>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_FLOAT:
              identity.append(std::to_string(value.data<float>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_DOUBLE:
              identity.append(std::to_string(value.data<double>()[0]));
              break;
            case TensorProto::DataType::TensorProto_DataType_STRING:
              identity.append(value.data<std::string>()[0]);
              break;
            default:
              break;
          }
        } else {
          // TODO: handle non-scalar constant inputs, using checksum or something else
          return ignore_identity;
        }
      } else {
        identity.append(name);
      }
    } else {
      return ignore_identity;
    }
    identity.append("####");
  }
  return {identity};
}
}  // namespace onnxruntime