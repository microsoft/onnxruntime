// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_proto_serializer.h"

namespace onnxruntime {

void GraphViewerToProto(const GraphViewer& graph_view,
                        ONNX_NAMESPACE::GraphProto& graph_proto,
                        bool include_initializer,
                        bool include_outer_scope_args,
                        ExecutionOrder order,
                        bool include_initializer_data) {
  graph_proto.set_name(graph_view.Name());
  graph_proto.set_doc_string(graph_view.Description());

  for (const auto* input_arg : graph_view.GetInputsIncludingInitializers()) {
    *(graph_proto.mutable_input()->Add()) = input_arg->ToProto();
  }

  for (const auto* output_arg : graph_view.GetOutputs()) {
    *(graph_proto.mutable_output()->Add()) = output_arg->ToProto();
  }

  const auto& value_infos = graph_view.GetValueInfo();

  // Reserve memory for the vector to avoid reallocations
  InlinedVector<const NodeArg*> value_info_sorted;
  value_info_sorted.reserve(value_infos.size());
  value_info_sorted.assign(value_infos.begin(), value_infos.end());

  auto sort_predicate = [](const NodeArg* v1, const NodeArg* v2) {
    return v1->Name() < v2->Name();
  };

  // This ensures consistent ordering of value_info entries in the output graph
  std::sort(value_info_sorted.begin(), value_info_sorted.end(), sort_predicate);

  for (const auto* value_info : value_info_sorted) {
    *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
  }

  if (include_outer_scope_args) {
    // add the NodeArg info for outer scope NodeArgs so we capture the type information
    for (const auto& name : graph_view.GetOuterScopeNodeArgNames()) {
      auto* node_arg = graph_view.GetNodeArg(name);
      ORT_ENFORCE(node_arg, "Outer scope node arg name '" + name + "'was added but does not exist. ");
      *(graph_proto.mutable_value_info()->Add()) = node_arg->ToProto();
    }
  }

  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  for (auto& node_idx : graph_view.GetNodesInTopologicalOrder(order)) {
    const gsl::not_null<ONNX_NAMESPACE::NodeProto*> node_proto{graph_proto.add_node()};
    const gsl::not_null<const Node*> p_node{graph_view.GetNode(node_idx)};
    // we need to update any GraphProto attributes for subgraphs so that any changes made by things
    // such as the optimizers are captured. otherwise we can end up saving an invalid graph.
    p_node->ToProto(*node_proto, /* update_subgraphs */ true);
  }

  if (include_initializer) {
    const auto& initializers = graph_view.GetAllInitializedTensors();

    // Sort initializers to maintain consistency in model proto created across inference requests
    InlinedVector<InitializedTensorSet::const_iterator> const_inits;
    const_inits.reserve(initializers.size());
    for (auto it = initializers.cbegin(), end = initializers.cend(); it != end; ++it) {
      const_inits.push_back(it);
    }
    std::sort(const_inits.begin(), const_inits.end(), [](const auto& i1, const auto& i2) {
      return i1->first < i2->first;
    });

    InlinedHashSet<std::string_view> current_scope_initializer_set;
    current_scope_initializer_set.reserve(const_inits.size());

    auto get_initializer_with_data = [&](const ONNX_NAMESPACE::TensorProto& init,
                                         ONNX_NAMESPACE::TensorProto& dest) -> Status {
      std::unique_ptr<ONNX_NAMESPACE::TensorProto> full_init;
      ORT_RETURN_IF_ERROR(utils::GetTensorProtoWithDataIfInMemory(init, full_init));
      if (full_init) {
        dest = std::move(*full_init);
      } else {
        dest = init;
      }
      return Status::OK();
    };

    // Handle this scope initializers
    for (const auto& it : const_inits) {
      const auto& [name, init] = *it;
      current_scope_initializer_set.insert(name);
      auto* p_initializer = graph_proto.add_initializer();

      // Do not save raw or external data into the graph, only the metadata
      if (!include_initializer_data && (init->has_raw_data() || init->has_data_location())) {
        // Set datatype
        if (init->has_data_type()) {
          p_initializer->set_data_type(init->data_type());
        }
        // Set name
        if (init->has_name()) {
          p_initializer->set_name(init->name());
        }

        // Set dims
        for (int i = 0; i < init->dims_size(); ++i) {
          p_initializer->add_dims(init->dims()[i]);
        }
      } else {
        ORT_THROW_IF_ERROR(get_initializer_with_data(*init, *p_initializer));
      }
    }

    // handle outer scope value which is a constant initializer
    if (include_outer_scope_args) {
      for (auto& node_idx : graph_view.GetNodesInTopologicalOrder(order)) {
        const auto& node = graph_view.GetNode(node_idx);
        for (const auto& input : node->InputDefs()) {
          if (current_scope_initializer_set.count(std::string_view{input->Name()}) > 0) {
            continue;
          }

          const auto* outer_scope_init = graph_view.GetConstantInitializer(input->Name(), true);
          if (outer_scope_init != nullptr) {
            current_scope_initializer_set.insert(input->Name());
            auto* p_initializer = graph_proto.add_initializer();
            ORT_THROW_IF_ERROR(get_initializer_with_data(*outer_scope_init, *p_initializer));
          }
        }
      }
    }
  }
}

}  // namespace onnxruntime
