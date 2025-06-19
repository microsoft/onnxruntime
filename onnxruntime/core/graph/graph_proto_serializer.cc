// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_proto_serializer.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/abi_graph_types.h"
#include "core/framework/abi_pointer_array.h"
// Convert an OrtConstPointerArray into a span of Ort___ pointers.
template <typename T>
OrtStatus* GetSpanFromConstPointerArray(const OrtConstPointerArray* ort_array,
                                        /*out*/ gsl::span<const T* const>& span) {
  size_t size = 0;
  ORT_API_RETURN_IF_ERROR(OrtApis::ConstPointerArray_GetSize(ort_array, &size));

  const void* const* raw_data = nullptr;
  ORT_API_RETURN_IF_ERROR(OrtApis::ConstPointerArray_GetData(ort_array, &raw_data));

  auto data = reinterpret_cast<const T* const*>(raw_data);
  span = gsl::span<const T* const>(data, size);
  return nullptr;
}

/* Ex:
value_info {
name:
  "my_tensor" type {
    tensor_type {
    elem_type:
      FLOAT
      shape {
        dim{dim_value : 1}
        dim{dim_value : 3}
        dim{dim_value : 224}
        dim{dim_value : 224}
      }
    }
  }
}
*/

//
// OrtValueInfo to ValueInfoProto
//
void OrtValueInfoToProto(const OrtValueInfo* ort_value_info, ONNX_NAMESPACE::ValueInfoProto& value_info_proto) {
  const OrtApi& ort_api = Ort::GetApi();

  value_info_proto.set_name(ort_value_info->GetName());

  // Get the type
  auto type_info = ort_value_info->GetTypeInfo();
  const OrtTensorTypeAndShapeInfo* type_and_shape_info;
  ort_api.CastTypeInfoToTensorInfo(type_info, &type_and_shape_info);
  ONNXTensorElementDataType elem_type;
  ort_api.GetTensorElementType(type_and_shape_info, &elem_type);

  // Set the type (Tensor)
  ONNX_NAMESPACE::TypeProto* type_proto = value_info_proto.mutable_type();
  ONNX_NAMESPACE::TypeProto_Tensor* tensor_type = type_proto->mutable_tensor_type();
  tensor_type->set_elem_type(elem_type);

  //Get the shape
  size_t dim_cnt = 0;
  ort_api.GetDimensionsCount(type_and_shape_info, &dim_cnt);
  ort_api.GetDimensions(type_and_shape_info);
  // Set the shape
  ONNX_NAMESPACE::TensorShapeProto* shape = tensor_type->mutable_shape();
  for (int64_t d : dims) {
    shape->add_dim()->set_dim_value(d);
  }
}

//
// OrtGraph to GraphProto
//
void OrtGraphToProto(const OrtGraph* graph, ONNX_NAMESPACE::GraphProto& graph_proto) {
  const OrtApi& ort_api = Ort::GetApi();

  graph_proto.set_name(graph->GetName());
  //graph_proto.set_doc_string(graph_view.Description());

  const OrtConstPointerArray* initializers_container = nullptr;
  gsl::span<const OrtValueInfo* const> initializers{};

  auto status = ort_api.Graph_GetInitializers(graph, &initializers_container);
  GetSpanFromConstPointerArray<OrtValueInfo>(initializers_container, initializers);

}

namespace onnxruntime {

void GraphViewerToProto(const GraphViewer& graph_view,
                        ONNX_NAMESPACE::GraphProto& graph_proto,
                        bool include_initializer,
                        bool include_outer_scope_args,
                        ExecutionOrder order) {
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
      ORT_THROW_IF_ERROR(get_initializer_with_data(*init, *p_initializer));
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
