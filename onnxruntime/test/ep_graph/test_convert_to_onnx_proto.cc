// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_convert_to_onnx_proto.h"

#include <unordered_map>
#include <vector>

#define C_API_RETURN_IF_ERROR(fn)  \
  do {                             \
    OrtStatus* _status = (fn);     \
    if (_status != nullptr) {      \
      return Ort::Status{_status}; \
    }                              \
  } while (0)

#define CXX_API_RETURN_IF_ERROR(fn) \
  do {                              \
    Ort::Status _status = (fn);     \
    if (!_status.IsOK()) {          \
      return _status;               \
    }                               \
  } while (0)

#define C_API_RETURN_IF(cond, ort_api, msg)                        \
  do {                                                             \
    if ((cond)) {                                                  \
      return Ort::Status{(ort_api).CreateStatus(ORT_FAIL, (msg))}; \
    }                                                              \
  } while (0)

namespace test {

struct OrtValueInfoFlags {
  enum Flags {
    kFlagNone = 0,
    kIsRequiredGraphInput = 1 << 0,
    kIsOptionalGraphInput = 1 << 1,
    kIsGraphOutput = 1 << 2,
    kIsConstantInitializer = 1 << 3,
    kIsOuterScope = 1 << 4,
  };

  size_t flags = 0;

  bool IsRequiredGraphInput() const { return flags & kIsRequiredGraphInput; }
  bool IsOptionalGraphInput() const { return flags & kIsOptionalGraphInput; }
  bool IsGraphOutput() const { return flags & kIsGraphOutput; }
  bool IsConstantInitializer() const { return flags & kIsConstantInitializer; }
  bool IsFromOuterScope() const { return flags & kIsOuterScope; }
  bool IsInternal() const { return flags == 0; }
};

static Ort::Status GetOrtValueInfoFlags(const OrtValueInfo& ort_value_info, /*out*/ OrtValueInfoFlags& flags);
static Ort::Status GetOrtValueInfoName(const OrtValueInfo& ort_value_info, /*out*/ std::string& name);
static Ort::Status OrtValueInfoToProto(const OrtValueInfo& ort_value_info, onnx::ValueInfoProto& value_info_proto);

struct OrtValueInfoStorage {
  std::unordered_map<std::string, const OrtValueInfo*> map;
  std::vector<const OrtValueInfo*> array;

  bool HasValueInfo(const char* name) {
    return map.count(name) != 0;
  }

  Ort::Status AddOrGetName(const OrtValueInfo* ort_value_info, std::string& name) {
    const OrtApi& ort_api = Ort::GetApi();

    C_API_RETURN_IF(ort_value_info == nullptr, ort_api, "Tried to add NULL OrtValueInfo");

    const char* value_name = nullptr;
    C_API_RETURN_IF_ERROR(ort_api.GetValueInfoName(ort_value_info, &value_name));
    name = std::string{value_name};

    if (HasValueInfo(value_name)) {
      return Ort::Status{nullptr};
    }

    map.emplace(name, ort_value_info);
    array.push_back(ort_value_info);
  }

  Ort::Status Add(const OrtValueInfo* ort_value_info) {
    const OrtApi& ort_api = Ort::GetApi();

    C_API_RETURN_IF(ort_value_info == nullptr, ort_api, "Tried to add NULL OrtValueInfo");

    const char* value_name = nullptr;
    C_API_RETURN_IF_ERROR(ort_api.GetValueInfoName(ort_value_info, &value_name));
    C_API_RETURN_IF(HasValueInfo(value_name), ort_api, "Tried to add existing OrtValueInfo");

    map.emplace(std::string{value_name}, ort_value_info);
    array.push_back(ort_value_info);
  }
};

Ort::Status OrtGraphToProto(const OrtGraph& ort_graph,
                            onnx::GraphProto& graph_proto,
                            bool save_as_external_data,
                            const ORTCHAR_T* external_data_location,
                            size_t external_data_size_threshold) {
  const OrtApi& ort_api = Ort::GetApi();

  OrtValueInfoStorage activation_ort_value_info_storage;

  (void)save_as_external_data;
  (void)external_data_location;
  (void)external_data_size_threshold;

  //
  // Set GraphProto metadata
  //
  const char* graph_name = nullptr;
  C_API_RETURN_IF_ERROR(ort_api.Graph_GetName(&ort_graph, &graph_name));
  graph_proto.set_name(graph_name);

  //
  // Set GraphProto inputs and outputs
  //
  size_t num_graph_inputs = 0;
  size_t num_graph_outputs = 0;
  C_API_RETURN_IF_ERROR(ort_api.Graph_GetNumInputs(&ort_graph, &num_graph_inputs));
  C_API_RETURN_IF_ERROR(ort_api.Graph_GetNumOutputs(&ort_graph, &num_graph_outputs));

  std::vector<const OrtValueInfo*> graph_inputs(num_graph_inputs);
  std::vector<const OrtValueInfo*> graph_outputs(num_graph_outputs);
  C_API_RETURN_IF_ERROR(ort_api.Graph_GetInputs(&ort_graph, graph_inputs.data(), graph_inputs.size()));
  C_API_RETURN_IF_ERROR(ort_api.Graph_GetOutputs(&ort_graph, graph_outputs.data(), graph_outputs.size()));

  for (const OrtValueInfo* ort_value_info : graph_inputs) {
    onnx::ValueInfoProto* value_info_proto = graph_proto.mutable_input()->Add();
    CXX_API_RETURN_IF_ERROR(OrtValueInfoToProto(*ort_value_info, *value_info_proto));
  }

  for (const OrtValueInfo* ort_value_info : graph_outputs) {
    onnx::ValueInfoProto* value_info_proto = graph_proto.mutable_output()->Add();
    CXX_API_RETURN_IF_ERROR(OrtValueInfoToProto(*ort_value_info, *value_info_proto));
  }

  //
  // Set GraphProto value_infos, nodes, initializers.
  //

  size_t num_nodes = 0;
  C_API_RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(&ort_graph, &num_nodes));

  std::vector<const OrtNode*> nodes(num_nodes);
  C_API_RETURN_IF_ERROR(ort_api.Graph_GetNodes(&ort_graph, nodes.data(), nodes.size()));

  for (size_t i = 0; i < num_nodes; i++) {
    const OrtNode* ort_node = nodes[i];
    onnx::NodeProto* node_proto = graph_proto.add_node();

    size_t num_inputs = 0;
    size_t num_implicit_inputs = 0;
    size_t num_outputs = 0;
    C_API_RETURN_IF_ERROR(ort_api.Node_GetNumInputs(ort_node, &num_inputs));
    C_API_RETURN_IF_ERROR(ort_api.Node_GetNumImplicitInputs(ort_node, &num_implicit_inputs));
    C_API_RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(ort_node, &num_outputs));

    if (num_inputs > 0) {
      std::vector<const OrtValueInfo*> ort_inputs(num_inputs);
      C_API_RETURN_IF_ERROR(ort_api.Node_GetInputs(ort_node, ort_inputs.data(), ort_inputs.size()));

      for (const OrtValueInfo* ort_value_info : ort_inputs) {
        if (ort_value_info == nullptr) {
          // missing optional input.
          node_proto->add_input("");
          continue;
        }

        std::string ort_value_info_name;
        CXX_API_RETURN_IF_ERROR(GetOrtValueInfoName(*ort_value_info, ort_value_info_name));
        node_proto->add_input(ort_value_info_name);

        OrtValueInfoFlags flags = {};
        CXX_API_RETURN_IF_ERROR(GetOrtValueInfoFlags(*ort_value_info, flags));

#if 0
        if (flags.is_required_graph_input || flags.is_graph_output) {
          // Don't add graph inputs or outputs to graph_proto's list of value_infos.
          // Do nothing, but do not skip because a graph output "could" come from a constant initializer.
        }

        if (flags.is_optional_graph_input) {
          // Don't add optional graph inputs to graph_proto's list of value_infos.
          // However, we need to add the non-constant initializer to graph_proto's list initializer tensors.
          // TODO
        }

        if (flags.is_constant_initializer) {
          // Add constant initializer's value_info to graph_proto's list of value_infos.
          // Add constant initializer's data to graph_proto's list of initializer tensors.
          // TODO
        }

        if (flags.is_from_outer_scope) {
          // Don't need to handle this explicitly. If this is a constant initializer from an outer scope, then we've
          // handled it above.
        }

        if (flags.IsInternal()) {
        }
#endif

        //std::string ort_value_info_name;
        //activation_ort_value_info_storage.AddOrGetName(ort_value_info, ort_value_info_name);
      }
    }
  }

  return Ort::Status{nullptr};
}

static Ort::Status GetOrtValueInfoFlags(const OrtValueInfo& ort_value_info, OrtValueInfoFlags& result) {
  const OrtApi& ort_api = Ort::GetApi();
  OrtValueInfoFlags flags = {};

  bool is_required_graph_input = false;
  bool is_optional_graph_input = false;
  bool is_graph_output = false;
  bool is_constant_initializer = false;
  bool is_from_outer_scope = false;

  C_API_RETURN_IF_ERROR(ort_api.ValueInfo_IsRequiredGraphInput(&ort_value_info, &is_required_graph_input));
  C_API_RETURN_IF_ERROR(ort_api.ValueInfo_IsOptionalGraphInput(&ort_value_info, &is_optional_graph_input));
  C_API_RETURN_IF_ERROR(ort_api.ValueInfo_IsGraphOutput(&ort_value_info, &is_graph_output));
  C_API_RETURN_IF_ERROR(ort_api.ValueInfo_IsConstantInitializer(&ort_value_info, &is_constant_initializer));
  C_API_RETURN_IF_ERROR(ort_api.ValueInfo_IsFromOuterScope(&ort_value_info, &is_from_outer_scope));

  if (is_required_graph_input) {
    flags.flags |= OrtValueInfoFlags::kIsRequiredGraphInput;
  }

  if (is_optional_graph_input) {
    flags.flags |= OrtValueInfoFlags::kIsOptionalGraphInput;
  }

  result = flags;
  return Ort::Status{nullptr};
}

static Ort::Status GetOrtValueInfoName(const OrtValueInfo& ort_value_info, /*out*/ std::string& name) {
  const OrtApi& ort_api = Ort::GetApi();
  const char* name_c_str = nullptr;
  C_API_RETURN_IF_ERROR(ort_api.GetValueInfoName(&ort_value_info, &name_c_str));

  name = name_c_str;
  return Ort::Status{nullptr};
}

static Ort::Status OrtValueInfoToProto(const OrtValueInfo& ort_value_info,
                                       onnx::ValueInfoProto& value_info_proto) {
  const OrtApi& ort_api = Ort::GetApi();

  const OrtTypeInfo* ort_type_info = nullptr;
  C_API_RETURN_IF_ERROR(ort_api.GetValueInfoTypeInfo(&ort_value_info, &ort_type_info));

  ONNXType ort_onnx_type = ONNX_TYPE_UNKNOWN;
  C_API_RETURN_IF_ERROR(ort_api.GetOnnxTypeFromTypeInfo(ort_type_info, &ort_onnx_type));

  // We currently only support ONNX tensors. Support for other types (e.g., ONNX_TYPE_SEQUENCE) can be added later.
  C_API_RETURN_IF(ort_onnx_type != ONNX_TYPE_TENSOR, ort_api,
                  "Internal error: OrtValueInfoToProto currently only supports ONNX_TYPE_TENSOR");

  std::string value_name;
  CXX_API_RETURN_IF_ERROR(GetOrtValueInfoName(ort_value_info, value_name));
  value_info_proto.set_name(value_name);

  onnx::TypeProto_Tensor* type_proto_tensor = value_info_proto.mutable_type()->mutable_tensor_type();

  const OrtTensorTypeAndShapeInfo* ort_type_shape = nullptr;
  ONNXTensorElementDataType ort_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  C_API_RETURN_IF_ERROR(ort_api.CastTypeInfoToTensorInfo(ort_type_info, &ort_type_shape));
  C_API_RETURN_IF_ERROR(ort_api.GetTensorElementType(ort_type_shape, &ort_elem_type));

  type_proto_tensor->set_elem_type(ort_elem_type);

  size_t ort_num_dims = 0;
  C_API_RETURN_IF_ERROR(ort_api.GetDimensionsCount(ort_type_shape, &ort_num_dims));

  std::vector<int64_t> ort_dims(ort_num_dims, 0);
  C_API_RETURN_IF_ERROR(ort_api.GetDimensions(ort_type_shape, ort_dims.data(), ort_dims.size()));

  std::vector<const char*> ort_dim_syms(ort_num_dims, nullptr);
  C_API_RETURN_IF_ERROR(ort_api.GetSymbolicDimensions(ort_type_shape, ort_dim_syms.data(), ort_dim_syms.size()));

  onnx::TensorShapeProto* shape_proto = type_proto_tensor->mutable_shape();

  for (size_t dim_idx = 0; dim_idx < ort_num_dims; dim_idx++) {
    onnx::TensorShapeProto_Dimension* dim_proto = shape_proto->add_dim();

    if (ort_dims[dim_idx] >= 0) {
      dim_proto->set_dim_value(ort_dims[dim_idx]);
    } else {
      const std::string dim_param = ort_dim_syms[dim_idx];

      // If dim_param is empty, leave dim_proto with neither the dim_value or dim_param set,
      // which represents an unknown dimension.
      if (!dim_param.empty()) {
        dim_proto->set_dim_param(dim_param);
      }
    }
  }

  return Ort::Status{nullptr};
}
}  // namespace test
