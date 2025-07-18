// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/ort_api.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace onnxruntime {

OrtNodeUnit::OrtNodeUnit(const OrtNode& node, const OrtApi& ort_api) : target_node_(node), type_(Type::SingleNode) {
  InitForSingleNode(ort_api);
}

void OrtNodeUnit::InitForSingleNode(const OrtApi& ort_api) {
  OrtArrayOfConstObjects* inputs_array = nullptr;
  OrtArrayOfConstObjects* outputs_array = nullptr;

  ort_api.Node_GetInputs(&target_node_, &inputs_array);
  ort_api.Node_GetOutputs(&target_node_, &outputs_array);

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  ort_api.ArrayOfConstObjects_GetSize(inputs_array, &num_inputs);
  ort_api.ArrayOfConstObjects_GetSize(outputs_array, &num_outputs);

  const void* const* inputs_data = nullptr;
  const void* const* outputs_data = nullptr;
  ort_api.ArrayOfConstObjects_GetData(inputs_array, &inputs_data);
  ort_api.ArrayOfConstObjects_GetData(outputs_array, &outputs_data);

  auto add_io_def = [&](std::vector<OrtNodeUnitIODef>& io_defs, const void* const* data, size_t num_data) {
    for (size_t idx = 0; idx < num_data; ++idx){
      const OrtValueInfo* io = static_cast<const OrtValueInfo*>(data[idx]);

      // Get name.
      const char* name = nullptr;
      ort_api.GetValueInfoName(io, &name);

      // Get type and shape.
      const OrtTypeInfo* type_info = nullptr;
      ort_api.GetValueInfoTypeInfo(io, &type_info);
      const OrtTensorTypeAndShapeInfo* type_shape = nullptr;
      ort_api.CastTypeInfoToTensorInfo(type_info, &type_shape);

      ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      ort_api.GetTensorElementType(type_shape, &elem_type);

      size_t num_dims = 0;
      ort_api.GetDimensionsCount(type_shape, &num_dims);

      std::vector<int64_t> shape;
      shape.resize(num_dims, 0);
      ort_api.GetDimensions(type_shape, shape.data(), shape.size());

      io_defs.push_back(OrtNodeUnitIODef{name, elem_type, shape});

      // TODO: SegFault if enabled release.
      // ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(type_shape));
      // ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info));
    }
  };

  inputs_.reserve(num_inputs);
  add_io_def(inputs_, inputs_data, num_inputs);

  outputs_.reserve(num_outputs);
  add_io_def(outputs_, outputs_data, num_outputs);

  ort_api.ReleaseArrayOfConstObjects(inputs_array);
  ort_api.ReleaseArrayOfConstObjects(outputs_array);

}

// std::vector<const Node*> Graph__Nodes(const Graph& graph) {
//   return graph.Nodes();
// }


#define NODE_ATTR_ITER_VAL(iter) (iter)->second()

OrtNodeAttrHelper::OrtNodeAttrHelper(const OrtApi& ort_api, const OrtNode& node) : node_(node), ort_api_(ort_api) {}

OrtNodeAttrHelper::OrtNodeAttrHelper(const OrtApi& ort_api, const OrtNodeUnit& node_unit) : node_(node_unit.GetNode()), ort_api_(ort_api) {}

float OrtNodeAttrHelper::Get(const std::string& key, float def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  return rt ? def_val : api_node_attr->attr_proto.f();
}

// int32_t OrtNodeAttrHelper::Get(const std::string& key, int32_t def_val) const {
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     return narrow<int32_t>(NODE_ATTR_ITER_VAL(entry).i());
//   }

//   return def_val;
// }

// uint32_t OrtNodeAttrHelper::Get(const std::string& key, uint32_t def_val) const {
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     return narrow<uint32_t>(NODE_ATTR_ITER_VAL(entry).i());
//   }

//   return def_val;
// }

int64_t OrtNodeAttrHelper::Get(const std::string& key, int64_t def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  return rt ? def_val : api_node_attr->attr_proto.i();
}

// const std::string& OrtNodeAttrHelper::Get(const std::string& key, const std::string& def_val) const {
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     return NODE_ATTR_ITER_VAL(entry).s();
//   }

//   return def_val;
// }

// std::vector<std::string> OrtNodeAttrHelper::Get(const std::string& key, const std::vector<std::string>& def_val) const {
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     std::vector<std::string> res;
//     for (int i = 0; i < NODE_ATTR_ITER_VAL(entry).strings_size(); i++) {
//       res.emplace_back(NODE_ATTR_ITER_VAL(entry).strings(i));
//     }
//     return res;
//   }

//   return def_val;
// }

// std::vector<int32_t> OrtNodeAttrHelper::Get(const std::string& key, const std::vector<int32_t>& def_val) const {
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
//     const int64_t* cbegin = values.data();
//     const int64_t* cend = values.data() + values.size();
//     std::vector<int32_t> v;
//     v.reserve(static_cast<size_t>(values.size()));
//     std::transform(cbegin, cend, std::back_inserter(v),
//                    [](int64_t val) -> int32_t { return narrow<int32_t>(val); });
//     return v;
//   }

//   return def_val;
// }

// std::vector<uint32_t> OrtNodeAttrHelper::Get(const std::string& key, const std::vector<uint32_t>& def_val) const {
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
//     const int64_t* cbegin = values.data();
//     const int64_t* cend = values.data() + values.size();
//     std::vector<uint32_t> v;
//     v.reserve(static_cast<size_t>(values.size()));
//     std::transform(cbegin, cend, std::back_inserter(v),
//                    [](int64_t val) -> uint32_t { return narrow<uint32_t>(val); });
//     return v;
//   }

//   return def_val;
// }

std::vector<int64_t> OrtNodeAttrHelper::Get(const std::string& key, const std::vector<int64_t>& def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return def_val;
  }

  const auto& ints_proto = api_node_attr->attr_proto.ints();
  std::vector<int64_t> result;
  result.reserve(ints_proto.size());
  for (int i = 0; i < ints_proto.size(); ++i) {
    result.push_back(ints_proto.Get(i));
  }
  return result;
}

// std::vector<float> OrtNodeAttrHelper::Get(const std::string& key, const std::vector<float>& def_val) const {
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     const auto& values = NODE_ATTR_ITER_VAL(entry).floats();
//     const float* cbegin = values.data();
//     const float* cend = values.data() + values.size();
//     return std::vector<float>{cbegin, cend};
//   }

//   return def_val;
// }

// std::optional<float> OrtNodeAttrHelper::GetFloat(const std::string& key) const {
//   std::optional<float> result;
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     result = NODE_ATTR_ITER_VAL(entry).f();
//   }

//   return result;
// }

// std::optional<int64_t> OrtNodeAttrHelper::GetInt64(const std::string& key) const {
//   std::optional<int64_t> result;
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     result = NODE_ATTR_ITER_VAL(entry).i();
//   }

//   return result;
// }

// std::optional<std::vector<float>> OrtNodeAttrHelper::GetFloats(const std::string& key) const {
//   std::optional<std::vector<float>> result;
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     const auto& values = NODE_ATTR_ITER_VAL(entry).floats();
//     const float* cbegin = values.data();
//     const float* cend = values.data() + values.size();
//     result = std::vector<float>(cbegin, cend);
//   }

//   return result;
// }

// std::optional<std::vector<int64_t>> OrtNodeAttrHelper::GetInt64s(const std::string& key) const {
//   std::optional<std::vector<int64_t>> result;
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     const auto& values = NODE_ATTR_ITER_VAL(entry).ints();
//     const int64_t* cbegin = values.data();
//     const int64_t* cend = values.data() + values.size();
//     result = std::vector<int64_t>(cbegin, cend);
//   }

//   return result;
// }

// std::optional<std::string> OrtNodeAttrHelper::GetString(const std::string& key) const {
//   std::optional<std::string> result;
//   if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
//     result = NODE_ATTR_ITER_VAL(entry).s();
//   }

//   return result;
// }

// bool OrtNodeAttrHelper::HasAttr(const std::string& key) const {
//   return node_attributes_.find(key) != node_attributes_.end();
// }

}  // namespace onnxruntime
