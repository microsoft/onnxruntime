// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "graph_flatbuffers_utils.h"
#include "core/framework/tensorprotoutils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace ::onnxruntime::experimental;

namespace onnxruntime {
namespace experimental {
namespace utils {

#if !defined(ORT_MINIMAL_BUILD)
static Status SaveTypeInfoOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                    const TypeProto& type_proto,
                                    flatbuffers::Offset<fbs::TypeInfo>& fbs_type_info) ORT_MUST_USE_RESULT;

static flatbuffers::Offset<fbs::Dimension> SaveTensorDimensionOrtFormat(
    flatbuffers::FlatBufferBuilder& builder,
    const TensorShapeProto_Dimension& tensor_shape_dim) {
  auto denotation = builder.CreateString(tensor_shape_dim.denotation());
  flatbuffers::Offset<fbs::DimensionValue> dim_val;
  if (tensor_shape_dim.has_dim_param()) {
    dim_val = fbs::CreateDimensionValueDirect(builder, fbs::DimensionValueType::PARAM, 0, tensor_shape_dim.dim_param().c_str());
  } else if (tensor_shape_dim.has_dim_value()) {
    dim_val = fbs::CreateDimensionValueDirect(builder, fbs::DimensionValueType::VALUE, tensor_shape_dim.dim_value());
  } else {
    dim_val = fbs::CreateDimensionValueDirect(builder);
  }

  return fbs::CreateDimension(builder, dim_val, denotation);
}

static Status SaveTensorShapeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                       const TensorShapeProto& tensor_shape_proto,
                                       flatbuffers::Offset<fbs::Shape>& fbs_shape) {
  std::vector<flatbuffers::Offset<fbs::Dimension>> dim;
  dim.reserve(tensor_shape_proto.dim_size());
  for (const auto& d : tensor_shape_proto.dim()) {
    auto fbs_d = SaveTensorDimensionOrtFormat(builder, d);
    dim.push_back(fbs_d);
  }
  fbs_shape = fbs::CreateShapeDirect(builder, &dim);
  return Status::OK();
}

static Status SaveSequenceTypeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                        const TypeProto_Sequence& sequence_type_proto,
                                        flatbuffers::Offset<fbs::SequenceType>& fbs_sequence_type) {
  flatbuffers::Offset<fbs::TypeInfo> fbs_type_info;
  ORT_RETURN_IF_ERROR(SaveTypeInfoOrtFormat(builder, sequence_type_proto.elem_type(), fbs_type_info));
  fbs_sequence_type = fbs::CreateSequenceType(builder, fbs_type_info);
  return Status::OK();
}

static Status SaveMapTypeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                   const TypeProto_Map& map_type_proto,
                                   flatbuffers::Offset<fbs::MapType>& fbs_map_type) {
  flatbuffers::Offset<fbs::TypeInfo> fbs_type_info;
  ORT_RETURN_IF_ERROR(SaveTypeInfoOrtFormat(builder, map_type_proto.value_type(), fbs_type_info));
  fbs_map_type = fbs::CreateMapType(
      builder, static_cast<fbs::TensorDataType>(map_type_proto.key_type()), fbs_type_info);
  return Status::OK();
}

static Status SaveTensorTypeAndShapeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                              const TypeProto_Tensor& tensor_type_proto,
                                              flatbuffers::Offset<fbs::TensorTypeAndShape>& fbs_tensor_type) {
  // A flatbuffers::Offset of 0 means this shape is missing (was null when serializing)
  flatbuffers::Offset<fbs::Shape> shape = 0;
  if (tensor_type_proto.has_shape()) {
    ORT_RETURN_IF_ERROR(SaveTensorShapeOrtFormat(builder, tensor_type_proto.shape(), shape));
  }

  fbs_tensor_type = fbs::CreateTensorTypeAndShape(
      builder, static_cast<fbs::TensorDataType>(tensor_type_proto.elem_type()), shape);

  return Status::OK();
}

static Status SaveTypeInfoOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                    const TypeProto& type_proto,
                                    flatbuffers::Offset<fbs::TypeInfo>& fbs_type_info) {
  auto denotation = builder.CreateString(type_proto.denotation());
  auto value_type = fbs::TypeInfoValue::tensor_type;
  flatbuffers::Offset<void> value;
  auto value_case = type_proto.value_case();
  switch (value_case) {
    case TypeProto::kTensorType: {
      flatbuffers::Offset<fbs::TensorTypeAndShape> fbs_tensor_type;
      ORT_RETURN_IF_ERROR(
          SaveTensorTypeAndShapeOrtFormat(builder, type_proto.tensor_type(), fbs_tensor_type));
      value = fbs_tensor_type.Union();
    } break;
    case TypeProto::kSequenceType: {
      value_type = fbs::TypeInfoValue::sequence_type;
      flatbuffers::Offset<fbs::SequenceType> fbs_sequence_type;
      ORT_RETURN_IF_ERROR(
          SaveSequenceTypeOrtFormat(builder, type_proto.sequence_type(), fbs_sequence_type));
      value = fbs_sequence_type.Union();
    } break;
    case TypeProto::kMapType: {
      value_type = fbs::TypeInfoValue::map_type;
      flatbuffers::Offset<fbs::MapType> fbs_map_type;
      ORT_RETURN_IF_ERROR(
          SaveMapTypeOrtFormat(builder, type_proto.map_type(), fbs_map_type));
      value = fbs_map_type.Union();
    } break;
    default: {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "We do not support type [", value_case, "] for now");
    } break;
  }

  fbs::TypeInfoBuilder tb(builder);
  tb.add_denotation(denotation);
  tb.add_value_type(value_type);
  tb.add_value(value);
  fbs_type_info = tb.Finish();
  return Status::OK();
}

Status SaveValueInfoOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                              const ValueInfoProto& value_info_proto,
                              flatbuffers::Offset<fbs::ValueInfo>& fbs_value_info) {
  auto name = builder.CreateSharedString(value_info_proto.name());
  auto doc_string = builder.CreateString(value_info_proto.doc_string());
  flatbuffers::Offset<fbs::TypeInfo> type_info = 0;  // 0 indicates null
  if (value_info_proto.has_type()) {
    ORT_RETURN_IF_ERROR(
        SaveTypeInfoOrtFormat(builder, value_info_proto.type(), type_info));
  } else {
    // we have a NodeArg for missing optional values (empty name, no type) so allow for that.
    // everything else should have type info
    if (!value_info_proto.name().empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "SaveValueInfoOrtFormat: value_info_proto for ", value_info_proto.name(),
                             " is missing type info.");
    }
  }

  fbs::ValueInfoBuilder vb(builder);
  vb.add_name(name);
  vb.add_doc_string(doc_string);
  vb.add_type(type_info);
  fbs_value_info = vb.Finish();
  return Status::OK();
}

Status SaveInitializerOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                const TensorProto& initializer,
                                flatbuffers::Offset<fbs::Tensor>& fbs_tensor) {
  auto name = builder.CreateString(initializer.name());
  auto doc_string = builder.CreateString(initializer.doc_string());
  std::vector<int64_t> dims_data(initializer.dims().size());
  std::copy(initializer.dims().cbegin(), initializer.dims().cend(), dims_data.begin());
  auto dims = builder.CreateVector(dims_data);
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> string_data;
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> raw_data;

  auto src_type = initializer.data_type();
  bool has_string_data = src_type == ONNX_NAMESPACE::TensorProto_DataType_STRING;
  if (has_string_data) {
    std::vector<std::string> string_data_vec(initializer.string_data().size());
    std::copy(initializer.string_data().cbegin(), initializer.string_data().cend(), string_data_vec.begin());
    string_data = builder.CreateVectorOfStrings(string_data_vec);
  } else {
    std::unique_ptr<uint8_t[]> unpacked_tensor;
    size_t tensor_byte_size = 0;
    ORT_RETURN_IF_ERROR(
        onnxruntime::utils::UnpackInitializerData(initializer, unpacked_tensor, tensor_byte_size));
    raw_data = builder.CreateVector(unpacked_tensor.get(), tensor_byte_size);
  }

  fbs::TensorBuilder tb(builder);
  tb.add_name(name);
  tb.add_doc_string(doc_string);
  tb.add_dims(dims);
  tb.add_data_type(static_cast<fbs::TensorDataType>(src_type));
  if (has_string_data)
    tb.add_string_data(string_data);
  else
    tb.add_raw_data(raw_data);
  fbs_tensor = tb.Finish();
  return Status::OK();
}

#define GET_FBS_ATTR(BUILDER, TYPE, DATA_NAME, DATA) \
  fbs::AttributeBuilder attr_builder(BUILDER);       \
  attr_builder.add_name(name);                       \
  attr_builder.add_doc_string(doc_string);           \
  attr_builder.add_type(TYPE);                       \
  attr_builder.add_##DATA_NAME(DATA);                \
  fbs_attr = attr_builder.Finish();

#define GET_DATA_VEC(TYPE, NAME, SRC_DATA) \
  std::vector<TYPE> NAME(SRC_DATA.size()); \
  std::copy(SRC_DATA.cbegin(), SRC_DATA.cend(), NAME.begin());

Status SaveAttributeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                              const AttributeProto& attr_proto,
                              flatbuffers::Offset<fbs::Attribute>& fbs_attr,
                              const onnxruntime::Graph* graph) {
  auto name = builder.CreateString(attr_proto.name());
  auto doc_string = builder.CreateString(attr_proto.doc_string());
  auto type = static_cast<fbs::AttributeType>(attr_proto.type());
  switch (type) {
    case fbs::AttributeType::FLOAT: {
      GET_FBS_ATTR(builder, type, f, attr_proto.f());
    } break;
    case fbs::AttributeType::INT: {
      GET_FBS_ATTR(builder, type, i, attr_proto.i());
    } break;
    case fbs::AttributeType::STRING: {
      auto s = builder.CreateString(attr_proto.s());
      GET_FBS_ATTR(builder, type, s, s);
    } break;
    case fbs::AttributeType::TENSOR: {
      flatbuffers::Offset<fbs::Tensor> fbs_tensor;
      ORT_RETURN_IF_ERROR(
          experimental::utils::SaveInitializerOrtFormat(builder, attr_proto.t(), fbs_tensor));
      GET_FBS_ATTR(builder, type, t, fbs_tensor);
    } break;
    case fbs::AttributeType::GRAPH: {
      ORT_RETURN_IF(nullptr == graph, "Graph attribute value was null. Invalid ORT format model.");
      flatbuffers::Offset<fbs::Graph> fbs_graph;
      ORT_RETURN_IF_ERROR(graph->SaveToOrtFormat(builder, fbs_graph));
      GET_FBS_ATTR(builder, type, g, fbs_graph);
    } break;
    case fbs::AttributeType::FLOATS: {
      GET_DATA_VEC(float, floats_vec_, attr_proto.floats());
      auto floats = builder.CreateVector(floats_vec_);
      GET_FBS_ATTR(builder, type, floats, floats);
    } break;
    case fbs::AttributeType::INTS: {
      GET_DATA_VEC(int64_t, ints_vec_, attr_proto.ints());
      auto ints = builder.CreateVector(ints_vec_);
      GET_FBS_ATTR(builder, type, ints, ints);
    } break;
    case fbs::AttributeType::STRINGS: {
      GET_DATA_VEC(std::string, strings_vec_, attr_proto.strings());
      auto strings = builder.CreateVectorOfStrings(strings_vec_);
      GET_FBS_ATTR(builder, type, strings, strings);
    } break;
    case fbs::AttributeType::TENSORS: {
      std::vector<flatbuffers::Offset<fbs::Tensor>> fbs_tensors_vec;
      fbs_tensors_vec.reserve(attr_proto.tensors().size());
      for (const auto& tensor : attr_proto.tensors()) {
        flatbuffers::Offset<fbs::Tensor> fbs_tensor;
        ORT_RETURN_IF_ERROR(
            experimental::utils::SaveInitializerOrtFormat(builder, tensor, fbs_tensor));
        fbs_tensors_vec.push_back(fbs_tensor);
      }
      auto tensors = builder.CreateVector(fbs_tensors_vec);
      GET_FBS_ATTR(builder, type, tensors, tensors);
    } break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "SaveAttributeOrtFormat: Unsupported attribute type: ", fbs::EnumNameAttributeType(type));
      break;
  }

  return Status::OK();
}

#endif

#undef GET_FBS_ATTR
#undef GET_DATA_VEC

#if defined(ENABLE_ORT_FORMAT_LOAD)

static Status LoadTypeInfoOrtFormat(const fbs::TypeInfo& fbs_type_info,
                                    TypeProto& type_proto) ORT_MUST_USE_RESULT;

Status LoadInitializerOrtFormat(const fbs::Tensor& fbs_tensor,
                                TensorProto& initializer) {
  initializer.Clear();

  LoadStringFromOrtFormat(*initializer.mutable_name(), fbs_tensor.name());
  LoadStringFromOrtFormat(*initializer.mutable_doc_string(), fbs_tensor.doc_string());

  auto fbs_dims = fbs_tensor.dims();
  ORT_RETURN_IF(nullptr == fbs_dims, "Missing dimensions for initializer. Invalid ORT format model.");
  initializer.mutable_dims()->Add(fbs_dims->cbegin(), fbs_dims->cend());

  auto fbs_data_type = fbs_tensor.data_type();
  initializer.set_data_type(static_cast<int32_t>(fbs_data_type));
  if (fbs_data_type == fbs::TensorDataType::STRING) {
    auto fbs_str_data = fbs_tensor.string_data();
    ORT_RETURN_IF(nullptr == fbs_str_data, "Missing string data for initializer. Invalid ORT format model.");
    auto mutable_str_data = initializer.mutable_string_data();
    mutable_str_data->Reserve(fbs_str_data->size());
    for (const auto& fbs_str : *fbs_str_data) {
      mutable_str_data->Add(fbs_str->str());
    }
  } else {
    auto fbs_raw_data = fbs_tensor.raw_data();
    ORT_RETURN_IF(nullptr == fbs_raw_data, "Missing raw data for initializer. Invalid ORT format model.");

    // fbs_raw_data is uint8_t vector, so the size is byte size
    initializer.set_raw_data(fbs_raw_data->Data(), fbs_raw_data->size());
  }

  return Status::OK();
}

static Status LoadTensorDimensionOrtFormat(const fbs::Dimension& fbs_dim,
                                           TensorShapeProto_Dimension& dim) {
  LoadStringFromOrtFormat(*dim.mutable_denotation(), fbs_dim.denotation());
  auto fbs_dim_val = fbs_dim.value();
  if (fbs_dim_val) {
    auto type = fbs_dim_val->dim_type();
    if (type == fbs::DimensionValueType::VALUE)
      dim.set_dim_value(fbs_dim_val->dim_value());
    else if (type == fbs::DimensionValueType::PARAM) {
      auto fbs_dim_param = fbs_dim_val->dim_param();
      ORT_RETURN_IF(nullptr == fbs_dim_param, "dim_param value with no name. Invalid ORT format model.");
      dim.set_dim_param(fbs_dim_param->str());
    } else {
      // unknown dimension. leave dim in VALUE_NOT_SET state as this is valid
    }
  } else {
    // tensor with unknown shape.
    // e.g. output from Reshape node where shape is determined by dynamic input at runtime
  }
  return Status::OK();
}

static Status LoadTensorTypeAndShapeOrtFormat(const fbs::TensorTypeAndShape& fbs_tensor_type,
                                              TypeProto_Tensor& tensor_type_proto) {
  tensor_type_proto.set_elem_type(static_cast<int32_t>(fbs_tensor_type.elem_type()));
  auto fbs_shape = fbs_tensor_type.shape();
  if (fbs_shape) {
    auto fbs_dims = fbs_shape->dim();
    if (fbs_dims) {
      auto dims = tensor_type_proto.mutable_shape()->mutable_dim();
      dims->Reserve(fbs_dims->size());
      for (const auto fbs_dim : *fbs_dims) {
        ORT_RETURN_IF(nullptr == fbs_dim, "Null entry in dimensions. Invalid ORT format model.");
        TensorShapeProto_Dimension dim;
        ORT_RETURN_IF_ERROR(LoadTensorDimensionOrtFormat(*fbs_dim, *dims->Add()));
      }
    }
  }
  return Status::OK();
}

static Status LoadSequenceTypeOrtFormat(const fbs::SequenceType& fbs_sequence_type,
                                        TypeProto_Sequence& sequence_type_proto) {
  auto fbs_type_info = fbs_sequence_type.elem_type();
  ORT_RETURN_IF(nullptr == fbs_type_info, "Null value type info in fbs::SequenceType. Invalid ORT format model.");
  ORT_RETURN_IF_ERROR(LoadTypeInfoOrtFormat(*fbs_type_info, *sequence_type_proto.mutable_elem_type()));
  return Status::OK();
}

static Status LoadMapTypeOrtFormat(const fbs::MapType& fbs_map_type,
                                   TypeProto_Map& map_type_proto) {
  map_type_proto.set_key_type(static_cast<int32_t>(fbs_map_type.key_type()));
  auto fbs_type_info = fbs_map_type.value_type();
  ORT_RETURN_IF(nullptr == fbs_type_info, "Null value type info in fbs::MapType. Invalid ORT format model.");
  ORT_RETURN_IF_ERROR(LoadTypeInfoOrtFormat(*fbs_type_info, *map_type_proto.mutable_value_type()));
  return Status::OK();
}

static Status LoadTypeInfoOrtFormat(const fbs::TypeInfo& fbs_type_info,
                                    TypeProto& type_proto) {
  LoadStringFromOrtFormat(*type_proto.mutable_denotation(), fbs_type_info.denotation());
  auto value_type = fbs_type_info.value_type();
  if (value_type == fbs::TypeInfoValue::tensor_type) {
    auto fbs_tensor_type = fbs_type_info.value_as_tensor_type();
    ORT_RETURN_IF(nullptr == fbs_tensor_type, "Null tensor type info. Invalid ORT format model.");
    ORT_RETURN_IF_ERROR(LoadTensorTypeAndShapeOrtFormat(*fbs_tensor_type, *type_proto.mutable_tensor_type()));
  } else if (value_type == fbs::TypeInfoValue::sequence_type) {
    auto fbs_sequence_type = fbs_type_info.value_as_sequence_type();
    ORT_RETURN_IF(nullptr == fbs_sequence_type, "Null sequence type info. Invalid ORT format model.");
    ORT_RETURN_IF_ERROR(LoadSequenceTypeOrtFormat(*fbs_sequence_type, *type_proto.mutable_sequence_type()));
  } else if (value_type == fbs::TypeInfoValue::map_type) {
    auto fbs_map_type = fbs_type_info.value_as_map_type();
    ORT_RETURN_IF(nullptr == fbs_map_type, "Null map type info. Invalid ORT format model.");
    ORT_RETURN_IF_ERROR(LoadMapTypeOrtFormat(*fbs_map_type, *type_proto.mutable_map_type()));
  } else {
    // We do not support SparseTensor and Opaque for now
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Type:",
                           fbs::EnumNameTypeInfoValue(value_type), " is not supported currently");
  }

  return Status::OK();
}

Status LoadValueInfoOrtFormat(const fbs::ValueInfo& fbs_value_info,
                              ONNX_NAMESPACE::ValueInfoProto& value_info_proto) {
  value_info_proto.Clear();

  LoadStringFromOrtFormat(*value_info_proto.mutable_name(), fbs_value_info.name());
  LoadStringFromOrtFormat(*value_info_proto.mutable_doc_string(), fbs_value_info.doc_string());

  auto fbs_type_info = fbs_value_info.type();
  if (fbs_type_info == nullptr) {
    // there is a NodeArg with empty name for missing optional inputs that can have null type info.
    // anything else should have a type
    ORT_RETURN_IF(!value_info_proto.name().empty(),
                  "Null type info for ", value_info_proto.name(), ". Invalid ORT format model.");
  } else {
    ORT_RETURN_IF_ERROR(LoadTypeInfoOrtFormat(*fbs_type_info, *value_info_proto.mutable_type()));
  }

  return Status::OK();
}

Status LoadAttributeOrtFormat(const fbs::Attribute& fbs_attr,
                              ONNX_NAMESPACE::AttributeProto& attr_proto,
                              std::unique_ptr<onnxruntime::Graph>& sub_graph,
                              Graph& graph, Node& node,
                              const logging::Logger& logger) {
  attr_proto.Clear();
  LoadStringFromOrtFormat(*attr_proto.mutable_name(), fbs_attr.name());
  LoadStringFromOrtFormat(*attr_proto.mutable_doc_string(), fbs_attr.doc_string());
  auto type = static_cast<AttributeProto_AttributeType>(fbs_attr.type());
  attr_proto.set_type(type);
  switch (type) {
    case AttributeProto_AttributeType_FLOAT: {
      attr_proto.set_f(fbs_attr.f());
    } break;
    case AttributeProto_AttributeType_INT: {
      attr_proto.set_i(fbs_attr.i());
    } break;
    case AttributeProto_AttributeType_STRING: {
      auto fbs_str = fbs_attr.s();
      ORT_RETURN_IF(nullptr == fbs_str, "Null string attribute. Invalid ORT format model.");
      attr_proto.set_s(fbs_str->str());
    } break;
    case AttributeProto_AttributeType_TENSOR: {
      auto fbs_tensor = fbs_attr.t();
      ORT_RETURN_IF(nullptr == fbs_tensor, "Null tensor attribute. Invalid ORT format model.");
      ORT_RETURN_IF_ERROR(LoadInitializerOrtFormat(*fbs_tensor, *attr_proto.mutable_t()));
    } break;
    case AttributeProto_AttributeType_GRAPH: {
      // If the attribute type is a graph, we will create an empty graph in attr_proto so that the ONNX checker
      // is happy in a full build, and deserialize the ORT Graph instance into the 'graph' param.
      auto fbs_graph = fbs_attr.g();
      ORT_RETURN_IF(nullptr == fbs_graph, "Null graph attribute. Invalid ORT format model.");
      attr_proto.mutable_g()->set_name("Empty graph proto from deserialization of ORT format model");
      ORT_RETURN_IF_ERROR(Graph::LoadFromOrtFormat(*fbs_graph, graph, node, logger, sub_graph));
    } break;
    case AttributeProto_AttributeType_FLOATS: {
      auto fbs_floats = fbs_attr.floats();
      ORT_RETURN_IF(nullptr == fbs_floats, "Null floats attribute. Invalid ORT format model.");
      auto floats = attr_proto.mutable_floats();
      floats->Reserve(fbs_floats->size());
      floats->Add(fbs_floats->cbegin(), fbs_floats->cend());
    } break;
    case AttributeProto_AttributeType_INTS: {
      auto fbs_ints = fbs_attr.ints();
      ORT_RETURN_IF(nullptr == fbs_ints, "Null ints attribute. Invalid ORT format model.");
      auto* ints = attr_proto.mutable_ints();
      ints->Reserve(fbs_ints->size());
      ints->Add(fbs_ints->cbegin(), fbs_ints->cend());
    } break;
    case AttributeProto_AttributeType_STRINGS: {
      auto fbs_strings = fbs_attr.strings();
      ORT_RETURN_IF(nullptr == fbs_strings, "Null strings attribute. Invalid ORT format model.");
      auto* strings = attr_proto.mutable_strings();
      strings->Reserve(fbs_strings->size());
      for (const auto* fbs_str : *fbs_strings) {
        ORT_RETURN_IF(nullptr == fbs_str, "Null string in strings attribute. Invalid ORT format model.");
        strings->Add(fbs_str->str());
      }
    } break;
    case AttributeProto_AttributeType_TENSORS: {
      auto fbs_tensors = fbs_attr.tensors();
      ORT_RETURN_IF(nullptr == fbs_tensors, "Null tensors attribute. Invalid ORT format model.");
      auto* tensors = attr_proto.mutable_tensors();
      tensors->Reserve(fbs_tensors->size());
      for (const auto* fbs_tensor : *fbs_tensors) {
        ORT_RETURN_IF(nullptr == fbs_tensor, "Null tensor in tensors attribute. Invalid ORT format model.");
        ORT_RETURN_IF_ERROR(LoadInitializerOrtFormat(*fbs_tensor, *tensors->Add()));
      }
    } break;

    default:
      break;
  }

  return Status::OK();
}

#endif  // defined(ENABLE_ORT_FORMAT_LOAD)

}  // namespace utils
}  // namespace experimental
}  // namespace onnxruntime
