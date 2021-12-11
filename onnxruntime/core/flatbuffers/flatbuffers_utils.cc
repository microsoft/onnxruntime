// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flatbuffers_utils.h"
#include "schema/ort.fbs.h"

#include "core/common/common.h"
#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "gsl/gsl"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime::fbs::utils {

#if !defined(ORT_MINIMAL_BUILD)

flatbuffers::Offset<flatbuffers::String> SaveStringToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                                               bool has_string, const std::string& src) {
  if (has_string)
    return builder.CreateString(src);

  // If the string does not exist, return 0 (the string does not exist in flatbuffer)
  return 0;
}

static Status SaveTypeInfoOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                    const TypeProto& type_proto,
                                    flatbuffers::Offset<fbs::TypeInfo>& fbs_type_info);

static flatbuffers::Offset<fbs::Dimension> SaveTensorDimensionOrtFormat(
    flatbuffers::FlatBufferBuilder& builder,
    const TensorShapeProto_Dimension& tensor_shape_dim) {
  auto denotation = SaveStringToOrtFormat(builder, tensor_shape_dim.has_denotation(), tensor_shape_dim.denotation());
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
  auto denotation = SaveStringToOrtFormat(builder, type_proto.has_denotation(), type_proto.denotation());
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
  auto doc_string = SaveStringToOrtFormat(builder, value_info_proto.has_doc_string(), value_info_proto.doc_string());
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

#endif  // #if !defined(ORT_MINIMAL_BUILD)

void LoadStringFromOrtFormat(std::string& dst, const flatbuffers::String* fbs_string) {
  if (fbs_string)
    dst = fbs_string->c_str();
}

static Status LoadTypeInfoOrtFormat(const fbs::TypeInfo& fbs_type_info,
                                    TypeProto& type_proto);

static Status LoadTensorDimensionOrtFormat(const fbs::Dimension& fbs_dim,
                                           TensorShapeProto_Dimension& dim) {
  LOAD_STR_FROM_ORT_FORMAT(dim, denotation, fbs_dim.denotation());
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

static Status LoadTensorShapeOrtFormat(const fbs::Shape& fbs_shape, TensorShapeProto& shape_proto) {
  auto fbs_dims = fbs_shape.dim();
  if (fbs_dims) {
    auto dims = shape_proto.mutable_dim();
    dims->Reserve(fbs_dims->size());
    for (const auto fbs_dim : *fbs_dims) {
      ORT_RETURN_IF(nullptr == fbs_dim, "Null entry in dimensions. Invalid ORT format model.");
      TensorShapeProto_Dimension dim;
      ORT_RETURN_IF_ERROR(LoadTensorDimensionOrtFormat(*fbs_dim, *dims->Add()));
    }
  }
  return Status::OK();
}

static Status LoadTensorTypeAndShapeOrtFormat(const fbs::TensorTypeAndShape& fbs_tensor_type,
                                              TypeProto_Tensor& tensor_type_proto) {
  tensor_type_proto.set_elem_type(static_cast<int32_t>(fbs_tensor_type.elem_type()));
  auto fbs_shape = fbs_tensor_type.shape();
  if (fbs_shape) {
    ORT_RETURN_IF_ERROR(LoadTensorShapeOrtFormat(*fbs_shape, *tensor_type_proto.mutable_shape()));
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
  LOAD_STR_FROM_ORT_FORMAT(type_proto, denotation, fbs_type_info.denotation());
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

  LOAD_STR_FROM_ORT_FORMAT(value_info_proto, name, fbs_value_info.name());
  LOAD_STR_FROM_ORT_FORMAT(value_info_proto, doc_string, fbs_value_info.doc_string());

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

Status LoadOpsetImportOrtFormat(const flatbuffers::Vector<flatbuffers::Offset<fbs::OperatorSetId>>* fbs_op_set_ids,
                                std::unordered_map<std::string, int>& domain_to_version) {
  ORT_RETURN_IF(nullptr == fbs_op_set_ids, "Model must have opset imports. Invalid ORT format model.");

  domain_to_version.clear();
  domain_to_version.reserve(fbs_op_set_ids->size());
  for (const auto* fbs_op_set_id : *fbs_op_set_ids) {
    ORT_RETURN_IF(nullptr == fbs_op_set_id, "opset id is null. Invalid ORT format model.");

    const auto* fbs_domain = fbs_op_set_id->domain();
    ORT_RETURN_IF(nullptr == fbs_domain, "opset import domain is null. Invalid ORT format model.");

    std::string domain = fbs_domain->str();

    // perform same aliasing that we do when loading an ONNX format model
    if (domain == kOnnxDomainAlias) {
      domain_to_version[kOnnxDomain] = gsl::narrow_cast<int>(fbs_op_set_id->version());
    } else {
      domain_to_version[domain] = gsl::narrow_cast<int>(fbs_op_set_id->version());
    }
  }
  return Status::OK();
}

bool IsOrtFormatModelBytes(const void* bytes, int num_bytes) {
  return num_bytes > 8 &&  // check buffer is large enough to contain identifier so we don't read random memory
         fbs::InferenceSessionBufferHasIdentifier(bytes);
}

}  // namespace onnxruntime::fbs::utils