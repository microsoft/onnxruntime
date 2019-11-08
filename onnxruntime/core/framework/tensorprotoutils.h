// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <type_traits>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/ml_value.h"
#include "core/framework/mem_buffer.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/env.h"

namespace ONNX_NAMESPACE {
class TensorProto;
class TensorShapeProto;
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
class Tensor;
namespace utils {
TensorShape GetTensorShapeFromTensorShapeProto(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto);
/**
 * deserialize a TensorProto into a preallocated memory buffer.
 * \param tensor_proto_path A local file path of where the 'input' was loaded from. Can be NULL if the tensor proto doesn't
 *                        have any external data or it was loaded from current working dir. This path could be either a
 *                        relative path or an absolute path.
 */
common::Status TensorProtoToMLValue(const Env& env, const ORTCHAR_T* tensor_proto_path,
                                    const ONNX_NAMESPACE::TensorProto& input, const MemBuffer& m, OrtValue& value,
                                    OrtCallback& deleter);

/** Creates a TensorProto from a Tensor.
    @param[in] tensor the Tensor whose data and shape will be used to create the TensorProto.
    @param[in] tensor_proto_name the name of the TensorProto.
    @param[in] tensor_proto_type the type of the TensorProto.
    @return the TensorProto. 
    
    Note: Method currently requires that data is in little-endian format.
    TODO Once the GetTensorProtoType supports all data types, we can remove the tensor_proto_type parameter and 
    instead get the type from the tensor. */
ONNX_NAMESPACE::TensorProto TensorToTensorProto(const Tensor& tensor, const std::string& tensor_proto_name,
                                                const ONNX_NAMESPACE::TypeProto& tensor_proto_type);

ONNXTensorElementDataType CApiElementTypeFromProtoType(int type);
ONNXTensorElementDataType GetTensorElementType(const ONNX_NAMESPACE::TensorProto& tensor_proto);

// How much memory it will need for putting the content of this tensor into a plain array
// complex64/complex128 tensors are not supported.
// The output value could be zero or -1.
template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);

template <typename T>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ T* p_data, int64_t expected_size);

inline bool HasDimValue(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim) {
  return dim.value_case() == ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue;
}

inline bool HasDimParam(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim) {
  return dim.value_case() == ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimParam;
}

inline bool HasTensorType(const ONNX_NAMESPACE::TypeProto& type_proto) {
  return type_proto.value_case() == ONNX_NAMESPACE::TypeProto::kTensorType;
}

inline bool HasElemType(const ONNX_NAMESPACE::TypeProto_Tensor& ten_proto) {
  return ten_proto.elem_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED;
  ;
}

inline bool HasShape(const ONNX_NAMESPACE::TypeProto_Tensor& ten_proto) {
  // XXX: Figure out how do in proto3
  return ten_proto.has_shape();
}

inline bool HasShape(const ONNX_NAMESPACE::TypeProto_SparseTensor& ten_proto) {
  // XXX: Figure out how do in proto3
  return ten_proto.has_shape();
}

inline bool HasRawData(const ONNX_NAMESPACE::TensorProto& ten_proto) {
  // Can not be UNDEFINED and can not be STRING but test for STRING is usually performed separately
  // to return an error
  return ten_proto.data_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED &&
         ten_proto.has_raw_data();  // XXX: Figure out how to do in proto3
}

inline bool HasDataType(const ONNX_NAMESPACE::TensorProto& ten_proto) {
  return ten_proto.data_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED;
}

inline bool HasName(const ONNX_NAMESPACE::TensorProto& ten_proto) {
  return ten_proto.has_name();  // XXX
}

inline bool HasElemType(const ONNX_NAMESPACE::TypeProto_Sequence& seq_proto) {
  return seq_proto.elem_type().value_case() != ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET;
}

inline bool HasElemType(const ONNX_NAMESPACE::TypeProto_SparseTensor& ten_proto) {
  return ten_proto.elem_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED;
}

inline bool HasKeyType(const ONNX_NAMESPACE::TypeProto_Map& map_proto) {
  return map_proto.key_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED;
}

inline bool HasValueType(const ONNX_NAMESPACE::TypeProto_Map& map_proto) {
  return map_proto.value_type().value_case() != ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET;
}

inline bool HasType(const ONNX_NAMESPACE::ValueInfoProto& vi_proto) {
  return vi_proto.type().value_case() != ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET;
}

inline bool HasName(const ONNX_NAMESPACE::ValueInfoProto& vi_proto) {
  return vi_proto.has_name();  // XXX: Figure out proto3 way
}

inline bool HasDomain(const ONNX_NAMESPACE::TypeProto_Opaque& op_proto) {
  return !op_proto.domain().empty();
}

inline bool HasName(const ONNX_NAMESPACE::TypeProto_Opaque& op_proto) {
  return !op_proto.name().empty();
}

inline bool HasType(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_UNDEFINED;
}

inline bool HasFloat(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT;
}

inline bool HasFloats(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOATS;
}

inline bool HasInt(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INT;
}

inline bool HasInts(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INTS;
}

inline bool HasString(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_STRING;
}

inline bool HasStrings(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_STRINGS;
}

inline bool HasTensor(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_TENSOR;
}

inline bool HasTensors(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_TENSORS;
}

inline bool HasGraph(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_GRAPH;
}

inline bool HasGraphs(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.type() == ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_GRAPHS;
}

inline bool HasName(const ONNX_NAMESPACE::AttributeProto& at_proto) {
  return at_proto.has_name();  // XXX: Fugure out proto3
}

inline bool HasGraph(const ONNX_NAMESPACE::ModelProto& m_proto) {
  return m_proto.has_graph();  // XXX proto3
}

inline bool HasIrVersion(const ONNX_NAMESPACE::ModelProto& m_proto) {
  return m_proto.has_ir_version();  // XXX proto3
}

inline bool HasModelVersion(const ONNX_NAMESPACE::ModelProto& m_proto) {
  return m_proto.has_model_version();  // XXX proto3
}

inline bool HasName(const ONNX_NAMESPACE::NodeProto& node_proto) {
  //XXX: Figure out proto3 style
  return node_proto.has_name();
}

}  // namespace utils
}  // namespace onnxruntime
