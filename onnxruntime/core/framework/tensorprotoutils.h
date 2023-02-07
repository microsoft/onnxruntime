// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <type_traits>

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/common/path.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/framework/endian_utils.h"
#include "core/framework/allocator.h"
#include "core/framework/ort_value.h"
#include "core/framework/mem_buffer.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/env.h"

namespace ONNX_NAMESPACE {
class TensorProto;
class TensorShapeProto;

/** Test if two TensorShapeProto dimensions are equal. */
bool operator==(const TensorShapeProto_Dimension& l, const TensorShapeProto_Dimension& r);
bool operator!=(const TensorShapeProto_Dimension& l, const TensorShapeProto_Dimension& r);

}  // namespace ONNX_NAMESPACE
#endif

namespace onnxruntime {
namespace utils {
#ifndef SHARED_PROVIDER

TensorShape GetTensorShapeFromTensorShapeProto(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto);

TensorShape GetTensorShapeFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto);

/**
 * deserialize a TensorProto into a preallocated memory buffer.
 * \param tensor_proto_path A local file path of where the 'input' was loaded from. Can be NULL if the tensor proto doesn't
 *                        have any external data or it was loaded from current working dir. This path could be either a
 *                        relative path or an absolute path.
 */
common::Status TensorProtoToMLValue(const Env& env, const ORTCHAR_T* tensor_proto_path,
                                    const ONNX_NAMESPACE::TensorProto& input, const MemBuffer& m, OrtValue& value);
/**
 * @brief Deserialize a TensorProto into a preallocated empty Tensor
 * @param env
 * @param model_path
 * @param tensor_proto  source data
 * @param tensorp       destination empty tensor
 * @return
 */
common::Status TensorProtoToTensor(const Env& env, const ORTCHAR_T* model_path,
                                   const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                   Tensor& tensor);

/** Creates a TensorProto from a Tensor.
    @param[in] tensor the Tensor whose data and shape will be used to create the TensorProto.
    @param[in] tensor_proto_name the name of the TensorProto.
    @return the TensorProto.

    Note: Method currently requires that data is in little-endian format.
 */
ONNX_NAMESPACE::TensorProto TensorToTensorProto(const Tensor& tensor, const std::string& tensor_proto_name);

ONNXTensorElementDataType CApiElementTypeFromProtoType(int type);
ONNXTensorElementDataType GetTensorElementType(const ONNX_NAMESPACE::TensorProto& tensor_proto);

// How much memory it will need for putting the content of this tensor into a plain array
// complex64/complex128 tensors are not supported.
// The output value could be zero or -1.
template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);

/**
Special marker used to indicate an existing memory buffer contains the TensorProto external data.
If the 'location' field of the external data info is set to this marker, the 'offset' field should contain the
address of the memory containing the data.
*/
constexpr const ORTCHAR_T* kTensorProtoMemoryAddressTag = ORT_TSTR("*/_ORT_MEM_ADDR_/*");

// Given a tensor proto with external data obtain a pointer to the data and its length.
// The ext_data_deleter argument is updated with a callback that owns/releases the data.
common::Status GetExtDataFromTensorProto(const Env& env, const ORTCHAR_T* model_path,
                                         const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                         void*& ext_data_buf, SafeInt<size_t>& ext_data_len,
                                         OrtCallback& ext_data_deleter);

// Convert the AttributeProto from a Constant node into a TensorProto that can be used as an initializer
// If AttributeProto contains a TensorProto, this tensor proto is converted as is including the case when the
// the data location is external. i.e. it does not load the external data.
// However if AttributeProto contains SparseTensorProto then it converts the data into dense tensor proto
// (including loading external data when applicable).
// model_path is used for contructing full path for external_data
// tensor_name specifies the name for the new TensorProto TensorProto
common::Status ConstantNodeProtoToTensorProto(const ONNX_NAMESPACE::NodeProto& node,
                                              const Path& model_path,
                                              ONNX_NAMESPACE::TensorProto& tensor, const std::string& tensor_name);

common::Status ConstantNodeProtoToTensorProto(const ONNX_NAMESPACE::NodeProto& node,
                                              const Path& model_path,
                                              ONNX_NAMESPACE::TensorProto& tensor);

#if !defined(DISABLE_SPARSE_TENSORS)
// Convert a SparseTensorProto to a dense TensorProto
// If the SparseTensorProto contains external data then it loads the data and converts to dense tensor proto
// The resulting TensorProto will contain the data as raw data.
// model_path is used for contructing full path for external_data
common::Status SparseTensorProtoToDenseTensorProto(const ONNX_NAMESPACE::SparseTensorProto& sparse,
                                                   const Path& model_path,
                                                   ONNX_NAMESPACE::TensorProto& dense);

#if !defined(ORT_MINIMAL_BUILD)
// Convert a TensorProto to a SparseTensorProto
// If the tensorproto contains external data then it loads the data and converts to sparse tensor
// The resulting SparseTensorProto will contain the data as raw data
// model_path is used for contructing full path for external_data
common::Status DenseTensorToSparseTensorProto(const ONNX_NAMESPACE::TensorProto& dense,
                                              const Path& model_path,
                                              ONNX_NAMESPACE::SparseTensorProto& sparse);
#endif  // !ORT_MINIMAL_BUILD
#endif  // !defined(DISABLE_SPARSE_TENSORS)
#endif

inline bool HasDimValue(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim) {
  return dim.value_case() == ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue;
}

inline bool HasDimParam(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim) {
  return dim.value_case() == ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimParam;
}

inline bool HasTensorType(const ONNX_NAMESPACE::TypeProto& type_proto) {
  return type_proto.value_case() == ONNX_NAMESPACE::TypeProto::kTensorType;
}

#if !defined(DISABLE_OPTIONAL_TYPE)
inline bool HasOptionalTensorType(const ONNX_NAMESPACE::TypeProto& type_proto) {
  return type_proto.value_case() == ONNX_NAMESPACE::TypeProto::kOptionalType &&
         type_proto.optional_type().elem_type().value_case() == ONNX_NAMESPACE::TypeProto::kTensorType;
}

inline bool HasOptionalTensorSequenceType(const ONNX_NAMESPACE::TypeProto& type_proto) {
  if (type_proto.value_case() != ONNX_NAMESPACE::TypeProto::kOptionalType) {
    return false;
  }

  const auto& tp = type_proto.optional_type().elem_type();

  if (tp.value_case() != ONNX_NAMESPACE::TypeProto::kSequenceType) {
    return false;
  }

  return tp.sequence_type().elem_type().value_case() == ONNX_NAMESPACE::TypeProto::kTensorType;
}

// Does not check if the TypeProto contains an optional - the caller must validate that
inline const ONNX_NAMESPACE::TypeProto& GetOptionalTypeProto(const ONNX_NAMESPACE::TypeProto& type_proto) {
  return type_proto.optional_type().elem_type();
}

// Does not check if the TypeProto contains an optional - the caller must validate that
inline ONNX_NAMESPACE::TypeProto* GetMutableOptionalTypeProto(ONNX_NAMESPACE::TypeProto& type_proto) {
  return type_proto.mutable_optional_type()->mutable_elem_type();
}

inline bool HasElemType(const ONNX_NAMESPACE::TypeProto_Optional& opt_proto) {
  return opt_proto.elem_type().value_case() != ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET;
}
#endif

inline bool HasElemType(const ONNX_NAMESPACE::TypeProto_Tensor& ten_proto) {
  return ten_proto.elem_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED;
}

inline bool HasShape(const ONNX_NAMESPACE::TypeProto_Tensor& ten_proto) {
  // XXX: Figure out how do in proto3
  return ten_proto.has_shape();
}

#if !defined(DISABLE_SPARSE_TENSORS)
inline bool HasSparseTensorType(const ONNX_NAMESPACE::TypeProto& type_proto) {
  return type_proto.value_case() == ONNX_NAMESPACE::TypeProto::kSparseTensorType;
}

inline bool HasShape(const ONNX_NAMESPACE::TypeProto_SparseTensor& ten_proto) {
  // XXX: Figure out how do in proto3
  return ten_proto.has_shape();
}

inline bool HasElemType(const ONNX_NAMESPACE::TypeProto_SparseTensor& ten_proto) {
  return ten_proto.elem_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED;
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

inline bool HasElementType(const ONNX_NAMESPACE::TypeProto& type_proto) {
  if (HasTensorType(type_proto) && HasElemType(type_proto.tensor_type())) {
    return true;
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (HasSparseTensorType(type_proto) && HasElemType(type_proto.sparse_tensor_type())) {
    return true;
  }
#endif  // !defined(DISABLE_SPARSE_TENSORS)

#if !defined(DISABLE_OPTIONAL_TYPE)
  if (HasOptionalTensorType(type_proto) &&
      HasElemType(GetOptionalTypeProto(type_proto).tensor_type())) {
    return true;
  }
#endif

  return false;
}

// Try to get the element data type.
// The element data type value corresponds to TensorProto_DataType. It is applicable to types with shapes.
inline bool TryGetElementDataType(const ONNX_NAMESPACE::TypeProto& type_proto, int32_t& element_data_type) {
  if (HasTensorType(type_proto) && HasElemType(type_proto.tensor_type())) {
    element_data_type = type_proto.tensor_type().elem_type();
    return true;
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (HasSparseTensorType(type_proto) && HasElemType(type_proto.sparse_tensor_type())) {
    element_data_type = type_proto.sparse_tensor_type().elem_type();
    return true;
  }
#endif  // !defined(DISABLE_SPARSE_TENSORS)

#if !defined(DISABLE_OPTIONAL_TYPE)
  if (HasOptionalTensorType(type_proto) &&
      HasElemType(GetOptionalTypeProto(type_proto).tensor_type())) {
    element_data_type = GetOptionalTypeProto(type_proto).tensor_type().elem_type();
    return true;
  }
#endif

  element_data_type = ONNX_NAMESPACE::TensorProto::UNDEFINED;
  return false;
}

inline bool HasShape(const ONNX_NAMESPACE::TypeProto& type_proto) {
  if (HasTensorType(type_proto) && HasShape(type_proto.tensor_type())) {
    return true;
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (HasSparseTensorType(type_proto) && HasShape(type_proto.sparse_tensor_type())) {
    return true;
  }
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
  if (HasOptionalTensorType(type_proto) && HasShape(GetOptionalTypeProto(type_proto).tensor_type())) {
    return true;
  }
#endif

  return false;
}

inline const ONNX_NAMESPACE::TensorShapeProto* TryGetShape(const ONNX_NAMESPACE::TypeProto& type_proto) {
  if (HasTensorType(type_proto) && HasShape(type_proto.tensor_type())) {
    return &type_proto.tensor_type().shape();
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (HasSparseTensorType(type_proto) && HasShape(type_proto.sparse_tensor_type())) {
    return &type_proto.sparse_tensor_type().shape();
  }
#endif

#if !defined(DISABLE_OPTIONAL_TYPE)
  if (HasOptionalTensorType(type_proto) && HasShape(GetOptionalTypeProto(type_proto).tensor_type())) {
    return &GetOptionalTypeProto(type_proto).tensor_type().shape();
  }
#endif

  return nullptr;
}

inline const ONNX_NAMESPACE::TensorShapeProto& GetShape(const ONNX_NAMESPACE::TypeProto& type_proto) {
  const auto* shape = TryGetShape(type_proto);
  ORT_ENFORCE(shape != nullptr, "TypeProto must have shape for this to run");
  return *shape;
}

inline bool HasRawData(const ONNX_NAMESPACE::TensorProto& ten_proto) {
  // Can not be UNDEFINED and can not be STRING but test for STRING is usually performed separately
  // to return an error
  return ten_proto.data_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED &&
         ten_proto.has_raw_data();  // XXX: Figure out how to do in proto3
}

inline bool HasExternalData(const ONNX_NAMESPACE::TensorProto& ten_proto) {
  // Can not be UNDEFINED and can not be STRING but test for STRING is usually performed separately
  // to return an error
  return ten_proto.data_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED &&
         ten_proto.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL;
}

inline bool HasDataType(const ONNX_NAMESPACE::TensorProto& ten_proto) {
  return ten_proto.data_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED;
}

inline bool HasString(const ONNX_NAMESPACE::TensorProto& ten_proto) {
  return ten_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_STRING;
}

#ifndef SHARED_PROVIDER
inline bool HasName(const ONNX_NAMESPACE::TensorProto& ten_proto) {
  return ten_proto.has_name();  // XXX
}

inline bool HasElemType(const ONNX_NAMESPACE::TypeProto_Sequence& seq_proto) {
  return seq_proto.elem_type().value_case() != ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET;
}

inline bool HasName(const ONNX_NAMESPACE::SparseTensorProto& ten_proto) {
  return ten_proto.values().has_name();  // XXX
}

inline bool HasKeyType(const ONNX_NAMESPACE::TypeProto_Map& map_proto) {
  return map_proto.key_type() != ONNX_NAMESPACE::TensorProto::UNDEFINED;
}

inline bool HasValueType(const ONNX_NAMESPACE::TypeProto_Map& map_proto) {
  return map_proto.value_type().value_case() != ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET;
}
#endif

inline bool HasType(const ONNX_NAMESPACE::ValueInfoProto& vi_proto) {
  return vi_proto.type().value_case() != ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET;
}

#ifndef SHARED_PROVIDER
inline bool HasName(const ONNX_NAMESPACE::ValueInfoProto& vi_proto) {
  return vi_proto.has_name();  // XXX: Figure out proto3 way
}

inline bool HasDomain(const ONNX_NAMESPACE::TypeProto_Opaque& op_proto) {
  return !op_proto.domain().empty();
}

inline bool HasName(const ONNX_NAMESPACE::TypeProto_Opaque& op_proto) {
  return !op_proto.name().empty();
}

#endif

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

#ifndef SHARED_PROVIDER
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
  // XXX: Figure out proto3 style
  return node_proto.has_name();
}
#endif

// UnpackTensor from raw data or the type specific data field. Does not handle external data.
// If the tensor does not contain raw data then raw_data should be nullptr and raw_data_len should be 0.
template <typename T>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ T* p_data, size_t expected_size);

// UnpackTensor from raw data, external data or the type specific data field.
// Uses the model path to construct the full path for loading external data. In case when model_path is empty
// it uses current directory.
template <typename T>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const Path& model_path,
                    /*out*/ T* p_data, size_t expected_size);

/**
 * Unpack the data from an initializer tensor
 * Please note, this function does not unpack string_data of an initializer tensor
 * @param initializer       given initializer tensor
 * @param model_path        model_path to construct external data dir path. When this is empty, current dir is used.
 * @param unpacked_tensor   the vector holds data from the initializer in byte form
 * @returns                 Status::OK() if data is unpacked successfully
 */
common::Status UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& initializer,
                                     const Path& model_path,
                                     std::vector<uint8_t>& unpacked_tensor);

/**
 * Unpack the data from an internal initializer tensor, will return error when the given initializer
 * contains external data
 * Please note, this function does not unpack string_data of an initializer tensor
 * @param initializer       given initializer tensor
 * @param unpacked_tensor   the vector holds data from the initializer in byte form
 * @returns                 Status::OK() if data is unpacked successfully
 */
common::Status UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& initializer,
                                     std::vector<uint8_t>& unpacked_tensor);
}  // namespace utils
}  // namespace onnxruntime
