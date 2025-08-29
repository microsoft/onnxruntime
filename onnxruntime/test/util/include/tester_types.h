// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/tensor.h"
#include "core/framework/TensorSeq.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct TTypeProto {
  explicit TTypeProto(const std::vector<int64_t>* shape = nullptr) {
    proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
    if (shape) {
      auto mutable_shape = proto.mutable_tensor_type()->mutable_shape();
      for (auto i : *shape) {
        auto* mutable_dim = mutable_shape->add_dim();
        if (i != -1)
          mutable_dim->set_dim_value(i);
        else
          mutable_dim->set_dim_param("symbolic");
      }
    }
  }
  ONNX_NAMESPACE::TypeProto proto;
};

// Variable template for ONNX_NAMESPACE::TensorProto_DataTypes, s_type_proto<float>, etc..
template <typename T>
struct TTensorType {
  static const TTypeProto<T> s_type_proto;
};

template <typename T>
const TTypeProto<T> TTensorType<T>::s_type_proto;

#if !defined(DISABLE_SPARSE_TENSORS)
struct TSparseTensorProto {
  explicit TSparseTensorProto(int32_t dtype, const std::vector<int64_t>* shape = nullptr) {
    proto.mutable_sparse_tensor_type()->set_elem_type(dtype);
    if (shape) {
      auto m_shape = proto.mutable_sparse_tensor_type()->mutable_shape();
      for (int64_t v : *shape) {
        auto* m_dim = m_shape->add_dim();
        if (v != -1)
          m_dim->set_dim_value(v);
        else
          m_dim->set_dim_param("symbolic");
      }
    }
  }
  ONNX_NAMESPACE::TypeProto proto;
};
#endif

// TypeProto for map<TKey, TVal>
template <typename TKey, typename TVal>
struct MTypeProto {
  MTypeProto() {
    proto.mutable_map_type()->set_key_type(utils::ToTensorProtoElementType<TKey>());
    proto.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(
        utils::ToTensorProtoElementType<TVal>());
    proto.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  }
  ONNX_NAMESPACE::TypeProto proto;
};

template <typename TKey, typename TVal>
struct MMapType {
  static const MTypeProto<TKey, TVal> s_map_type_proto;
};

template <typename TKey, typename TVal>
const MTypeProto<TKey, TVal> MMapType<TKey, TVal>::s_map_type_proto;

// TypeProto for vector<map<TKey, TVal>>
template <typename TKey, typename TVal>
struct VectorOfMapTypeProto {
  VectorOfMapTypeProto() {
    auto* map_type = proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type();
    map_type->set_key_type(utils::ToTensorProtoElementType<TKey>());
    map_type->mutable_value_type()->mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<TVal>());
    map_type->mutable_value_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
  }
  ONNX_NAMESPACE::TypeProto proto;
};

template <typename TKey, typename TVal>
struct VectorOfMapType {
  static const VectorOfMapTypeProto<TKey, TVal> s_vec_map_type_proto;
};

template <typename TKey, typename TVal>
const VectorOfMapTypeProto<TKey, TVal> VectorOfMapType<TKey, TVal>::s_vec_map_type_proto;

template <typename ElemType>
struct SequenceTensorTypeProto {
  SequenceTensorTypeProto() {
    MLDataType dt = DataTypeImpl::GetTensorType<ElemType>();
    const auto* elem_proto = dt->GetTypeProto();
    proto.mutable_sequence_type()->mutable_elem_type()->CopyFrom(*elem_proto);
  }
  ONNX_NAMESPACE::TypeProto proto;
};

template <typename ElemType>
struct SequenceTensorType {
  static const SequenceTensorTypeProto<ElemType> s_sequence_tensor_type_proto;
};

template <typename ElemType>
const SequenceTensorTypeProto<ElemType> SequenceTensorType<ElemType>::s_sequence_tensor_type_proto;

#if !defined(DISABLE_OPTIONAL_TYPE)

struct OptionalTypeProto {
  OptionalTypeProto(const ONNX_NAMESPACE::TypeProto& type_proto) {
    proto.mutable_optional_type()->mutable_elem_type()->CopyFrom(type_proto);
  }
  ONNX_NAMESPACE::TypeProto proto;
};

#endif

template <typename T>
struct SeqTensors {
  void AddTensor(const std::vector<int64_t>& shape0, const std::vector<T>& data0) {
    tensors.push_back(Tensor<T>{shape0, data0});
  }

  template <typename U>
  struct Tensor {
    std::vector<int64_t> shape;
    std::vector<U> data;
  };
  std::vector<Tensor<T>> tensors;
};

}  // namespace test
}  // namespace onnxruntime
