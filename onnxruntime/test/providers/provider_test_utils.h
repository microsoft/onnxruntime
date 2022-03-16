// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/logging/logging.h"
#include "core/common/optional.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/customregistry.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/run_options.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/framework/data_types.h"
#include "test/test_environment.h"
#include "test/framework/TestAllocatorManager.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/session_options.h"
#include "core/providers/providers.h"
#include "test/util/include/asserts.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <gsl/gsl>
#include "core/util/math_cpuonly.h"
#include <variant>

namespace onnxruntime {
class InferenceSession;
struct SessionOptions;

namespace test {
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

// Function templates to translate C++ types into ONNX_NAMESPACE::TensorProto_DataTypes
template <typename T>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType();

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<float>() {
  return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<double>() {
  return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int32_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_INT32;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int64_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_INT64;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<bool>() {
  return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int8_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_INT8;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int16_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_INT16;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint8_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_UINT8;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint16_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_UINT16;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint32_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_UINT32;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint64_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_UINT64;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<std::string>() {
  return ONNX_NAMESPACE::TensorProto_DataType_STRING;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<MLFloat16>() {
  return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
}

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<BFloat16>() {
  return ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
}

template <typename T>
struct TTypeProto {
  explicit TTypeProto(const std::vector<int64_t>* shape = nullptr) {
    proto.mutable_tensor_type()->set_elem_type(TypeToDataType<T>());
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
    proto.mutable_map_type()->set_key_type(TypeToDataType<TKey>());
    proto.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TypeToDataType<TVal>());
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
    map_type->set_key_type(TypeToDataType<TKey>());
    map_type->mutable_value_type()->mutable_tensor_type()->set_elem_type(TypeToDataType<TVal>());
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

struct CheckParams {
  bool sort_output_ = false;
  optional<float> absolute_error_;
  optional<float> relative_error_;
};

// To use OpTester:
//  1. Create one with the op name
//  2. Call AddAttribute with any attributes
//  3. Call AddInput for all the inputs
//  4. Call AddOutput with all expected outputs,
//     Or call AddReferenceOutputs to compute reference outputs with the model
//  5. Call Run
// Not all tensor types and output types are added, if a new input type is used, add it to the TypeToDataType list
// above for new output types, add a new specialization for Check<> See current usage for an example, should be self
// explanatory
class OpTester {
 public:
  // Default to the first opset that ORT was available (7).
  // When operators are updated they need to explicitly add tests for the new opset version.
  // This is due to the kernel matching logic. See KernelRegistry::VerifyKernelDef.
  // Additionally, -1 is supported and defaults to the latest known opset.
  //
  // Defaulting to the latest opset version would result in existing operator implementations for non-CPU EPs to
  // lose their test coverage until an implementation for the new version is added.
  //   e.g. there are CPU and GPU implementations for version 1 of an op. both are tested by a single OpTester test.
  //        opset changes from 1 to 2 and CPU implementation gets added. If 'opset_version' is 2 the kernel matching
  //        will find and run the CPU v2 implementation, but will not match the GPU v1 implementation.
  //        OpTester will say it was successful as at least one EP ran, and the GPU implementation of v1 no longer has
  //        test coverage.
  explicit OpTester(const char* op, int opset_version = 7, const char* domain = onnxruntime::kOnnxDomain, bool verify_output = true)
      : op_(op), domain_(domain), opset_version_(opset_version), verify_output_(verify_output) {
    if (opset_version_ < 0) {
      static int latest_onnx_version =
          ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange().Map().at(ONNX_NAMESPACE::ONNX_DOMAIN).second;
      opset_version_ = latest_onnx_version;
    }
  }

  virtual ~OpTester();

  // Set whether the NodeArg created by AddInput/AddOutput should include shape information
  // for Tensor types. If not added, shape inferencing should resolve. If added, shape inferencing
  // should validate. Default is to not add.
  // Additionally a symbolic dimension will be added if symbolic_dim matches a dimension in the input.
  OpTester& AddShapeToTensorData(bool add_shape = true, int symbolic_dim = -1) {
    add_shape_to_tensor_data_ = add_shape;
    add_symbolic_dim_to_tensor_data_ = symbolic_dim;
    return *this;
  }

  // We have an initializer_list and vector version of the Add functions because std::vector is specialized for
  // bool and we can't get the raw data out. So those cases must use an initializer_list

  // Dims variant is needed to reduce the number of overloads
  // MS compiler refuses to create a gsl::span from initializer_list especially
  // if it contains a single element
  using DimsVariant = std::variant<std::vector<int64_t>, TensorShapeVector>;

  template <typename T>
  void AddInput(const char* name, std::initializer_list<int64_t> dims, std::initializer_list<T> values,
                bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(input_data_, name, dims_var, values.begin(), values.size(), is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, std::initializer_list<int64_t> dims, const std::vector<T>& values,
                bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(input_data_, name, dims_var, values.data(), values.size(), is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, std::initializer_list<int64_t> dims, const T* p_values,
                const size_t size, bool is_initializer = false,
                const std::vector<std::string>* dim_params = nullptr) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(input_data_, name, dims_var, p_values, size, is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, std::initializer_list<int64_t> dims, const TensorShapeVector& values,
                bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(input_data_, name, dims_var, values.data(), values.size(), is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, const DimsVariant& dims, std::initializer_list<T> values,
                bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    AddData(input_data_, name, dims, values.begin(), values.size(), is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, const DimsVariant& dims, const std::vector<T>& values,
                bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    AddData(input_data_, name, dims, values.data(), values.size(), is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, const DimsVariant& dims, const T* p_values,
                const size_t size, bool is_initializer = false,
                const std::vector<std::string>* dim_params = nullptr) {
    AddData(input_data_, name, dims, p_values, size, is_initializer, false, dim_params);
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  // Useful to add boolean data
  template <typename T>
  void AddSparseCooInput(const char* name, const std::vector<int64_t>& dims,
                         const std::initializer_list<T>& values, const std::vector<int64_t>& indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(input_data_, ml_type, name, dims,
                           gsl::make_span(values).as_bytes(),
                           gsl::make_span(indices),
                           CheckParams(), dim_params);
  }

  template <typename T>
  void AddSparseCooInput(const char* name, const std::vector<int64_t>& dims,
                         const std::vector<T>& values, const std::vector<int64_t>& indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(input_data_, ml_type, name, dims,
                           gsl::make_span(values).as_bytes(),
                           gsl::make_span(indices),
                           CheckParams(), dim_params);
  }

  template <typename T>
  void AddSparseCooInput(const char* name, const std::vector<int64_t>& dims,
                         gsl::span<const T> values_span,
                         const std::vector<int64_t>& indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(input_data_, ml_type, name, dims,
                           values_span.as_bytes(),
                           gsl::make_span(indices),
                           CheckParams(), dim_params);
  }

  void AddSparseCooInput(const char* name, const std::vector<int64_t>& dims,
                         const std::vector<std::string>& values,
                         const std::vector<int64_t>& indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    AddSparseCooTensorStrings(input_data_, name, dims,
                              gsl::make_span(values),
                              gsl::make_span(indices),
                              dim_params);
  }

  // Useful to add boolean data
  template <typename T>
  void AddSparseCsrInput(const char* name, const std::vector<int64_t>& dims,
                         const std::initializer_list<T>& values,
                         const std::vector<int64_t>& inner_indices,
                         const std::vector<int64_t>& outer_indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(input_data_, ml_type, name, dims,
                           gsl::make_span(values).as_bytes(),
                           gsl::make_span(inner_indices),
                           gsl::make_span(outer_indices),
                           CheckParams(), dim_params);
  }

  template <typename T>
  void AddSparseCsrInput(const char* name, const std::vector<int64_t>& dims,
                         const std::vector<T>& values,
                         const std::vector<int64_t>& inner_indices,
                         const std::vector<int64_t>& outer_indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(input_data_, ml_type, name, dims,
                           gsl::make_span(values).as_bytes(),
                           gsl::make_span(inner_indices),
                           gsl::make_span(outer_indices),
                           CheckParams(), dim_params);
  }

  template <typename T>
  void AddSparseCsrInput(const char* name, const std::vector<int64_t>& dims,
                         gsl::span<const T> values_span,
                         const std::vector<int64_t>& inner_indices,
                         const std::vector<int64_t>& outer_indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(input_data_, ml_type, name, dims,
                           values_span.as_bytes(),
                           gsl::make_span(inner_indices),
                           gsl::make_span(outer_indices),
                           CheckParams(), dim_params);
  }

  void AddSparseCsrInput(const char* name, const std::vector<int64_t>& dims,
                         const std::vector<std::string>& values,
                         const std::vector<int64_t>& inner_indices,
                         const std::vector<int64_t>& outer_indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    AddSparseCsrTensorStrings(input_data_, name, dims,
                              gsl::make_span(values),
                              gsl::make_span(inner_indices),
                              gsl::make_span(outer_indices),
                              dim_params);
  }
#endif

  // Add other registered types, possibly experimental
  template <typename T>
  void AddInput(const char* name, const T& val) {
    auto mltype = DataTypeImpl::GetType<T>();
    ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
    auto ptr = std::make_unique<T>(val);
    OrtValue value;
    value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
    ptr.release();
    input_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value), optional<float>(),
                               optional<float>()));
  }

  template <typename T>
  void AddInput(const char* name, T&& val) {
    auto mltype = DataTypeImpl::GetType<T>();
    ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
    auto ptr = std::make_unique<T>(std::move(val));
    OrtValue value;
    value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
    ptr.release();
    input_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value), optional<float>(),
                               optional<float>()));
  }

  template <typename T>
  void AddSeqInput(const char* name, const SeqTensors<T>& seq_tensors) {
    AddSeqData<T>(input_data_, name, &seq_tensors);
  }

  template <typename T>
  void AddSeqOutput(const char* name, const SeqTensors<T>& seq_tensors) {
    AddSeqData<T>(output_data_, name, &seq_tensors);
  }

#if !defined(DISABLE_OPTIONAL_TYPE)

  template <typename T>
  void AddOptionalTypeTensorInput(const char* name, const DimsVariant& dims,
                                  const std::initializer_list<T>* values = nullptr,
                                  bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    AddData(input_data_, name, dims, values ? values->begin() : nullptr,
            values ? values->size() : 0, is_initializer, false, dim_params, 0.0f, 0.0f, true);
  }

  template <typename T>
  void AddOptionalTypeTensorInput(const char* name, std::initializer_list<int64_t> dims,
                                  const std::initializer_list<T>* values = nullptr,
                                  bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(input_data_, name, dims_var, values ? values->begin() : nullptr,
            values ? values->size() : 0, is_initializer, false, dim_params, 0.0f, 0.0f, true);
  }

  template <typename T>
  void AddOptionalTypeTensorOutput(const char* name, const DimsVariant& dims,
                                   const std::initializer_list<T>* expected_values = nullptr,
                                   bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    AddData(output_data_, name, dims, expected_values ? expected_values->begin() : nullptr,
            expected_values ? expected_values->size() : 0, false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error, true);
  }

  template <typename T>
  void AddOptionalTypeTensorOutput(const char* name, std::initializer_list<int64_t> dims,
                                   const std::initializer_list<T>* expected_values = nullptr,
                                   bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(output_data_, name, dims_var, expected_values ? expected_values->begin() : nullptr,
            expected_values ? expected_values->size() : 0, false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error, true);
  }

  template <typename T>
  void AddOptionalTypeSeqInput(const char* name,
                               const SeqTensors<T>* seq_tensors) {
    AddSeqData<T>(input_data_, name, seq_tensors, true);
  }

  template <typename T>
  void AddOptionalTypeSeqOutput(const char* name,
                                const SeqTensors<T>* seq_tensors) {
    AddSeqData<T>(output_data_, name, seq_tensors, true);
  }

#endif

  template <typename TKey, typename TVal>
  void AddInput(const char* name, const std::map<TKey, TVal>& val) {
    std::unique_ptr<std::map<TKey, TVal>> ptr = std::make_unique<std::map<TKey, TVal>>(val);
    OrtValue value;
    value.Init(ptr.release(), DataTypeImpl::GetType<std::map<TKey, TVal>>(),
               DataTypeImpl::GetType<std::map<TKey, TVal>>()->GetDeleteFunc());
    input_data_.push_back(Data(NodeArg(name, &MMapType<TKey, TVal>::s_map_type_proto.proto), std::move(value),
                               optional<float>(), optional<float>()));
  }

  /*
   * Use this API to add an input *edge* to the node/op being tested that won't
   * have any data passed into.
   * Such an edge will have the qualifier OpSchema::Optional in the schema.
   * This is exposed to ensure the op kernel implementations can be tested to handle
   * presence/absence of such optional input edges.
   */
  template <typename T>
  void AddOptionalInputEdge() {
    std::string name;  // empty == input doesn't exist
    input_data_.push_back(Data(NodeArg(name, &TTensorType<T>::s_type_proto.proto), OrtValue(), optional<float>(),
                               optional<float>()));
  }

  template <typename T>
  void AddOutput(const char* name, std::initializer_list<int64_t> dims, std::initializer_list<T> expected_values,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(output_data_, name, dims_var, expected_values.begin(), expected_values.size(), false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  template <typename T>
  void AddOutput(const char* name, std::initializer_list<int64_t> dims, const std::vector<T>& expected_values,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(output_data_, name, dims_var, expected_values.data(), expected_values.size(), false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  template <typename T>
  void AddOutput(const char* name, std::initializer_list<int64_t> dims, const T* p_values, const size_t size,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(output_data_, name, dims, p_values, size, false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  template <typename T>
  void AddOutput(const char* name, const DimsVariant& dims, std::initializer_list<T> expected_values,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    AddData(output_data_, name, dims, expected_values.begin(), expected_values.size(), false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  // This function doesn't work for vector<bool> because const vector<bool> cannot invoke its data().
  template <typename T>
  void AddOutput(const char* name, const DimsVariant& dims, const std::vector<T>& expected_values,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    AddData(output_data_, name, dims, expected_values.data(), expected_values.size(), false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  template <typename T>
  void AddOutput(const char* name, const DimsVariant& dims, const TensorShapeVector& expected_values,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    AddData(output_data_, name, dims, expected_values.data(), expected_values.size(), false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  template <typename T>
  void AddOutput(const char* name, const DimsVariant& dims, const T* p_values, const size_t size,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    AddData(output_data_, name, dims, p_values, size, false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  template <typename T>
  void AddSparseCooOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::initializer_list<T>& expected_values,
                          const std::vector<int64_t>& expected_indices,
                          const CheckParams& check_params = CheckParams()) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(output_data_, ml_type, name, dims,
                           gsl::make_span(expected_values).as_bytes(),
                           gsl::make_span(expected_indices),
                           check_params, nullptr /*dim_params*/);
  }

  template <typename T>
  void AddSparseCooOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::vector<T>& expected_values,
                          const std::vector<int64_t>& expected_indices,
                          const CheckParams& check_params = CheckParams()) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(output_data_, ml_type, name, dims,
                           gsl::make_span(expected_values).as_bytes(),
                           gsl::make_span(expected_indices),
                           check_params, nullptr /*dim_params*/);
  }

  template <typename T>
  void AddSparseCooOutput(const char* name, const std::vector<int64_t>& dims,
                          gsl::span<const T> expected_values_span,
                          const std::vector<int64_t>& expected_indices,
                          const CheckParams& check_params = CheckParams()) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(output_data_, ml_type, name, dims,
                           expected_values_span.as_bytes(),
                           gsl::make_span(expected_indices),
                           check_params, nullptr /*dim_params*/);
  }

  void AddSparseCooOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::vector<std::string>& expected_values,
                          const std::vector<int64_t>& expected_indices) {
    AddSparseCooTensorStrings(output_data_, name, dims,
                              gsl::make_span(expected_values),
                              gsl::make_span(expected_indices));
  }

  template <typename T>
  void AddSparseCsrOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::initializer_list<T>& values,
                          const std::vector<int64_t>& inner_indices,
                          const std::vector<int64_t>& outer_indices,
                          const CheckParams& check_params = CheckParams()) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(output_data_, ml_type, name, dims,
                           gsl::make_span(values).as_bytes(),
                           gsl::make_span(inner_indices),
                           gsl::make_span(outer_indices),
                           check_params, nullptr /*dim_params*/);
  }

  template <typename T>
  void AddSparseCsrOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::vector<T>& values,
                          const std::vector<int64_t>& inner_indices,
                          const std::vector<int64_t>& outer_indices,
                          const CheckParams& check_params = CheckParams()) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(output_data_, ml_type, name, dims,
                           gsl::make_span(values).as_bytes(),
                           gsl::make_span(inner_indices),
                           gsl::make_span(outer_indices),
                           check_params, nullptr /*dim_params*/);
  }

  template <typename T>
  void AddSparseCsrOutput(const char* name, const std::vector<int64_t>& dims,
                          gsl::span<const T> expected_values_span,
                          const std::vector<int64_t>& expected_inner_indices,
                          const std::vector<int64_t>& expected_outer_indices,
                          const CheckParams& check_params = CheckParams()) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(output_data_, ml_type, name, dims,
                           expected_values_span.as_bytes(),
                           gsl::make_span(expected_inner_indices),
                           gsl::make_span(expected_outer_indices),
                           check_params, nullptr /*dim_params*/);
  }

  void AddSparseCsrOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::vector<std::string>& expected_values,
                          const std::vector<int64_t>& expected_inner_indices,
                          const std::vector<int64_t>& expected_outer_indices) {
    AddSparseCsrTensorStrings(output_data_, name, dims,
                              gsl::make_span(expected_values),
                              gsl::make_span(expected_inner_indices),
                              gsl::make_span(expected_outer_indices));
  }
#endif

  /*
   * Use this API to add an output *edge* to the node/op being tested that shouldn't have any
   * data produced into.
   * Such an edge will have the qualifier OpSchema::Optional in the schema.
   * This is exposed to ensure the op kernel implementations can be tested to handle
   * presence/absence of such optional output edges.
   */
  template <typename T>
  void AddOptionalOutputEdge() {
    std::string name;  // empty == output doesn't exist
    output_data_.push_back(Data(NodeArg(name, &TTensorType<T>::s_type_proto.proto), OrtValue(), optional<float>(),
                                optional<float>()));
  }

  // Add other registered types, possibly experimental
  template <typename T>
  void AddOutput(const char* name, const T& val) {
    auto mltype = DataTypeImpl::GetType<T>();
    ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
    auto ptr = std::make_unique<T>(val);
    OrtValue value;
    value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
    ptr.release();
    output_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value), optional<float>(),
                                optional<float>()));
  }

  template <typename T>
  void AddOutput(const char* name, T&& val) {
    auto mltype = DataTypeImpl::GetType<T>();
    ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
    auto ptr = std::make_unique<T>(std::move(val));
    OrtValue value;
    value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
    ptr.release();
    output_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value), optional<float>(),
                                optional<float>()));
  }

  // Add non tensor output
  template <typename TKey, typename TVal>
  void AddOutput(const char* name, const std::vector<std::map<TKey, TVal>>& val) {
    auto ptr = std::make_unique<std::vector<std::map<TKey, TVal>>>(val);
    OrtValue ml_value;
    ml_value.Init(ptr.release(), DataTypeImpl::GetType<std::vector<std::map<TKey, TVal>>>(),
                  DataTypeImpl::GetType<std::vector<std::map<TKey, TVal>>>()->GetDeleteFunc());
    output_data_.push_back(Data(NodeArg(name, &VectorOfMapType<TKey, TVal>::s_vec_map_type_proto.proto), std::move(ml_value),
                                optional<float>(), optional<float>()));
  }

  // Generate the reference outputs with the model file
  void AddReferenceOutputs(const std::string& model_path);

  void AddCustomOpRegistry(std::shared_ptr<CustomRegistry> registry) {
    custom_schema_registries_.push_back(registry->GetOpschemaRegistry());
    custom_session_registries_.push_back(registry);
  }

  void SetOutputAbsErr(const char* name, float v);
  void SetOutputRelErr(const char* name, float v);

  // Number of times to call InferenceSession::Run. The same feeds are used each time.
  // e.g. used to verify the generator ops behave as expected
  void SetNumRunCalls(int n) {
    ORT_ENFORCE(n > 0);
    num_run_calls_ = n;
  }

  using CustomOutputVerifierFn =
      std::function<void(const std::vector<OrtValue>& /*fetches*/, const std::string& /*provider_type*/)>;

  void SetCustomOutputVerifier(CustomOutputVerifierFn custom_output_verifier) {
    custom_output_verifier_ = custom_output_verifier;
  }

  void AddAttributeProto(ONNX_NAMESPACE::AttributeProto attr) {
    add_attribute_funcs_.emplace_back([attr = std::move(attr)](onnxruntime::Node& node) {
      node.AddAttributeProto(attr);
    });
  }

  template <typename T>
  void AddAttribute(std::string name, T value) {
    // Generate a the proper AddAttribute call for later
    add_attribute_funcs_.emplace_back([name = std::move(name), value = std::move(value)](onnxruntime::Node& node) {
      node.AddAttribute(name, value);
    });
  }

  enum class ExpectResult { kExpectSuccess,
                            kExpectFailure };

  void Run(ExpectResult expect_result = ExpectResult::kExpectSuccess, const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr,
           ExecutionMode execution_mode = ExecutionMode::ORT_SEQUENTIAL,
           const Graph::ResolveOptions& resolve_options = {});

  void Run(SessionOptions session_options,
           ExpectResult expect_result = ExpectResult::kExpectSuccess,
           const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr,
           const Graph::ResolveOptions& resolve_options = {},
           /*out*/ size_t* number_of_pre_packed_weights_counter = nullptr,
           /*out*/ size_t* number_of_shared_pre_packed_weights_counter = nullptr);

  std::vector<OrtValue>
  GetFetches() { return fetches_; }

  std::unique_ptr<onnxruntime::Model> BuildGraph(const std::unordered_map<std::string, int>& extra_domain_to_version = {},
                                                 bool allow_released_onnx_opset_only = true);

  // storing p_model as cache
  void SetModelCache(std::shared_ptr<onnxruntime::Model> model) {
    cached_model_ = model;
  }

  std::shared_ptr<onnxruntime::Model> GetModelCache() {
    return cached_model_;
  }

  // clear input/output data, fetches will be cleared in Run()
  void ClearData() {
    input_data_.clear();
    output_data_.clear();
    initializer_index_.clear();
  }

  struct Data {
    onnxruntime::NodeArg def_;
    OrtValue data_;
    optional<float> relative_error_;
    optional<float> absolute_error_;
    bool sort_output_;
    Data(onnxruntime::NodeArg&& def, OrtValue&& data, optional<float>&& rel, optional<float>&& abs,
         bool sort_output = false)
        : def_(std::move(def)),
          data_(std::move(data)),
          relative_error_(std::move(rel)),
          absolute_error_(abs),
          sort_output_(sort_output) {}
    Data(Data&&) = default;
    Data& operator=(Data&&) = default;
  };

  std::vector<Data>& GetInputData() {
    return input_data_;
  }

  std::vector<Data>& GetOutputData() {
    return output_data_;
  }

  void SetDeterminism(bool use_determinism) {
    use_determinism_ = use_determinism;
  }

  void EnableSharingOfPrePackedWeightsAcrossSessions() {
    add_prepacked_shared_container_to_sessions_ = true;
  }

  size_t GetNumPrePackedWeightsShared() const {
    return prepacked_weights_container_.GetNumberOfElements();
  }

  bool test_allow_released_onnx_opset_only_ = true;

 protected:
  // Set test_allow_released_onnx_opset_only_ to false or override this method and return false
  // if inheriting from OpTester to allow testing of a non-released ONNX opset operator
  virtual bool IsAllowReleasedONNXOpsetsOnlySetForThisTest() const {
    return test_allow_released_onnx_opset_only_;
  }

  virtual void AddNodes(onnxruntime::Graph& graph, std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                        std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                        std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs);

  void AddInitializers(onnxruntime::Graph& graph);

  void FillFeedsAndOutputNames(std::unordered_map<std::string, OrtValue>& feeds,
                               std::vector<std::string>& output_names);

  void FillFeeds(std::unordered_map<std::string, OrtValue>& feeds);

  template <class SessionType>
  std::vector<OrtValue> ExecuteModel(Model& model,
                                     SessionType& session_object,
                                     ExpectResult expect_result,
                                     const std::string& expected_failure_string,
                                     const RunOptions* run_options,
                                     const std::unordered_map<std::string, OrtValue>& feeds,
                                     const std::vector<std::string>& output_names,
                                     const std::string& provider_type,
                                     bool allow_released_onnx_opset_only = true);

  const char* op_;
  std::vector<Data> input_data_;
  std::vector<Data> output_data_;
  std::vector<OrtValue> fetches_;

  // for gradient unit tests only
  std::shared_ptr<onnxruntime::Model> cached_model_;

#ifndef NDEBUG
  bool run_called_{};
#endif

 protected:
  gsl::span<const int64_t> ToDimsSpan(const DimsVariant& dims_var) {
    gsl::span<const int64_t> result;
    switch (dims_var.index()) {
      case 0:
        result = std::get<0>(dims_var);
        break;
      case 1:
        result = std::get<1>(dims_var);
        break;
      default:
        ORT_THROW("Unhandled dims variant");
    }
    return result;
  }

  template <typename T>
  void AddData(std::vector<Data>& data, const char* name, const DimsVariant& dims_var, const T* values,
               int64_t values_count, bool is_initializer = false, bool sort_output = false,
               const std::vector<std::string>* dim_params = nullptr,
               float rel_error = 0.0f, float abs_error = 0.0f, bool is_optional_type_tensor = false) {
    auto dims = ToDimsSpan(dims_var);
#if defined(DISABLE_OPTIONAL_TYPE)
    if (is_optional_type_tensor) {
      ORT_THROW("Optional type is not supported in this build");
    }
#endif

    ORT_TRY {
      TensorShape shape{dims};

      OrtValue value;

      if (!is_optional_type_tensor || (is_optional_type_tensor && values != nullptr)) {
        // In case values is nullptr for optional type tensor, it means we are creating
        // an optional type tensor which is None and we hence skip values count validation
        ORT_ENFORCE(shape.Size() == values_count,
                    values_count, " input values doesn't match tensor size of ",
                    shape.Size());

        // If it is an optional tensor type with no values (i.e.) None,
        // we won't even pass it in to Run() as part of the feeds,
        // so we don't even have to create a Tensor.
        // Conversely, if it is an optional tensor type with values,
        // we pass it in as a regular tensor.
        auto allocator = test::AllocatorManager::Instance().GetAllocator(CPU);
        Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), shape, std::move(allocator), value);

        // values *could* be nullptr for a non-optional tensor if it is empty.
        // Update the data buffer of the input only if values if non-nullptr.
        if (values != nullptr) {
          auto* data_ptr = value.GetMutable<Tensor>()->template MutableData<T>();
          for (int64_t i = 0; i < values_count; i++) {
            data_ptr[i] = values[i];
          }
        }
      } else {  // "None" Tensor OrtValue. Initialize appropriately.
        auto ml_tensor = DataTypeImpl::GetType<Tensor>();
        value.Init(nullptr, ml_tensor, ml_tensor->GetDeleteFunc());
      }

      std::vector<int64_t> dims_for_proto = GetDimsForProto(dims);
      TTypeProto<T> tensor_type_proto(add_shape_to_tensor_data_ ? &dims_for_proto : nullptr);

#if !defined(DISABLE_OPTIONAL_TYPE)
      OptionalTypeProto optional_type_proto(tensor_type_proto.proto);
      auto node_arg = NodeArg(name, !is_optional_type_tensor ? &tensor_type_proto.proto : &optional_type_proto.proto);
#else
      auto node_arg = NodeArg(name, &tensor_type_proto.proto);
#endif

      AddShapeToTensorData(node_arg, dims, dim_params);

      optional<float> rel;
      optional<float> abs;

      if (rel_error != 0.0f) {
        rel = rel_error;
      }

      if (abs_error != 0.0f) {
        abs = abs_error;
      }

      data.push_back(Data(std::move(node_arg), std::move(value), std::move(rel), std::move(abs), sort_output));

      // Optional values cannot be initializers
      if (is_initializer && !is_optional_type_tensor) {
        initializer_index_.push_back(data.size() - 1);
      }
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        std::cerr << "AddData for '" << name << "' threw: " << ex.what();
      });
      ORT_RETHROW;
    }
  }

 private:
  template <typename T>
  void AddSeqData(std::vector<Data>& data, const char* name,
                  const SeqTensors<T>* seq_tensors,
                  bool is_optional_sequence_tensor_type = false) {
#if defined(DISABLE_OPTIONAL_TYPE)
    if (is_optional_sequence_tensor_type) {
      ORT_THROW("Optional type is not supported in this build");
    }
#endif

    std::unique_ptr<TensorSeq> ptr;

    if (seq_tensors) {
      auto num_tensors = seq_tensors->tensors.size();
      std::vector<Tensor> tensors;
      tensors.resize(num_tensors);
      auto elem_type = DataTypeImpl::GetType<T>();
      for (size_t i = 0; i < num_tensors; ++i) {
        TensorShape shape{seq_tensors->tensors[i].shape};
        auto values_count = static_cast<int64_t>(seq_tensors->tensors[i].data.size());
        ORT_ENFORCE(shape.Size() == values_count, values_count,
                    " input values doesn't match tensor size of ", shape.Size());

        auto allocator = test::AllocatorManager::Instance().GetAllocator(CPU);
        auto& tensor = tensors[i];

        tensor = Tensor(elem_type,
                        shape,
                        allocator);

        auto* data_ptr = tensor.template MutableData<T>();
        for (int64_t x = 0; x < values_count; ++x) {
          data_ptr[x] = seq_tensors->tensors[i].data[x];
        }
      }

      ptr = std::make_unique<TensorSeq>(elem_type);
      ptr->SetElements(std::move(tensors));
    }

    OrtValue value;
    auto mltype = DataTypeImpl::GetType<TensorSeq>();

    // nullptr means None OrtValue which we will skip inserting into the feeds
    value.Init(ptr ? ptr.release() : nullptr, mltype, mltype->GetDeleteFunc());

    SequenceTensorTypeProto<T> sequence_tensor_proto;
#if !defined(DISABLE_OPTIONAL_TYPE)
    OptionalTypeProto optional_type_proto(sequence_tensor_proto.proto);
    auto node_arg = NodeArg(name, !is_optional_sequence_tensor_type
                                      ? &sequence_tensor_proto.proto
                                      : &optional_type_proto.proto);
#else
    auto node_arg = NodeArg(name, &sequence_tensor_proto.proto);
#endif

    data.push_back(Data(std::move(node_arg), std::move(value), optional<float>(), optional<float>()));
  }

  std::vector<int64_t> GetDimsForProto(gsl::span<const int64_t> dims);

  void AddShapeToTensorData(NodeArg& node_arg, gsl::span<const int64_t> dims, const std::vector<std::string>* dim_params);

  void CopyDataToTensor(gsl::span<const gsl::byte> data, Tensor& dst);

#if !defined(DISABLE_SPARSE_TENSORS)
  NodeArg MakeSparseNodeArg(int32_t dtype, const char* name,
                            const gsl::span<const int64_t>& dims,
                            const std::vector<std::string>* dim_params);

  void AddSparseCooTensorData(std::vector<Data>& data,
                              MLDataType data_type,
                              const char* name,
                              gsl::span<const int64_t> dims,
                              gsl::span<const gsl::byte> values,
                              gsl::span<const int64_t> indices,
                              const CheckParams& check_params,
                              const std::vector<std::string>* dim_params = nullptr);

  void AddSparseCooTensorStrings(std::vector<Data>& data,
                                 const char* name,
                                 gsl::span<const int64_t> dims,
                                 gsl::span<const std::string> values,
                                 gsl::span<const int64_t> indices,
                                 const std::vector<std::string>* dim_params = nullptr);

  void AddSparseCsrTensorData(std::vector<Data>& data,
                              MLDataType data_type,
                              const char* name,
                              gsl::span<const int64_t> dims,
                              gsl::span<const gsl::byte> values,
                              gsl::span<const int64_t> inner_indices,
                              gsl::span<const int64_t> outer_indices,
                              const CheckParams& check_params,
                              const std::vector<std::string>* dim_params = nullptr);

  void AddSparseCsrTensorStrings(std::vector<Data>& data,
                                 const char* name,
                                 gsl::span<const int64_t> dims,
                                 gsl::span<const std::string> values,
                                 gsl::span<const int64_t> inner_indices,
                                 gsl::span<const int64_t> outer_indices,
                                 const std::vector<std::string>* dim_params = nullptr);

  void AddSparseTensorData(std::vector<Data>& data, NodeArg node_arg,
                           std::unique_ptr<SparseTensor> p_tensor,
                           const CheckParams& check_params);
#endif

  const char* domain_;
  int opset_version_;
  bool add_shape_to_tensor_data_ = true;
  int add_symbolic_dim_to_tensor_data_ = -1;
  int num_run_calls_ = 1;
  std::vector<size_t> initializer_index_;
  std::vector<std::function<void(onnxruntime::Node& node)>> add_attribute_funcs_;

  IOnnxRuntimeOpSchemaRegistryList custom_schema_registries_;
  std::vector<std::shared_ptr<CustomRegistry>> custom_session_registries_;

  bool verify_output_;

  bool use_determinism_ = false;

  CustomOutputVerifierFn custom_output_verifier_;

  bool add_prepacked_shared_container_to_sessions_ = false;

  onnxruntime::PrepackedWeightsContainer prepacked_weights_container_;
};

template <typename TException>
void ExpectThrow(OpTester& test, const std::string& error_msg) {
  ORT_TRY {
    test.Run();
    // should throw and not reach this
    EXPECT_TRUE(false) << "Expected Run() to throw";
  }
  ORT_CATCH(TException ex) {
    ORT_UNUSED_PARAMETER(error_msg);
    ORT_HANDLE_EXCEPTION([&]() {
      EXPECT_THAT(ex.what(), testing::HasSubstr(error_msg));
    });
  }
}

void DebugTrap();

void Check(const OpTester::Data& expected_data, const Tensor& output_tensor, const std::string& provider_type);

inline const Tensor& FetchTensor(const OrtValue& ort_value) {
  if (ort_value.Fence()) {
    ort_value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, 0);
  }
  return ort_value.Get<Tensor>();
}

inline void ConvertFloatToMLFloat16(const float* f_datat, MLFloat16* h_data, int input_size) {
  auto in_vector = ConstEigenVectorMap<float>(f_datat, input_size);
  auto output_vector = EigenVectorMap<Eigen::half>(static_cast<Eigen::half*>(static_cast<void*>(h_data)), input_size);
  output_vector = in_vector.template cast<Eigen::half>();
}

inline void ConvertMLFloat16ToFloat(const MLFloat16* h_data, float* f_data, int input_size) {
  auto in_vector =
      ConstEigenVectorMap<Eigen::half>(static_cast<const Eigen::half*>(static_cast<const void*>(h_data)), input_size);
  auto output_vector = EigenVectorMap<float>(f_data, input_size);
  output_vector = in_vector.template cast<float>();
}

inline std::vector<MLFloat16> FloatsToMLFloat16s(const std::vector<float>& f) {
  std::vector<MLFloat16> m(f.size());
  ConvertFloatToMLFloat16(f.data(), m.data(), static_cast<int>(f.size()));
  return m;
}

inline std::vector<BFloat16> MakeBFloat16(const std::initializer_list<float>& input) {
  std::vector<BFloat16> output;
  std::transform(input.begin(), input.end(), std::back_inserter(output), [](float f) { return BFloat16(f); });
  return output;
}

inline std::vector<BFloat16> FloatsToBFloat16s(const std::vector<float>& input) {
  std::vector<BFloat16> output;
  std::transform(input.begin(), input.end(), std::back_inserter(output), [](float f) { return BFloat16(f); });
  return output;
}

inline CheckParams MakeCheckParams(const OpTester::Data& d) {
  return CheckParams{d.sort_output_, d.absolute_error_, d.relative_error_};
}

inline std::vector<int64_t> GetShapeVector(const TensorShape& shape) {
  std::vector<int64_t> result;
  const auto dims = shape.GetDims();
  result.resize(dims.size());
  result.assign(dims.cbegin(), dims.cend());
  return result;
}

}  // namespace test
}  // namespace onnxruntime
