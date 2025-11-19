// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <functional>
#include <vector>
#include <cmath>
#include <filesystem>
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/allocator.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/optimizer/graph_transformer.h"
#include "core/util/math.h"

namespace onnxruntime {

class Initializer final {
 public:
  // Construct an initializer with the provided name and data type, with all values initialized to 0
  Initializer(ONNX_NAMESPACE::TensorProto_DataType data_type,
              std::string_view name,
              gsl::span<const int64_t> dims);

  Initializer(const ONNX_NAMESPACE::TensorProto& tensor_proto,
              const std::filesystem::path& model_path = {});

  Initializer(const Graph& graph, const ONNX_NAMESPACE::TensorProto& tensor_proto,
              const std::filesystem::path& model_path = {}, bool check_outer_scope = false);

  ~Initializer();

  /// <summary>
  /// This function creates a new tensor_proto with a complete copy of the data
  /// </summary>
  /// <param name="tensor_proto">output</param>
  void ToProto(ONNX_NAMESPACE::TensorProto& tensor_proto) const {
    tensor_proto = utils::TensorToTensorProto(*data_, name_);
  }

  /// <summary>
  /// This function creates a pair of TensorProto and OrtValue. Unless the data
  /// is short, tensor_proto will be a reference to the data in OrtValue.
  /// Useful when adding a new initializer to the graph with external data in memory.
  /// </summary>
  /// <param name="tensor_proto"></param>
  /// <param name="ort_value"></param>
  void ToProtoWithOrtValue(ONNX_NAMESPACE::TensorProto& tensor_proto, OrtValue& ort_value) const;

#if !defined(ORT_EXTENDED_MINIMAL_BUILD)
  // XXX: Below two used only in training, convert to OrtValue result
  ONNX_NAMESPACE::TensorProto ToFP16(const std::string& name) const;
  ONNX_NAMESPACE::TensorProto ToBFloat16(const std::string& name) const;

  OrtValue ToFP16() const;
  OrtValue ToBFloat16() const;
  OrtValue ToFloat32(onnxruntime::concurrency::ThreadPool* thread_pool = nullptr) const;

#endif  // ORT_EXTENDED_MINIMAL_BUILD
  int data_type() const {
    return data_->GetElementType();
  }

  const std::string& name() const {
    return name_;
  }

  template <typename T>
  T* data() {
    return data_->MutableData<T>();
  }

  template <typename T>
  const T* data() const {
    return data_->Data<T>();
  }

  const void* data_raw() const {
    return data_->DataRaw();
  }

  void* mutable_data_raw() {
    return data_->MutableDataRaw();
  }

  template <typename T>
  auto DataAsSpan() const {
    return data_->DataAsSpan<T>();
  }

  gsl::span<const uint8_t> DataAsByteSpan() const {
    return gsl::make_span(reinterpret_cast<const uint8_t*>(data_->DataRaw()), data_->SizeInBytes());
  }

  gsl::span<uint8_t> MutableDataAsByteSpan() {
    return gsl::make_span(reinterpret_cast<uint8_t*>(data_->MutableDataRaw()), data_->SizeInBytes());
  }

  gsl::span<const int64_t> dims() const {
    return data_->Shape().GetDims();
  }

  size_t size() const { return narrow<size_t>(data_->Shape().Size()); }

#if !defined(ORT_EXTENDED_MINIMAL_BUILD)
  Initializer& add(float value);

  Initializer& add(const Initializer& other);

  Initializer& sub(const Initializer& other);

  Initializer& mul(const Initializer& other);

  Initializer& div(const Initializer& other);

  Initializer& sqrt();

  void scale_by_axis(const Initializer& other, int axis, bool column_major = false);
#endif  // ORT_EXTENDED_MINIMAL_BUILD
 private:
  std::string name_;
  OrtValue ort_value_;
  Tensor* data_;
};

}  // namespace onnxruntime
