// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace onnxruntime {
namespace training {
namespace api {
namespace utils {

// Get names of graph inputs and outputs
void GetGraphInputOutputNames(const std::unique_ptr<onnxruntime::InferenceSession>& session_object,
                              std::vector<std::string>& input_names,
                              std::vector<std::string>& output_names);
// Fetch the parameter name from suffix: name = param_name+suffix,
// returns True if suffix is present in name else False
bool GetParamNameFromSuffix(const std::string& name, const std::string& suffix, std::string& param_name);

// Fetch the parameter name from all possible gradient suffix: name = param_name+suffix
// returns True if suffix is present in name else False
bool GetParamNameFromGradient(const std::string& grad_name, std::string& param_name);

// Allocate OrtValue like the input ortvalue on the same device
Status OrtValueLike(const SessionState& sess_state, const OrtValue& input_val, OrtValue& output_val);

// Create OrtValue from a single value of type T
template <typename T>
void WrapInOrtValue(T value,
                    OrtValue* p_ortvalue,
                    AllocatorPtr alloc = nullptr) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.GetAllocator(0, OrtMemTypeDefault);

  TensorShape shape({1});
  auto element_type = DataTypeImpl::GetType<T>();
  auto allocator = alloc ? alloc : cpu_allocator;
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);
  memcpy(p_tensor->MutableDataRaw(), reinterpret_cast<void*>(&value), p_tensor->SizeInBytes());

  p_ortvalue->Init(p_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

// Create OrtValue on CPU out of provided inputs
template <typename T>
static void CreateInputOrtValue(gsl::span<const int64_t> dims,
                                const std::vector<T>& value,
                                OrtValue* p_ortvalue,
                                AllocatorPtr alloc = nullptr) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.GetAllocator(0, OrtMemTypeDefault);

  TensorShape shape(dims);
  assert(shape.Size() == static_cast<int64_t>(value.size()));
  auto element_type = DataTypeImpl::GetType<T>();
  auto allocator = alloc ? alloc : cpu_allocator;
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);

  // TODO: Handle memcpy for other allocators
  if (value.size() > 0 && !alloc) {  // using CPU allocator
    memcpy(p_tensor->MutableDataRaw(), value.data(), p_tensor->SizeInBytes());
  }

  p_ortvalue->Init(p_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

template <typename T>
T GetValue(OrtValue& ort_value) {
  const Tensor& tensor = ort_value.Get<Tensor>();
  T val;
  if (DataTypeImpl::GetType<T>() == tensor.DataType()) {
    val = *(tensor.template Data<T>());
  } else {
    ORT_THROW("OrtValue data type not supported.");
  }
  return val;
}

}  // namespace utils
}  // namespace api
}  // namespace training
}  // namespace onnxruntime
