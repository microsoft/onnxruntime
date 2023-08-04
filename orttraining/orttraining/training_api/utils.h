// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace onnxruntime {
namespace training {
namespace api {

struct ModelIdentifiers {
  // ModelIdentifiers struct enables an easy way to store and identify the models used for training, evaluation
  // and optimization.
  // The model can be specified by a path to the model file or by a span of bytes containing the model data. However,
  // only one of these two options can be used at a time.
  // Training Session creation api enforces this by allowing the user to specify only one of the two options.
  const std::optional<std::string> train_model_path, eval_model, optim_model = std::nullopt;
  gsl::span<const uint8_t> train_model_data;
  gsl::span<const uint8_t> eval_model_data;
  gsl::span<const uint8_t> optim_model_data;

  ModelIdentifiers(const std::string& train_model_uri,
                   const std::optional<std::string>& eval_model_uri,
                   const std::optional<std::string>& optim_model_uri)
      : train_model_path(std::optional<std::string>(train_model_uri)),
        eval_model(eval_model_uri),
        optim_model(optim_model_uri) {}

  ModelIdentifiers(gsl::span<const uint8_t> train_data,
                   gsl::span<const uint8_t> eval_data,
                   gsl::span<const uint8_t> optim_data)
      : train_model_data(train_data), eval_model_data(eval_data), optim_model_data(optim_data) {}
};

namespace utils {

// Get names of graph inputs and outputs
void GetGraphInputOutputNames(const std::unique_ptr<onnxruntime::InferenceSession>& session_object,
                              InlinedVector<std::string>& input_names,
                              InlinedVector<std::string>& output_names);
// Fetch the parameter name from suffix: name = param_name+suffix,
// returns True if suffix is present in name else False
bool GetParamNameFromSuffix(const std::string& name, const std::string& suffix, std::string& param_name);

// Fetch the parameter name from all possible gradient suffix: name = param_name+suffix
// returns True if suffix is present in name else False
bool GetParamNameFromGradient(const std::string& grad_name, std::string& param_name);

// Allocate OrtValue like the input ortvalue on the same device
Status CreateZeroValuedOrtValueLike(const SessionState& sess_state, const OrtValue& input_val, OrtValue& output_val);

// Create OrtValue from a single value of type T
template <typename T>
void WrapInOrtValue(T value,
                    OrtValue* p_ortvalue,
                    AllocatorPtr alloc = nullptr) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.CreatePreferredAllocators()[0];

  TensorShape shape({1});
  auto element_type = DataTypeImpl::GetType<T>();
  auto allocator = alloc ? alloc : cpu_allocator;
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);
  memcpy(p_tensor->MutableDataRaw(), reinterpret_cast<void*>(&value), p_tensor->SizeInBytes());

  p_ortvalue->Init(p_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

template <typename T>
T GetScalarFromOrtValue(OrtValue& ort_value) {
  const Tensor& tensor = ort_value.Get<Tensor>();
  const TensorShape& shape = tensor.Shape();
  size_t dim_count = shape.NumDimensions();
  // Be noted: TensorShape returns 1 for rank 0 tensor.
  ORT_ENFORCE(shape.Size() == 1 && (dim_count == 0 || dim_count == 1));
  T val;
  if (DataTypeImpl::GetType<T>() == tensor.DataType()) {
    val = *(tensor.template Data<T>());
  } else {
    ORT_THROW("OrtValue data type not supported.");
  }
  return val;
}

ONNX_NAMESPACE::TensorProto CopyTensorToTensorProto(const Tensor& tensor, const std::string& tensor_proto_name,
                                                    const DataTransferManager& data_transfer_manager);

}  // namespace utils
}  // namespace api
}  // namespace training
}  // namespace onnxruntime
