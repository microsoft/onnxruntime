// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <memory>

#include "core/framework/data_types.h"

namespace onnxruntime {
namespace ep {
namespace adapter {

/// <summary>
/// Gets an OrtMLDataType for a tensor type. Throws on error.
/// </summary>
/// <param name="elem_type"></param>
/// <returns></returns>
inline const OrtDataType* GetTensorType(ONNXTensorElementDataType elem_type) {
  const OrtEpApi& ep_api = Ort::GetEpApi();
  const OrtDataType* result = nullptr;

  Ort::ThrowOnError(ep_api.GetTensorDataType(elem_type, &result));
  return result;
}

inline const OrtDataType* MLDataTypeToOrtDataType(MLDataType ml_type) {
  auto tensor_type = ml_type->AsTensorType();
  EP_ENFORCE(tensor_type != nullptr, "EP Kernel registration only supports tensor types.");
  auto elem_type = tensor_type->GetElementType();
  auto primitive_type = static_cast<const PrimitiveDataTypeBase*>(elem_type);
  auto onnx_type = static_cast<ONNXTensorElementDataType>(primitive_type->GetDataType());
  return GetTensorType(onnx_type);
}

/// <summary>
/// An adapter class partially implementing the interface of `onnxruntime::KernelDefBuilder`.
/// </summary>
struct KernelDefBuilder {
  static std::unique_ptr<KernelDefBuilder> Create() { return std::make_unique<KernelDefBuilder>(); }

  explicit KernelDefBuilder() {}

  KernelDefBuilder& SetName(const char* op_name) {
    builder_.SetOperatorType(op_name);
    return *this;
  }

  KernelDefBuilder& SetDomain(const char* domain) {
    builder_.SetDomain(domain);
    return *this;
  }

  KernelDefBuilder& SinceVersion(int since_version) {
    return SinceVersion(since_version, INT_MAX);
  }

  KernelDefBuilder& SinceVersion(int since_version_start, int since_version_end) {
    builder_.SetSinceVersion(since_version_start, since_version_end);
    return *this;
  }

  KernelDefBuilder& Provider(const char* provider_type) {
    builder_.SetExecutionProvider(provider_type);
    return *this;
  }

  KernelDefBuilder& TypeConstraint(const char* arg_name, std::vector<MLDataType> types) {
    std::vector<const OrtDataType*> ort_types;
    ort_types.reserve(types.size());
    for (const auto& type : types) {
      ort_types.push_back(MLDataTypeToOrtDataType(type));
    }
    builder_.AddTypeConstraint(arg_name, ort_types);
    return *this;
  }

  KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType type) {
    builder_.AddTypeConstraint(arg_name, MLDataTypeToOrtDataType(type));
    return *this;
  }

  KernelDefBuilder& MayInplace(const std::vector<std::pair<int, int>>& inplaces) {
    for (const auto& pair : inplaces) {
      builder_.AddInputOutputMutableAlias(pair.first, pair.second);
    }
    return *this;
  }
  KernelDefBuilder& MayInplace(int input_index, int output_index) {
    builder_.AddInputOutputMutableAlias(input_index, output_index);
    return *this;
  }

  KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases) {
    for (const auto& pair : aliases) {
      builder_.AddInputOutputAlias(pair.first, pair.second);
    }
    return *this;
  }
  KernelDefBuilder& Alias(int input_index, int output_index) {
    builder_.AddInputOutputAlias(input_index, output_index);
    return *this;
  }

  KernelDefBuilder& InputMemoryType(OrtMemType type, int input_index) {
    builder_.SetInputMemType(input_index, type);
    return *this;
  }

  KernelDefBuilder& InputMemoryType(OrtMemType type, const std::vector<int>& input_indexes) {
    for (int input_index : input_indexes) {
      builder_.SetInputMemType(input_index, type);
    }
    return *this;
  }

  KernelDefBuilder& OutputMemoryType(OrtMemType type, int output_index) {
    builder_.SetOutputMemType(output_index, type);
    return *this;
  }

  KernelDefBuilder& OutputMemoryType(OrtMemType type, const std::vector<int>& output_indexes) {
    for (int output_index : output_indexes) {
      builder_.SetOutputMemType(output_index, type);
    }
    return *this;
  }

  KernelDefBuilder& ExecQueueId(int /*queue_id*/) { return *this; }

  Ort::KernelDef Build() { return builder_.Build(); }

 private:
  Ort::KernelDefBuilder builder_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
