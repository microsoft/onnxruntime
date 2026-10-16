// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <string>
#include <string_view>

#include "core/common/common.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"

namespace onnxruntime::stateful_execution {

inline const NodeArg* FindArgument(gsl::span<const NodeArg* const> arguments, std::string_view name) {
  auto it = std::find_if(arguments.begin(), arguments.end(),
                         [&](const NodeArg* arg) { return arg->Name() == name; });
  return it == arguments.end() ? nullptr : *it;
}

inline Status CheckTensorType(const NodeArg& arg) {
  const auto* type = arg.TypeAsProto();
  ORT_RETURN_IF_NOT(type && type->has_tensor_type() &&
                        type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_STRING,
                    "Stateful execution supports only dense, non-string tensors: ", arg.Name());
  return Status::OK();
}

inline Status CheckInput(const NodeArg& arg, const OrtValue& value) {
  ORT_RETURN_IF_NOT(value.IsTensor(), "Stateful input is not a tensor: ", arg.Name());
  const Tensor& tensor = value.Get<Tensor>();
  const auto& type = arg.TypeAsProto()->tensor_type();
  ORT_RETURN_IF_NOT(tensor.GetElementType() == type.elem_type(), "Stateful input dtype mismatch: ", arg.Name());
  if (type.has_shape()) {
    ORT_RETURN_IF_NOT(tensor.Shape().NumDimensions() == static_cast<size_t>(type.shape().dim_size()),
                      "Stateful input rank mismatch: ", arg.Name());
    for (int i = 0; i < type.shape().dim_size(); ++i) {
      const auto& dimension = type.shape().dim(i);
      ORT_RETURN_IF(dimension.has_dim_value() && dimension.dim_value() != tensor.Shape()[i],
                    "Stateful input dimension mismatch: ", arg.Name());
    }
  }
  return Status::OK();
}

inline Status CheckCompatibleTypes(const NodeArg& source, const NodeArg& destination) {
  ORT_RETURN_IF_ERROR(CheckTensorType(source));
  ORT_RETURN_IF_ERROR(CheckTensorType(destination));
  ORT_RETURN_IF_NOT(source.TypeAsProto()->tensor_type().elem_type() ==
                        destination.TypeAsProto()->tensor_type().elem_type(),
                    "Binding dtype mismatch: ", source.Name(), " -> ", destination.Name());
  const auto* from = source.Shape();
  const auto* to = destination.Shape();
  if (from && to) {
    ORT_RETURN_IF(from->dim_size() != to->dim_size(),
                  "Binding rank mismatch: ", source.Name(), " -> ", destination.Name());
    for (int i = 0; i < from->dim_size(); ++i) {
      const auto& source_dim = from->dim(i);
      const auto& destination_dim = to->dim(i);
      ORT_RETURN_IF(source_dim.has_dim_value() && destination_dim.has_dim_value() &&
                        source_dim.dim_value() != destination_dim.dim_value(),
                    "Binding dimension mismatch: ", source.Name(), " -> ", destination.Name());
    }
  }
  return Status::OK();
}

inline size_t FindOutput(gsl::span<const std::string> names, std::string_view name) {
  return static_cast<size_t>(std::find(names.begin(), names.end(), name) - names.begin());
}

}  // namespace onnxruntime::stateful_execution
