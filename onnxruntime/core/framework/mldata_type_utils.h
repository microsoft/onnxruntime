// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_types.h"
#include "core/graph/graph_viewer.h"
#include "onnx/defs/data_type_utils.h"

namespace onnxruntime {
namespace utils {
MLDataType GetMLDataType(const onnxruntime::NodeArg& arg);

#if !defined(DISABLE_OPTIONAL_TYPE)
inline bool IsOptionalTensor(MLDataType type) {
  return type->IsOptionalType() &&
         type->AsOptionalType()->GetElementType()->IsTensorType();
}

inline MLDataType GetElementTypeFromOptionalTensor(MLDataType type) {
  ORT_ENFORCE(IsOptionalTensor(type),
              "Provided type is not an optional tensor");

  return type->AsOptionalType()
      ->GetElementType()
      ->AsTensorType()
      ->GetElementType();
}

inline bool IsOptionalSeqTensor(MLDataType type) {
  return type->IsOptionalType() &&
         type->AsOptionalType()->GetElementType()->IsTensorSequenceType();
}

inline MLDataType GetElementTypeFromOptionalSeqTensor(MLDataType type) {
  ORT_ENFORCE(IsOptionalSeqTensor(type),
              "Provided type is not an optional sequence tensor");

  return type->AsOptionalType()
      ->GetElementType()
      ->AsSequenceTensorType()
      ->GetElementType();
}
#endif

}  // namespace utils
}  // namespace onnxruntime
