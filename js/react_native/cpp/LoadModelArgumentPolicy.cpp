// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "LoadModelArgumentPolicy.h"

namespace onnxruntimejsi {

LoadModelArgumentLayout resolveLoadModelArgumentLayout(
    const LoadModelArgumentType* argumentTypes, size_t argumentCount) noexcept {
  if (argumentCount == 2 &&
      argumentTypes[0] == LoadModelArgumentType::String &&
      argumentTypes[1] == LoadModelArgumentType::Object) {
    return {true, false, 1};
  }

  if (argumentCount == 4 &&
      argumentTypes[0] == LoadModelArgumentType::ArrayBuffer &&
      argumentTypes[1] == LoadModelArgumentType::Number &&
      argumentTypes[2] == LoadModelArgumentType::Number &&
      argumentTypes[3] == LoadModelArgumentType::Object) {
    return {true, true, 3};
  }

  return {};
}

}  // namespace onnxruntimejsi
