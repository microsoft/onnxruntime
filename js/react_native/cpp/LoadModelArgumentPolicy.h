// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

namespace onnxruntimejsi {

enum class LoadModelArgumentType {
  String,
  ArrayBuffer,
  Number,
  Object,
  Other,
};

struct LoadModelArgumentLayout {
  bool valid = false;
  bool usesModelBuffer = false;
  size_t optionsIndex = 0;
};

/**
 * @brief Validate a React Native loadModel overload and identify its options argument.
 */
LoadModelArgumentLayout resolveLoadModelArgumentLayout(
    const LoadModelArgumentType* argumentTypes, size_t argumentCount) noexcept;

}  // namespace onnxruntimejsi
