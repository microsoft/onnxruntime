// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {
struct OrtCallback {
  void (*f)(void* param) noexcept;
  void* param;
};

/**
 *  f will be freed in this call
 */
void OrtRunCallback(OrtCallback* f) noexcept;
}  // namespace onnxruntime