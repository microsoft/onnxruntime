// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/callback.h"
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif
namespace onnxruntime {
void OrtRunCallback(OrtCallback* f) noexcept {
  if (f == nullptr) return;
  if (f->f != nullptr) {
    f->f(f->param);
    delete f;
  }
}
}  // namespace onnxruntime
