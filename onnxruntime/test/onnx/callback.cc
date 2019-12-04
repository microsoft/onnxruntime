// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "callback.h"

namespace onnxruntime {
namespace test {
void OrtRunCallback(OrtCallback* f) noexcept {
  if (f == nullptr) return;
  if (f->f != nullptr) {
    f->f(f->param);
    delete f;
  }
}
}  // namespace test
}  // namespace onnxruntime
