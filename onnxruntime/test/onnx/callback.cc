// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "callback.h"

namespace onnxruntime {
namespace test {

void OrtCallback::Run() noexcept {
  if (f != nullptr) {
    f(param);
  }
}

}  // namespace test
}  // namespace onnxruntime
