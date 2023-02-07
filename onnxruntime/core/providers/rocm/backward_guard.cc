// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/rocm/backward_guard.h"

namespace onnxruntime {

thread_local bool BackwardPassGuard::is_backward_pass_;

BackwardPassGuard::BackwardPassGuard() {
  is_backward_pass_ = true;
}

BackwardPassGuard::~BackwardPassGuard() {
  is_backward_pass_ = false;
}

bool BackwardPassGuard::is_backward_pass() {
  return is_backward_pass_;
}

}  // namespace onnxruntime
