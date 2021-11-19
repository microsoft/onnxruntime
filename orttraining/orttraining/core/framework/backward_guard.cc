// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "orttraining/core/framework/backward_guard.h"

namespace onnxruntime {
namespace training {

thread_local bool BackwardPassGuard::is_backward_pass_;

BackwardPassGuard::BackwardPassGuard(bool value) {
  is_backward_pass_ = value;
}

BackwardPassGuard::~BackwardPassGuard() {
  is_backward_pass_ = false;
}

bool BackwardPassGuard::is_backward_pass() {
  return is_backward_pass_;
}

}  // namespace training
}  // namespace onnxruntime
