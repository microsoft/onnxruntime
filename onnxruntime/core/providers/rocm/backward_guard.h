// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {

struct BackwardPassGuard {
  BackwardPassGuard();
  ~BackwardPassGuard();
  static bool is_backward_pass();

 private:
  static thread_local bool is_backward_pass_;
};

}  // namespace onnxruntime
