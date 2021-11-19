// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {
namespace training {

struct BackwardPassGuard {
  BackwardPassGuard(bool value = true);
  ~BackwardPassGuard();
  static bool is_backward_pass();
private:
  static thread_local bool is_backward_pass_;
};

}  // namespace training
}  // namespace onnxruntime
