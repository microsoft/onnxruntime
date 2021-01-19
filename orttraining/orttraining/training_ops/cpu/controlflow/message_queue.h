// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <queue>

#include "core/common/common.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {
namespace contrib {

class OrtMessageQueue final {
 public:
  static OrtMessageQueue& GetInstance() {
    static OrtMessageQueue instance_;
    return instance_;
  }

  void PushOutputGrad(const OrtValue& ort_value) { output_grads.emplace(ort_value); }
  OrtValue PopOutputGrad() {
    OrtValue ort_value = output_grads.front();
    output_grads.pop();
    return ort_value;
  }

 private:
  OrtMessageQueue() = default;
  ~OrtMessageQueue() = default;
  OrtMessageQueue(const OrtMessageQueue&) = delete;
  OrtMessageQueue& operator=(const OrtMessageQueue&) = delete;

  std::queue<OrtValue> output_grads;
};

}  // namespace contrib
}  // namespace onnxruntime
