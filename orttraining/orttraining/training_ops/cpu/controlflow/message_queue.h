// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

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

  void AddOutputGrad(const OrtValue& ort_value) { output_grad.emplace_back(ort_value); }
  const std::vector<OrtValue>& GetOutputGrads() { return output_grad; }
  void ClearOutputGrads() { output_grad.clear(); }

 private:
  OrtMessageQueue() = default;
  ~OrtMessageQueue() = default;
  OrtMessageQueue(const OrtMessageQueue&) = delete;
  OrtMessageQueue& operator=(const OrtMessageQueue&) = delete;

  std::vector<OrtValue> output_grad;
};

}  // namespace contrib
}  // namespace onnxruntime
