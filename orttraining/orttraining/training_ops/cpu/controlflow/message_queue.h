// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <queue>
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

  void Push(const OrtValue& ort_value) { ort_values.emplace(ort_value); }
  OrtValue Pop() {
    OrtValue ort_value = ort_values.front();
    ort_values.pop();
    return ort_value;
  }

  void PopAll(std::vector<OrtValue>& results) {
    while (!ort_values.empty()) {
      OrtValue ort_value = ort_values.front();
      ort_values.pop();
      results.emplace_back(ort_value);
    }
  }

 private:
  OrtMessageQueue() = default;
  ~OrtMessageQueue() = default;
  OrtMessageQueue(const OrtMessageQueue&) = delete;
  OrtMessageQueue& operator=(const OrtMessageQueue&) = delete;

  std::queue<OrtValue> ort_values;
};

}  // namespace contrib
}  // namespace onnxruntime
