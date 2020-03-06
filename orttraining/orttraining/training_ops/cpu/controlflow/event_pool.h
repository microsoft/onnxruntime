// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <cstdint>
#include <mutex>
#include <memory>

namespace onnxruntime {
namespace contrib {

class OrtEventPool final {
 public:
  static OrtEventPool& GetInstance() {
    static OrtEventPool instance_;
    return instance_;
  }
  void CreateEvent(int64_t id);
  void RecordEvent(int64_t id);
  void DeleteEvent(int64_t id);
  bool QueryEvent(int64_t id);

 private:
  OrtEventPool() = default;
  ~OrtEventPool() = default;
  OrtEventPool(const OrtEventPool&) = delete;
  OrtEventPool& operator=(const OrtEventPool&) = delete;

  std::unordered_map<int64_t, volatile bool> pool_;
  std::mutex mutex_;
};

}  // namespace contrib
}  // namespace onnxruntime