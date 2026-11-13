// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <utility>

#include "core/common/common.h"
#include "core/common/status.h"

namespace onnxruntime {
#ifndef SHARED_PROVIDER
class Tensor;
#endif

namespace cuda {

inline constexpr size_t kGdsIoAlignment = 4096;

template <typename Driver>
class GdsDriverHandle {
 public:
  GdsDriverHandle() : state_(SharedState()) {}

  ~GdsDriverHandle() {
    // Lock before the final strong reference expires, not inside Driver's destructor.
    std::lock_guard lock(state_->mutex);
    driver_.reset();
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GdsDriverHandle);

  template <typename... Args>
  common::Status Acquire(Args&&... args) {
    std::lock_guard lock(state_->mutex);
    if (driver_) {
      return common::Status::OK();
    }

    driver_ = state_->driver.lock();
    if (!driver_) {
      auto candidate = std::make_shared<Driver>(std::forward<Args>(args)...);
      ORT_RETURN_IF_ERROR(candidate->Initialize());
      state_->driver = candidate;
      driver_ = std::move(candidate);
    }
    return common::Status::OK();
  }

  Driver* operator->() const { return driver_.get(); }

 private:
  struct State {
    std::mutex mutex;
    std::weak_ptr<Driver> driver;
  };

  static std::shared_ptr<State> SharedState() {
    static auto state = std::make_shared<State>();
    return state;
  }

  std::shared_ptr<State> state_;
  std::shared_ptr<Driver> driver_;
};

class GdsLoader {
 public:
  virtual ~GdsLoader() = default;

  virtual common::Status Load(int file_descriptor,
                              int64_t data_offset,
                              size_t data_length,
                              Tensor& tensor) const = 0;

  static common::Status Create(int device_id, std::unique_ptr<GdsLoader>& loader);
};

}  // namespace cuda
}  // namespace onnxruntime
