// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <memory>
#include "core/common/common.h"
#include "core/common/status.h"

namespace onnxruntime {
/**
   Provides the runtime environment for onnxruntime.
   Create one instance for the duration of execution.
*/
class Environment {
 public:
  /**
     Create and initialize the runtime environment.
  */
  static Status Create(std::unique_ptr<Environment>& environment);

  /**
     This function will call ::google::protobuf::ShutdownProtobufLibrary
  */
  ~Environment();

  /**
     Returns whether any runtime environment instance has been initialized.
  */
  static bool IsInitialized() { return is_initialized_; }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Environment);

  Environment() = default;
  Status Initialize();

  static std::atomic<bool> is_initialized_;
};
}  // namespace onnxruntime
