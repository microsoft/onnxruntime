// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <atomic>
#include "core/session/onnxruntime_c_api.h"

/**
 * Configuration information for a Run call.
 */
struct OrtRunOptions {
  unsigned run_log_verbosity_level = 0;  ///< Logging level
  std::string run_tag;                   ///< A tag for the Run() calls using this.

  // Set to 'true' to ensure the termination of all the outstanding Run() calls 
  // that use this OrtRunOptions instance. Some of the outstanding Run() calls may
  // be forced to terminate with an error status.
  bool terminate = false;
  
  OrtRunOptions() = default;
  ~OrtRunOptions() = default;

  // Disable copy, move and assignment. we don't want accidental copies, to
  // ensure that the instance provided to the Run() call never changes and the
  // terminate mechanism will work.
  OrtRunOptions(const OrtRunOptions&) = delete;
  OrtRunOptions(OrtRunOptions&&) = delete;
  OrtRunOptions& operator=(const OrtRunOptions&) = delete;
  OrtRunOptions& operator=(OrtRunOptions&&) = delete;
};

namespace onnxruntime {
using RunOptions = OrtRunOptions;
}
