// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <string>
#include <unordered_map>
#include "core/framework/allocator.h"
#include "core/session/onnxruntime_c_api.h"

using namespace std;

/**
 * Configuration information for a Run call.
 */
struct OrtRunOptions {
  /// Log severity.  See https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/common/logging/severity.h
  /// Default = -1 (use the log severity from the InferenceSession that the Run is for).
  int run_log_severity_level = -1;
  int run_log_verbosity_level = 0;  ///< VLOG level if debug build and run_log_severity_level is 0 (VERBOSE).
  std::string run_tag;              ///< A tag for the Run() calls using this.

  // An execution provider specific run option is required for some execution providers, 
  // each of which has its own resources shared within its platform.
  unordered_map<string, void*> provider_run_options;

  // Override the default CPU allocator with a custom allocator passed from the hosting service during each run. 
  onnxruntime::AllocatorPtr custom_cpu_allocator;

  // Set to 'true' to ensure the termination of all the outstanding Run() calls
  // that use this OrtRunOptions instance. Some of the outstanding Run() calls may
  // be forced to terminate with an error status.
  bool terminate = false;

  // Set to 'true' to run only the nodes from feeds to required fetches.
  // So it is possible that only some of the nodes are executed.
  bool only_execute_path_to_fetches = false;

  // Set to 'true' to run in training mode.
  bool training_mode = false;

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
