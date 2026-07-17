// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "core/common/inlined_containers_fwd.h"
#include "core/framework/cancellation.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/config_options.h"

namespace onnxruntime {
namespace lora {
class LoraAdapter;
}
}  // namespace onnxruntime

/**
 * Configuration information for a Run call.
 */
struct OrtRunOptions {
  /// Log severity.  See https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/severity.h
  /// Default = -1 (use the log severity from the InferenceSession that the Run is for).
  int run_log_severity_level = -1;
  int run_log_verbosity_level = 0;  ///< VLOG level if debug build and run_log_severity_level is 0 (VERBOSE).
  std::string run_tag;              ///< A tag for the Run() calls using this.

  // Set to 'true' to run only the nodes from feeds to required fetches.
  // So it is possible that only some of the nodes are executed.
  bool only_execute_path_to_fetches = false;

  // Set to 'true' to enable profiling for this run.
  bool enable_profiling = false;

  // File prefix for profiling result for this run.
  // The actual filename will be: <profile_file_prefix>_<timestamp>.json
  // Only used when enable_profiling is true.
  std::basic_string<ORTCHAR_T> profile_file_prefix = ORT_TSTR("onnxruntime_run_profile");

#ifdef ENABLE_TRAINING
  // Used by onnxruntime::training::TrainingSession. This class is now deprecated.
  // Delete training_mode when TrainingSession is deleted.
  // Set to 'true' to run in training mode.
  bool training_mode = true;
#endif

  // Stores the configurations for this run
  // To add a configuration value to this specific run, call OrtApis::AddRunConfigEntry
  // To get a configuration value, call OrtApis::GetRunConfigEntry
  // The configuration keys and value formats are defined in
  // /include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h
  onnxruntime::ConfigOptions config_options;

  onnxruntime::InlinedVector<const onnxruntime::lora::LoraAdapter*> active_adapters;

  // Optional sync stream for external resource import.
  // When set, the EP uses this stream for execution, enabling proper
  // synchronization with imported external semaphores.
  OrtSyncStream* sync_stream = nullptr;

  OrtRunOptions();
  OrtRunOptions(const OrtRunOptions& other);
  OrtRunOptions& operator=(const OrtRunOptions& other);
  OrtRunOptions(OrtRunOptions&&) noexcept = default;
  OrtRunOptions& operator=(OrtRunOptions&&) noexcept = default;
  ~OrtRunOptions() = default;

  // The token snapshots the current termination state. ResetTerminate replaces
  // that state for future snapshots without reviving already-stopped tokens.
  onnxruntime::CancellationToken GetTerminateToken() const;
  void RequestTerminate();
  void ResetTerminate();

 private:
  // Thread-safe holder so RunOptionsSetTerminate/RunOptionsUnsetTerminate can be
  // called concurrently with active runs. Reset() installs a fresh source for
  // future snapshots while previously-snapshotted, in-flight tokens stay stopped.
  class TerminationState {
   public:
    explicit TerminationState(bool stop_requested = false) {
      if (stop_requested) {
        source_.request_stop();
      }
    }

    onnxruntime::CancellationToken GetToken() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return source_.get_token();
    }

    void RequestStop() {
      onnxruntime::CancellationSource source;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        source = source_;
      }

      source.request_stop();
    }

    void Reset() {
      std::lock_guard<std::mutex> lock(mutex_);
      source_ = onnxruntime::CancellationSource{};
    }

   private:
    mutable std::mutex mutex_;
    onnxruntime::CancellationSource source_;
  };

  std::shared_ptr<TerminationState> termination_state_;
};

// The special members and termination helpers are defined inline (rather than in
// run_options.cc) so that OrtRunOptions stays self-contained in every translation
// unit and dynamically loaded provider library. ONNX Runtime only exports the C
// API from its shared library, so an out-of-line OrtRunOptions constructor would
// be an unresolved symbol when a provider .so that constructs RunOptions is loaded.
inline OrtRunOptions::OrtRunOptions()
    : termination_state_{std::make_shared<TerminationState>()} {
}

inline OrtRunOptions::OrtRunOptions(const OrtRunOptions& other)
    : run_log_severity_level{other.run_log_severity_level},
      run_log_verbosity_level{other.run_log_verbosity_level},
      run_tag{other.run_tag},
      only_execute_path_to_fetches{other.only_execute_path_to_fetches},
      enable_profiling{other.enable_profiling},
      profile_file_prefix{other.profile_file_prefix},
#ifdef ENABLE_TRAINING
      training_mode{other.training_mode},
#endif
      config_options{other.config_options},
      active_adapters{other.active_adapters},
      sync_stream{other.sync_stream},
      termination_state_{
          std::make_shared<TerminationState>(other.GetTerminateToken().stop_requested())} {
}

inline OrtRunOptions& OrtRunOptions::operator=(const OrtRunOptions& other) {
  if (this != &other) {
    OrtRunOptions copy{other};
    *this = std::move(copy);
  }

  return *this;
}

inline onnxruntime::CancellationToken OrtRunOptions::GetTerminateToken() const {
  return termination_state_->GetToken();
}

inline void OrtRunOptions::RequestTerminate() {
  termination_state_->RequestStop();
}

inline void OrtRunOptions::ResetTerminate() {
  termination_state_->Reset();
}

namespace onnxruntime {
using RunOptions = ::OrtRunOptions;
}  // namespace onnxruntime
