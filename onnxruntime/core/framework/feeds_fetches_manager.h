// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/framework/ml_value.h"

namespace onnxruntime {
class ExecutionProviders;
class MLValueNameIdxMap;

enum class DeviceCopyCheck {
  Unknown,
  NoCopy,
  Copy
};

struct DeviceCopyChecks {
  DeviceCopyCheck status = DeviceCopyCheck::Unknown;  ///< Overall status. If NoCopy no input or output copies are needed
  DeviceCopyCheck input_copy_needed = DeviceCopyCheck::Unknown;
  DeviceCopyCheck output_copy_needed = DeviceCopyCheck::Unknown;
};

struct FeedsFetchesInfo {
  FeedsFetchesInfo() = default;
  FeedsFetchesInfo(std::vector<std::string>& feed_names_in,
                   std::vector<std::string>& output_names_in)
      : feed_names{feed_names_in}, output_names{output_names_in} {}

  // set the mlvalue_idxs for the current values in feed_names and output_names
  Status SetMLValueIdxs(const MLValueNameIdxMap& mlvalue_name_idx_map);

  std::vector<std::string> feed_names;
  std::vector<std::string> output_names;

  std::vector<int> feeds_mlvalue_idxs;
  std::vector<int> fetches_mlvalue_idxs;
};

class FeedsFetchesManager {
 public:
  using MLValueCopyFunc = std::function<Status(const MLValue&, MLValue&)>;

  static Status Create(const std::vector<std::string>& feed_names,
                       const std::vector<std::string>& output_names,
                       const MLValueNameIdxMap& mlvalue_name_idx_map,
                       std::unique_ptr<FeedsFetchesManager>& feeds_fetches_manager);

  FeedsFetchesManager(FeedsFetchesInfo&& info) : feeds_fetches_info_{info} {}

  // check if all the execution providers use the same allocator. if so, no copies between devices should be required,
  // and the overall status for DeviceCopyChecks can be set to NoCopy
  DeviceCopyCheck CheckExecutionProviders(const ExecutionProviders& execution_providers);

  const FeedsFetchesInfo& GetFeedsFetchesInfo() const { return feeds_fetches_info_; }

  std::vector<MLValueCopyFunc>& GetFeedsDeviceCopiers() { return feeds_device_copiers_; }
  std::vector<bool>& GetCanUseFetchDuringExecutionFlags() { return can_use_fetch_during_execution_flags_; }
  std::vector<MLValueCopyFunc>& GetFetchesDeviceCopiers() { return fetches_device_copiers_; }

  DeviceCopyChecks GetDeviceCopyChecks() const { return device_copy_checks_; }
  void SetDeviceCopyChecks(DeviceCopyChecks checks);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FeedsFetchesManager);

  DeviceCopyChecks device_copy_checks_ = {};

  FeedsFetchesInfo feeds_fetches_info_;

  std::vector<MLValueCopyFunc> feeds_device_copiers_;
  std::vector<bool> can_use_fetch_during_execution_flags_;
  std::vector<MLValueCopyFunc> fetches_device_copiers_;
};
}  // namespace onnxruntime
