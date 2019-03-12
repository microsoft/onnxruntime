// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/framework/ml_value.h"

namespace onnxruntime {
class ExecutionProviders;
class IExecutionProvider;
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
  FeedsFetchesInfo(const std::vector<std::string>& feed_names_in,
                   const std::vector<std::string>& output_names_in)
      : feed_names{feed_names_in}, output_names{output_names_in} {}

  static Status MapNamesToMLValueIdxs(const std::vector<std::string>& names,
                                      const MLValueNameIdxMap& mlvalue_name_idx_map,
                                      std::vector<int>& mlvalue_idxs);

  // set the mlvalue_idxs for the current values in feed_names and output_names
  Status SetMLValueIdxs(const MLValueNameIdxMap& mlvalue_name_idx_map);

  std::vector<std::string> feed_names;
  std::vector<std::string> output_names;

  std::vector<int> feeds_mlvalue_idxs;
  std::vector<int> fetches_mlvalue_idxs;
};

class FeedsFetchesManager {
 public:
  struct MLValueCopyInfo {
    int allocation_device_id = 0;
    const IExecutionProvider* allocation_provider = nullptr;
    const IExecutionProvider* copy_provider = nullptr;
  };

  static Status Create(const std::vector<std::string>& feed_names,
                       const std::vector<std::string>& output_names,
                       const MLValueNameIdxMap& mlvalue_name_idx_map,
                       std::unique_ptr<FeedsFetchesManager>& feeds_fetches_manager);

  FeedsFetchesManager(FeedsFetchesInfo&& info) : feeds_fetches_info_{info} {}

  const FeedsFetchesInfo& GetFeedsFetchesInfo() const { return feeds_fetches_info_; }

  std::vector<MLValueCopyInfo>& GetMutableFeedsDeviceCopiers() { return feeds_device_copiers_; }
  const std::vector<MLValueCopyInfo>& GetFeedsDeviceCopiers() const { return feeds_device_copiers_; }

  std::vector<bool>& GetMutableCanUseFetchDuringExecutionFlags() { return can_use_fetch_during_execution_flags_; }
  const std::vector<bool>& GetCanUseFetchDuringExecutionFlags() const { return can_use_fetch_during_execution_flags_; }

  std::vector<MLValueCopyInfo>& GetMutableFetchesDeviceCopiers() { return fetches_device_copiers_; }
  const std::vector<MLValueCopyInfo>& GetFetchesDeviceCopiers() const { return fetches_device_copiers_; }

  DeviceCopyChecks GetDeviceCopyChecks() const { return device_copy_checks_; }
  void SetDeviceCopyChecks(DeviceCopyChecks checks);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FeedsFetchesManager);

  DeviceCopyChecks device_copy_checks_ = {};

  FeedsFetchesInfo feeds_fetches_info_;

  std::vector<MLValueCopyInfo> feeds_device_copiers_;
  std::vector<bool> can_use_fetch_during_execution_flags_;
  std::vector<MLValueCopyInfo> fetches_device_copiers_;
};
}  // namespace onnxruntime
