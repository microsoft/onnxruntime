// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#ifndef SHARED_PROVIDER
#include "core/framework/ml_value.h"
#endif

namespace onnxruntime {
class ExecutionProviders;
class IExecutionProvider;
class OrtValueNameIdxMap;
class SessionState;

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
                   const std::vector<std::string>& output_names_in,
                   const OrtValueNameIdxMap& ort_value_name_idx_map)
      : feed_names{feed_names_in}, output_names{output_names_in} {
    ORT_THROW_IF_ERROR(SetMLValueIdxs(ort_value_name_idx_map));
  }

  static Status MapNamesToMLValueIdxs(const std::vector<std::string>& names,
                                      const OrtValueNameIdxMap& ort_value_name_idx_map,
                                      std::vector<int>& ort_value_idxs);

  // set the ort_value_idxs for the current values in feed_names and output_names
  Status SetMLValueIdxs(const OrtValueNameIdxMap& ort_value_name_idx_map);

  std::vector<std::string> feed_names;
  std::vector<std::string> output_names;

  std::vector<int> feeds_mlvalue_idxs;
  std::vector<int> fetches_mlvalue_idxs;
};

struct MLValueCopyInfo {
  OrtDevice source_device{};
  OrtDevice target_device{};  // default is CPU
};

class FeedsFetchesManager {
 public:
  static Status Create(const std::vector<std::string>& feed_names, const std::vector<std::string>& output_names,
                       const OrtValueNameIdxMap& ort_value_name_idx_map,
                       std::unique_ptr<FeedsFetchesManager>& feeds_fetches_manager);

  FeedsFetchesManager(FeedsFetchesInfo&& info);

  const FeedsFetchesInfo& GetFeedsFetchesInfo() const { return feeds_fetches_info_; }

  std::vector<MLValueCopyInfo>& GetMutableFeedsDeviceCopyInfo() { return feeds_device_copy_info_; }
  const std::vector<MLValueCopyInfo>& GetFeedsDeviceCopyInfo() const { return feeds_device_copy_info_; }

  std::vector<MLValueCopyInfo>& GetMutableFetchesDeviceCopyInfo() { return fetches_device_copy_info_; }
  const std::vector<MLValueCopyInfo>& GetFetchesDeviceCopyInfo() const { return fetches_device_copy_info_; }

  const DeviceCopyChecks& GetDeviceCopyChecks() const { return device_copy_checks_; }
  void SetDeviceCopyChecks(DeviceCopyCheck input_copy_needed, DeviceCopyCheck output_copy_needed);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FeedsFetchesManager);

  DeviceCopyChecks device_copy_checks_ = {};

  FeedsFetchesInfo feeds_fetches_info_;

  std::vector<MLValueCopyInfo> feeds_device_copy_info_;
  std::vector<MLValueCopyInfo> fetches_device_copy_info_;
};
}  // namespace onnxruntime
