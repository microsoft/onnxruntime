// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <optional>
#include "core/common/inlined_containers_fwd.h"

#ifndef SHARED_PROVIDER
#include "core/framework/ort_value.h"
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
  FeedsFetchesInfo(gsl::span<const std::string> feed_names_in,
                   gsl::span<const std::string> output_names_in,
                   const OrtValueNameIdxMap& ort_value_name_idx_map)
      : feed_names(),
        output_names() {
    feed_names.reserve(feed_names_in.size());
    feed_names.assign(feed_names_in.begin(), feed_names_in.end());
    output_names.reserve(output_names_in.size());
    output_names.assign(output_names_in.begin(), output_names_in.end());
    ORT_THROW_IF_ERROR(SetMLValueIdxs(ort_value_name_idx_map));
  }

  FeedsFetchesInfo(gsl::span<const std::string_view> feed_names_in,
                   gsl::span<const std::string> output_names_in,
                   const OrtValueNameIdxMap& ort_value_name_idx_map)
      : feed_names(),
        output_names() {
    feed_names.reserve(feed_names_in.size());
    feed_names.assign(feed_names_in.begin(), feed_names_in.end());
    output_names.reserve(output_names_in.size());
    output_names.assign(output_names_in.begin(), output_names_in.end());
    ORT_THROW_IF_ERROR(SetMLValueIdxs(ort_value_name_idx_map));
  }

  static Status MapNamesToMLValueIdxs(gsl::span<const std::string> names,
                                      const OrtValueNameIdxMap& ort_value_name_idx_map,
                                      InlinedVector<int>& ort_value_idxs);

  // set the ort_value_idxs for the current values in feed_names and output_names
  Status SetMLValueIdxs(const OrtValueNameIdxMap& ort_value_name_idx_map);

  InlinedVector<std::string> feed_names;
  InlinedVector<std::string> output_names;

  InlinedVector<int> feeds_mlvalue_idxs;
  InlinedVector<int> fetches_mlvalue_idxs;
};

struct MLValueCopyInfo {
  OrtDevice source_device{};
  OrtDevice target_device{};  // default is CPU
};

class FeedsFetchesManager {
 public:
  static Status Create(gsl::span<const std::string> feed_names, gsl::span<const std::string> output_names,
                       const OrtValueNameIdxMap& ort_value_name_idx_map,
                       std::unique_ptr<FeedsFetchesManager>& feeds_fetches_manager);

  static Status Create(gsl::span<const std::string_view> feed_names, gsl::span<const std::string> output_names,
                       const OrtValueNameIdxMap& ort_value_name_idx_map,
                       std::optional<FeedsFetchesManager>& feeds_fetches_manager);

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
