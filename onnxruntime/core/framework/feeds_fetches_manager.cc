// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/feeds_fetches_manager.h"

#include "core/framework/execution_providers.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/utils.h"

namespace onnxruntime {
common::Status FeedsFetchesInfo::MapNamesToMLValueIdxs(const std::vector<std::string>& names,
                                                       const OrtValueNameIdxMap& ort_value_name_idx_map,
                                                       std::vector<int>& ort_value_idxs) {
  auto status = Status::OK();

  ort_value_idxs.reserve(names.size());

  for (const auto& name : names) {
    int idx;
    status = ort_value_name_idx_map.GetIdx(name, idx);
    ORT_RETURN_IF_ERROR(status);

    ort_value_idxs.push_back(idx);
  }

  return status;
}

Status FeedsFetchesInfo::SetMLValueIdxs(const OrtValueNameIdxMap& ort_value_name_idx_map) {
  auto status = MapNamesToMLValueIdxs(feed_names, ort_value_name_idx_map, feeds_mlvalue_idxs);
  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Error mapping feeds: " + status.ErrorMessage());
  }

  status = MapNamesToMLValueIdxs(output_names, ort_value_name_idx_map, fetches_mlvalue_idxs);
  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Error mapping output names: " + status.ErrorMessage());
  }

  return status;
}

Status FeedsFetchesManager::Create(const std::vector<std::string>& feed_names,
                                   const std::vector<std::string>& output_names,
                                   const OrtValueNameIdxMap& ort_value_name_idx_map,
                                   std::unique_ptr<FeedsFetchesManager>& feed_fetch_manager) {
  FeedsFetchesInfo info{feed_names, output_names, ort_value_name_idx_map};

  feed_fetch_manager = std::make_unique<FeedsFetchesManager>(std::move(info));

  return Status::OK();
}

FeedsFetchesManager::FeedsFetchesManager(FeedsFetchesInfo&& info)
    : feeds_fetches_info_{info} {
  // init with default values
  feeds_device_copy_info_.resize(info.feed_names.size());
  fetches_device_copy_info_.resize(info.output_names.size());
}

void FeedsFetchesManager::SetDeviceCopyChecks(DeviceCopyCheck input_copy_needed, DeviceCopyCheck output_copy_needed) {
  ORT_ENFORCE(input_copy_needed != DeviceCopyCheck::Unknown &&
              output_copy_needed != DeviceCopyCheck::Unknown);

  device_copy_checks_.input_copy_needed = input_copy_needed;
  device_copy_checks_.output_copy_needed = output_copy_needed;

  // make sure overall status is correct
  device_copy_checks_.status =
      input_copy_needed == DeviceCopyCheck::NoCopy && output_copy_needed == DeviceCopyCheck::NoCopy
          ? DeviceCopyCheck::NoCopy
          : DeviceCopyCheck::Copy;
}
}  // namespace onnxruntime
