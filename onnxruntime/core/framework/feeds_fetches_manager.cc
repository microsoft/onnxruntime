// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/feeds_fetches_manager.h"

#include "core/framework/execution_providers.h"
#include "core/framework/ort_value_name_idx_map.h"

namespace onnxruntime {
common::Status FeedsFetchesInfo::MapNamesToMLValueIdxs(const std::vector<std::string>& names,
                                                       const MLValueNameIdxMap& ort_value_name_idx_map,
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

Status FeedsFetchesInfo::SetMLValueIdxs(const MLValueNameIdxMap& ort_value_name_idx_map) {
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
                                   const MLValueNameIdxMap& ort_value_name_idx_map,
                                   std::unique_ptr<FeedsFetchesManager>& feed_fetch_manager) {
  FeedsFetchesInfo info;
  info.feed_names = feed_names;
  info.output_names = output_names;

  ORT_RETURN_IF_ERROR(info.SetMLValueIdxs(ort_value_name_idx_map));

  feed_fetch_manager = std::make_unique<FeedsFetchesManager>(std::move(info));

  return Status::OK();
}

void FeedsFetchesManager::SetDeviceCopyChecks(DeviceCopyChecks checks) {
  ORT_ENFORCE(checks.input_copy_needed != DeviceCopyCheck::Unknown &&
              checks.output_copy_needed != DeviceCopyCheck::Unknown);

  device_copy_checks_ = checks;

  // make sure overall status is correct
  device_copy_checks_.status =
      checks.input_copy_needed == DeviceCopyCheck::NoCopy && checks.output_copy_needed == DeviceCopyCheck::NoCopy
          ? DeviceCopyCheck::NoCopy
          : DeviceCopyCheck::Copy;
}
}  // namespace onnxruntime
