// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/framework/feeds_fetches_manager.h"

namespace onnxruntime {
namespace controlflow {
namespace detail {

// helper to execute the subgraph by calling the Execute method of the provided implementation class with
// with the cached or newly created FeedsFetchesManager
template <typename TImpl>
common::Status SubgraphExecuteHelper(std::unique_ptr<FeedsFetchesManager>& cached_feeds_fetches_manager, TImpl& impl) {
  auto status = Status::OK();

  if (cached_feeds_fetches_manager) {
    // make it clear we don't update this instance when executing so there are no potential concurrency issues
    const FeedsFetchesManager* cached_ffm = &*cached_feeds_fetches_manager;
    status = impl.Execute(nullptr, cached_ffm);
  } else {
    // use a local instance until we know we're successful, and cache if it is
    std::unique_ptr<FeedsFetchesManager> new_ffm;
    ORT_RETURN_IF_ERROR(impl.CreateFeedsFetchesManager(new_ffm));

    status = impl.Execute(&*new_ffm, nullptr);
    if (status.IsOK()) {
      cached_feeds_fetches_manager = std::move(new_ffm);
    }
  }

  return status;
}

}  // namespace detail
}  // namespace controlflow
}  // namespace onnxruntime
