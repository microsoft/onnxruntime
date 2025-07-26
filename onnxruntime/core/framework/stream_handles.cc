// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/stream_handles.h"

#include <algorithm>

namespace onnxruntime {

void Stream::UpdateWithAwaitedNotification(const synchronize::Notification& notification) {
  const std::unordered_map<const Stream*, uint64_t>& stream_sync_info = notification.GetStreamSyncInfo();
  for (const auto& kv : stream_sync_info) {
    auto ret = producer_stream_sync_info_.insert(kv);
    if (!ret.second) {
      // we already have an entry. use the highest value for the producer stream.
      ret.first->second = std::max(ret.first->second, kv.second);
    }
  }
}
}  // namespace onnxruntime
