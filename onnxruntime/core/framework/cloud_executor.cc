// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/cloud_executor.h"
#include "core/framework/cloud_invoker.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

common::Status CloudExecutor::Execute(const SessionState& session_state, gsl::span<const int>,
                                      gsl::span<const OrtValue> feeds, gsl::span<const int>,
                                      std::vector<OrtValue>& fetches,
                                      const std::unordered_map<size_t, CustomAllocator>&,
                                      const logging::Logger&) {

  CloudEndPointInvoker* invoker = session_state.GetCloudInvoker();
  if (invoker) {
    if (invoker->GetStaus().IsOK()) {
      invoker->Send(feeds, fetches);
      return invoker->GetStaus();
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get cloud endpoint invoker");
  }
  const auto& invoker_status = invoker->GetStaus();
  if (!invoker_status.IsOK()) return invoker_status;
  return onnxruntime::Status::OK();
}

}  // namespace onnxruntime