// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cloud/cloud_executor.h"
#include "core/providers/cloud/cloud_execution_provider.h"
#include "core/providers/cloud/endpoint_invoker.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

common::Status CloudExecutor::Execute(const SessionState& session_state, gsl::span<const int> /*feed_mlvalue_idxs*/,
                                      gsl::span<const OrtValue> feeds, gsl::span<const int> /*fetch_mlvalue_idxs*/,
                                      std::vector<OrtValue>& fetches,
                                      const std::unordered_map<size_t, CustomAllocator>& /*fetch_allocators*/,
                                      const logging::Logger& /*logger*/) {

  const CloudExecutionProvider* cloud_ep = reinterpret_cast<const CloudExecutionProvider*>(session_state.GetExecutionProviders().Get(onnxruntime::kCloudExecutionProvider));
  ORT_ENFORCE(cloud_ep, "unable to find the cloud ep for execution");
  //cloud::EndPointInvoker* invoker = cloud_ep->GetInvoker();
  auto invoker = cloud::EndPointInvoker::CreateInvoker(cloud_ep->GetConfig());
  if (invoker) {
    if (invoker->GetStaus().IsOK()) {
      invoker->Send(feeds, fetches);
      return invoker->GetStaus();
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create endpoint invoker");
  }
  const auto& invoker_status = invoker->GetStaus();
  if (!invoker_status.IsOK()) return invoker_status;
  return onnxruntime::Status::OK();
}

}  // namespace onnxruntime