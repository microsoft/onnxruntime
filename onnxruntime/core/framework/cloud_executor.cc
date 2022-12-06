// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CLOUD
#include "core/framework/cloud_executor.h"
#include "core/framework/cloud_invoker.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

common::Status CloudExecutor::Execute(const SessionState& session_state, gsl::span<const int>,
                                      gsl::span<const OrtValue> feeds, gsl::span<const int>,
                                      std::vector<OrtValue>& fetches,
                                      const std::unordered_map<size_t, CustomAllocator>&,
                                      const logging::Logger&) {
  InlinedVector<std::string> input_names;
  for (const auto& input : session_state.GetGraphViewer().GetInputs()) {
    input_names.push_back(input->Name());
  }
  InlinedVector<std::string> output_names;
  for (const auto& output : session_state.GetGraphViewer().GetOutputs()) {
    output_names.push_back(output->Name());
  }
  auto invoker = CloudEndPointInvoker::CreateInvoker(session_state.GetSessionOption().config_options.configurations);
  if (invoker) {
    return invoker->Send(run_options_, input_names, feeds, output_names, fetches);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create cloud endpoint invoker");
  }
}
}  // namespace onnxruntime
#endif