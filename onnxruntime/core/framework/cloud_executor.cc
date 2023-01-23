// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_AZURE
#include "core/framework/cloud_executor.h"
#include "core/framework/cloud_invoker.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

common::Status CloudExecutor::Execute(const SessionState& session_state, gsl::span<const int>,
                                      gsl::span<const OrtValue> feeds, gsl::span<const int>,
                                      std::vector<OrtValue>& fetches,
                                      const std::unordered_map<size_t, CustomAllocator>&,
                                      const logging::Logger&) {
  //collect input names
  const auto& inputs = session_state.GetGraphViewer().GetInputs();
  InlinedVector<std::string> input_names;
  input_names.reserve(inputs.size());
  for (const auto& input : inputs) {
    input_names.push_back(input->Name());
  }

  //collect output names
  const auto& outputs = session_state.GetGraphViewer().GetOutputs();
  InlinedVector<std::string> output_names;
  output_names.reserve(outputs.size());
  for (const auto& output : outputs) {
    output_names.push_back(output->Name());
  }

  //create invoker
  static const OrtDevice cpu_device;
  auto allocator = session_state.GetAllocator(cpu_device);
  std::unique_ptr<CloudEndPointInvoker> invoker;

  auto status = CloudEndPointInvoker::CreateInvoker(
      session_state.GetSessionOptions().config_options.configurations,
      allocator, invoker);

  if (invoker) {
    return invoker->Send(run_options_, input_names, feeds, output_names, fetches);
  } else {
    return status;
  }
}

}  // namespace onnxruntime
#endif