// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/IOBinding.h"
#include "core/common/logging/logging.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel.h"
#include "core/framework/utils.h"

namespace onnxruntime {
IOBinding::IOBinding(const SessionState& session_state) : session_state_(session_state) {
}

common::Status IOBinding::BindInput(const std::string& name, const OrtValue& ml_value) {
  if (!ml_value.IsTensor()) {
    feed_names_.push_back(name);
    feeds_.push_back(ml_value);
    return Status::OK();
  }

  OrtValue new_mlvalue;
  ORT_RETURN_IF_ERROR(utils::CopyOneInputAcrossDevices(session_state_, name, ml_value, new_mlvalue));
  feed_names_.push_back(name);
  feeds_.push_back(new_mlvalue);

  return Status::OK();
}

static common::Status SyncProviders(const SessionState::NameNodeInfoMapType& node_info_map,
                                    const SessionState& session_state) {
  std::set<std::string> providers;
  for (auto& pair : node_info_map) {
    for (auto& node_info : pair.second) {
      if (node_info.p_node->GetExecutionProviderType() != onnxruntime::kCpuExecutionProvider) {
        providers.insert(node_info.p_node->GetExecutionProviderType());
      }
    }
  }
  for (auto& provider_type : providers) {
    auto* p_provider = session_state.GetExecutionProviders().Get(provider_type);
    if (!p_provider) {
      continue;
    }

    ORT_RETURN_IF_ERROR(p_provider->Sync());
  }
  return Status::OK();
}

common::Status IOBinding::SynchronizeInputs() {
  ORT_RETURN_IF_ERROR(SyncProviders(session_state_.GetInputNodeInfoMap(), session_state_));
  return Status::OK();
}

common::Status IOBinding::SynchronizeOutputs() {
  ORT_RETURN_IF_ERROR(SyncProviders(session_state_.GetOutputNodeInfoMap(), session_state_));
  return Status::OK();
}

static std::pair<bool, size_t> Contains(const std::vector<std::string>& output_names, const std::string& oname) {
  auto it = std::find(std::begin(output_names), std::end(output_names), oname);
  if (it == std::end(output_names)) {
    return {false, 0};
  }
  return {true, it - std::begin(output_names)};
}

common::Status IOBinding::BindOutput(const std::string& name, const OrtValue& ml_value) {
  auto rc = Contains(output_names_, name);
  if (rc.first) {
    outputs_[rc.second] = ml_value;
    return Status::OK();
  }

  output_names_.push_back(name);
  outputs_.push_back(ml_value);
  return Status::OK();
}

const std::vector<std::string>& IOBinding::GetOutputNames() const {
  return output_names_;
}

std::vector<OrtValue>& IOBinding::GetOutputs() { return outputs_; }

const std::vector<std::string>& IOBinding::GetInputNames() const {
  return feed_names_;
}

const std::vector<OrtValue>& IOBinding::GetInputs() const { return feeds_; }

AllocatorPtr IOBinding::GetCPUAllocator(int id, onnxruntime::ProviderType provider_type) const {
  auto& exec_providers = session_state_.GetExecutionProviders();
  auto* p_provider = exec_providers.Get(provider_type);
  ORT_ENFORCE(p_provider);
  auto allocator = p_provider->GetAllocator(id, OrtMemTypeCPU);

  // if the provider does not implement CPU allocator, fall back to CPU
  if (allocator)
    return allocator;

  auto* cpu_provider = exec_providers.Get(onnxruntime::kCpuExecutionProvider);
  return cpu_provider->GetAllocator(0, OrtMemTypeDefault);
}

}  // namespace onnxruntime
