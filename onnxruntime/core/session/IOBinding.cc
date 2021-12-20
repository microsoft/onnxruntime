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
  ORT_ENFORCE(mapped_feed_names_.size() == feed_names_.size(), "Size mismatch(0):", mapped_feed_names_.size(), "!=", feed_names_.size());
  auto it = mapped_feed_names_.emplace(name, feed_names_.size());
  size_t index = it.first->second;
  if (it.second) {
    feed_names_.push_back(name);
    ORT_ENFORCE(index >= 0, "index cannot be negative, index=", index, " it.second=", it.second);
    ORT_ENFORCE(mapped_feed_names_.size() == feed_names_.size(), "Size mismatch(1):", mapped_feed_names_.size(), "!=", feed_names_.size(), " index=", index, " it.second=", it.second);
    OrtValue new_mlvalue;
    feeds_.push_back(new_mlvalue);
    // The inserted pointer points to name.c_str(), a pointer the class
    // does not own. It needs to be replaced by a pointer the class owns
    // pointing to the same string.
    ORT_ENFORCE(mapped_feed_names_.size() == feed_names_.size(), "Size mismatch(A):", mapped_feed_names_.size(), "!=", feed_names_.size(), " index=", index, " it.second=", it.second);
    ORT_ENFORCE(mapped_feed_names_.erase(name) == 1, "Key '", name, "' was not removed.");
    mapped_feed_names_[VariableNameWrapper(feed_names_[index])] = index;
    ORT_ENFORCE(mapped_feed_names_.size() == feed_names_.size(), "Size mismatch(2):", mapped_feed_names_.size(), "!=", feed_names_.size(), " index=", index, " it.second=", it.second);
  }
  ORT_ENFORCE(mapped_feed_names_.size() == feed_names_.size(), "Size mismatch(3):", mapped_feed_names_.size(), "!=", feed_names_.size(), " index=", index, " it.second=", it.second);

  if (ml_value.IsTensor() || ml_value.IsSparseTensor()) {
    OrtValue new_mlvalue;
    // Do not replace new_mlvalue by feeds_[index] in the following line.
    // It may copy the data instead of copying the pointer.
    ORT_RETURN_IF_ERROR(utils::CopyOneInputAcrossDevices(session_state_, name, ml_value, new_mlvalue));
    feeds_[index] = new_mlvalue;
  } else {
    feeds_[index] = ml_value;
  }

  return Status::OK();
}

void IOBinding::ClearInputs() {
  mapped_feed_names_.clear();
  feed_names_.clear();
  feeds_.clear();
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

common::Status IOBinding::BindOutput(const std::string& name, const OrtValue& ml_value) {
  // device value is ignored when ml_value is pre-allocated
  return BindOutputImpl(name, ml_value, {});
}

common::Status IOBinding::BindOutput(const std::string& name, OrtDevice device) {
  return BindOutputImpl(name, {}, device);
}

common::Status IOBinding::BindOutputImpl(const std::string& name, const OrtValue& ml_value, OrtDevice device) {
  ORT_ENFORCE(mapped_output_names_.size() == output_names_.size(), "Size mismatch.", mapped_output_names_.size(), "!=", output_names_.size());
  auto it = mapped_output_names_.emplace(name, output_names_.size());
  size_t index = it.first->second;
  if (it.second) {
    output_names_.push_back(name);
    outputs_.push_back(ml_value);
    outputs_device_info_.push_back(device);
    // The inserted pointer points to name.c_str(), a pointer the class
    // does not own. It needs to be replaced by a pointer the class owns
    // pointing to the same string.
    mapped_output_names_.extract(it.first);
    mapped_output_names_[VariableNameWrapper(output_names_[index])] = index;
  } else {
    outputs_[index] = ml_value;
    outputs_device_info_[index] = device;
  }

  return Status::OK();
}

void IOBinding::ClearOutputs() {
  mapped_output_names_.clear();
  output_names_.clear();
  outputs_.clear();
  outputs_device_info_.clear();
}

const std::vector<std::string>& IOBinding::GetOutputNames() const { return output_names_; }

const std::vector<OrtValue>& IOBinding::GetOutputs() const { return outputs_; }

std::vector<OrtValue>& IOBinding::GetOutputs() { return outputs_; }

const std::vector<OrtDevice>& IOBinding::GetOutputsDeviceInfo() const {
  return outputs_device_info_;
}

const std::vector<std::string>& IOBinding::GetInputNames() const { return feed_names_; }

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
