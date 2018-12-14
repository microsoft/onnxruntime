// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/IOBinding.h"
#include "core/common/logging/logging.h"
#include "core/framework/session_state.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
IOBinding::IOBinding(const SessionState& session_state) : session_state_(session_state) {
}

common::Status IOBinding::BindInput(const std::string& name, const MLValue& ml_value) {
  if (!ml_value.IsTensor()) {
    feeds_[name] = ml_value;
    return Status::OK();
  }

  MLValue new_mlvalue;
  ONNXRUNTIME_RETURN_IF_ERROR(CopyOneInputAcrossDevices(session_state_, name, ml_value, new_mlvalue));
  feeds_[name] = new_mlvalue;
  return Status::OK();
}

static common::Status AllocateHelper(const SessionState& session_state,
                                     int id, onnxruntime::ProviderType provider_type,
                                     const MLValue& fetched_mlvalue,
                                     MLValue& output_mlvalue) {
  auto* p_provider = session_state.GetExecutionProviders().Get(provider_type);
  ONNXRUNTIME_ENFORCE(p_provider);
  auto allocator = p_provider->GetAllocator(id, OrtMemTypeDefault);
  ONNXRUNTIME_ENFORCE(allocator != nullptr);
  auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();
  void* buffer = allocator->Alloc(fetched_tensor.Size());
  ONNXRUNTIME_ENFORCE(buffer);
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(fetched_tensor.DataType(),
                                                              fetched_tensor.Shape(),
                                                              buffer,
                                                              allocator->Info(),
                                                              allocator);
  output_mlvalue.Init(p_tensor.release(),
                      DataTypeImpl::GetType<Tensor>(),
                      DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

// TODO should we handle the case of one input name feeding 2 nodes placed on different
// devices.
common::Status IOBinding::CopyOneInputAcrossDevices(const SessionState& session_state,
                                                    const std::string& input_name,
                                                    const MLValue& orig_mlvalue,
                                                    MLValue& new_mlvalue) {
  //TODO: make it configurable
  const int target_device_id = 0;
  std::vector<SessionState::NodeInfo> node_info_vec;
  ONNXRUNTIME_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));

  for (auto& node_info : node_info_vec) {
    size_t index = node_info.index;
    auto& node = *node_info.p_node;
    const KernelCreateInfo* kci = node_info.kci;
    const auto* node_input_mem_types = (kci != nullptr) ? &kci->kernel_def->InputMemoryType() : nullptr;

    // node may declare input_mem_type to be on CPU explicitly
    bool node_input_on_cpu = node_input_mem_types && MemTypeOnCpuExplicitly(*node_input_mem_types, index);
    auto& required_provider_type = node_input_on_cpu ? onnxruntime::kCpuExecutionProvider : node.GetExecutionProviderType();
    if (!orig_mlvalue.IsTensor()) {
      // copying not supported for non-tensor types
      new_mlvalue = orig_mlvalue;
      return Status::OK();
    }
    auto& input_tensor = orig_mlvalue.Get<Tensor>();
    auto& input_tensor_loc = input_tensor.Location();
    auto& exec_providers = session_state.GetExecutionProviders();

    auto* p_input_provider = exec_providers.Get(input_tensor_loc);
    if (!p_input_provider) {
      p_input_provider = exec_providers.Get(onnxruntime::kCpuExecutionProvider);
      ONNXRUNTIME_ENFORCE(p_input_provider);
    }

    auto input_provider_type = p_input_provider->Type();
    if (input_provider_type == required_provider_type && input_tensor_loc.mem_type == OrtMemTypeDefault) {
      new_mlvalue = orig_mlvalue;
      return Status::OK();
    }

    //If node require input on cpu and input tensor is allocated with pinned memory allocator, don't do copy
    if (node_input_on_cpu && (input_tensor_loc.mem_type == OrtMemTypeCPU || input_tensor_loc.mem_type == OrtMemTypeCPUOutput)) {
      new_mlvalue = orig_mlvalue;
      return Status::OK();
    }

    auto* node_provider = exec_providers.Get(required_provider_type);
    ONNXRUNTIME_ENFORCE(node_provider);
    ONNXRUNTIME_RETURN_IF_ERROR(AllocateHelper(session_state, target_device_id, required_provider_type, orig_mlvalue, new_mlvalue));
    auto* new_tensor = new_mlvalue.GetMutable<Tensor>();
    auto* node_exec_provider = exec_providers.Get(required_provider_type);
    ONNXRUNTIME_ENFORCE(node_exec_provider);

    // our CPU exec provider doesn't support copy from GPU->CPU
    if (required_provider_type != onnxruntime::kCpuExecutionProvider) {
      ONNXRUNTIME_RETURN_IF_ERROR(node_exec_provider->CopyTensor(input_tensor, *new_tensor));
    } else {
      ONNXRUNTIME_RETURN_IF_ERROR(p_input_provider->CopyTensor(input_tensor, *new_tensor));
    }
  }

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

    ONNXRUNTIME_RETURN_IF_ERROR(p_provider->Sync());
  }
  return Status::OK();
}

common::Status IOBinding::SynchronizeInputs() {
  ONNXRUNTIME_RETURN_IF_ERROR(SyncProviders(session_state_.GetInputNodeInfoMap(), session_state_));
  return Status::OK();
}

common::Status IOBinding::SynchronizeOutputs() {
  ONNXRUNTIME_RETURN_IF_ERROR(SyncProviders(session_state_.GetOutputNodeInfoMap(), session_state_));
  return Status::OK();
}

static std::pair<bool, size_t> Contains(const std::vector<std::string>& output_names, const std::string& oname) {
  auto it = std::find(std::begin(output_names), std::end(output_names), oname);
  if (it == std::end(output_names)) {
    return {false, 0};
  }
  return {true, it - std::begin(output_names)};
}

common::Status IOBinding::BindOutput(const std::string& name, const MLValue& ml_value) {
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

std::vector<MLValue>& IOBinding::GetOutputs() {
  return outputs_;
}

const std::unordered_map<std::string, MLValue>& IOBinding::GetInputs() const {
  return feeds_;
}

AllocatorPtr IOBinding::GetCPUAllocator(int id, onnxruntime::ProviderType provider_type) const {
  auto& exec_providers = session_state_.GetExecutionProviders();
  auto* p_provider = exec_providers.Get(provider_type);
  ONNXRUNTIME_ENFORCE(p_provider);
  auto allocator = p_provider->GetAllocator(id, OrtMemTypeCPU);

  // if the provider does not implement CPU allocator, fall back to CPU
  if (allocator)
    return allocator;

  auto* cpu_provider = exec_providers.Get(onnxruntime::kCpuExecutionProvider);
  return cpu_provider->GetAllocator(0, OrtMemTypeDefault);
}

}  // namespace onnxruntime
