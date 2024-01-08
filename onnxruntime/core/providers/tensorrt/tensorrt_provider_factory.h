// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include "core/framework/arena_extend_strategy.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"

namespace onnxruntime {
struct ProviderInfo_TensorRT {
  virtual OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) = 0;
  virtual OrtStatus* UpdateProviderOptions(void* provider_options, const ProviderOptions& options, bool string_copy) = 0;
  virtual OrtStatus* GetTensorRTCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list, const std::string extra_plugin_lib_paths) = 0;
  virtual OrtStatus* ReleaseCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list) = 0;

  virtual std::unique_ptr<onnxruntime::IAllocator> CreateCUDAAllocator(int16_t device_id, const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IAllocator> CreateCUDAPinnedAllocator(const char* name) = 0;
  virtual std::unique_ptr<onnxruntime::IDataTransfer> CreateGPUDataTransfer() = 0; protected:
  ~ProviderInfo_TensorRT() = default;  // Can only be destroyed through a subclass instance
};
}  // namespace onnxruntime
