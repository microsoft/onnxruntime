// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)

#include <functional>
#include <memory>
#include <vector>

#include "core/framework/allocator.h"
#include "core/framework/external_data_loader.h"
#include "core/providers/webgpu/webgpu_provider_options.h"

struct ID3D12Device;

namespace onnxruntime {
namespace webgpu {

class WebGpuContext;
struct CommandRecordingState;

common::Status CheckD3D12AcceleratedExternalWeightsSupport(WebGpuContext& context);

common::Status ResolveWeightLoadAccelerationMode(
    WeightLoadAccelerationMode mode,
    const common::Status& support_status,
    bool& enabled);

// Shared by the loader and allocator so imported resources outlive initializer loading.
class D3D12AcceleratedInitializerState {
 public:
  ~D3D12AcceleratedInitializerState();

 private:
  struct Impl;

  D3D12AcceleratedInitializerState();

  std::unique_ptr<Impl> impl_;

  friend AllocatorPtr CreateD3D12AcceleratedWebGpuAllocator(
      WebGpuContext& context,
      std::function<CommandRecordingState&()> recording_getter,
      std::shared_ptr<D3D12AcceleratedInitializerState>& out_state);
  friend class D3D12AcceleratedExternalDataLoader;
  friend class D3D12AcceleratedWebGpuAllocator;
};

AllocatorPtr CreateD3D12AcceleratedWebGpuAllocator(
    WebGpuContext& context,
    std::function<CommandRecordingState&()> recording_getter,
    std::shared_ptr<D3D12AcceleratedInitializerState>& out_state);

class D3D12AcceleratedExternalDataLoader final : public IExternalDataLoader {
 public:
  D3D12AcceleratedExternalDataLoader(
      WebGpuContext& context,
      std::shared_ptr<D3D12AcceleratedInitializerState> state,
      WeightLoadAccelerationMode mode);
  ~D3D12AcceleratedExternalDataLoader() override;

  bool CanLoad(const OrtMemoryInfo& target_memory_info) const override;
  bool SupportsDataType(int32_t tensor_data_type) const override;
  bool CreatesTensorForDevice(const OrtDevice& target_device) const override;
  common::Status BeginLoad() const override;
  common::Status PrepareTensor(const Env& env,
                               const std::filesystem::path& data_file_path,
                               std::string_view tensor_name,
                               FileOffsetType data_offset,
                               SafeInt<size_t> data_length) const override;
  common::Status FinalizeLoad(const std::function<bool()>& is_canceled) const override;
  void AbortLoad() const noexcept override;
  common::Status LoadTensor(const Env& env,
                            const std::filesystem::path& data_file_path,
                            std::string_view tensor_name,
                            FileOffsetType data_offset,
                            SafeInt<size_t> data_length,
                            const std::shared_ptr<IAllocator>& allocator,
                            Tensor& tensor) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // defined(_WIN32) && defined(ENABLE_D3D12_FILE_LOADING)
