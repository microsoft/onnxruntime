// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)

#include <memory>

#include "core/framework/allocator.h"
#include "core/framework/external_data_loader.h"
#include "core/providers/webgpu/webgpu_provider_options.h"

struct ID3D12Device;

namespace onnxruntime {
namespace webgpu {

class WebGpuContext;

common::Status CheckDirectStorageExternalWeightsSupport(WebGpuContext& context);

common::Status ResolveWeightLoadAccelerationMode(
    WeightLoadAccelerationMode mode,
    const common::Status& support_status,
    bool& enabled);

// Shared by the loader and allocator so imported resources outlive initializer loading.
class DirectStorageInitializerState {
 public:
  ~DirectStorageInitializerState();

 private:
  struct Impl;

  DirectStorageInitializerState();

  std::unique_ptr<Impl> impl_;

  friend AllocatorPtr CreateDirectStorageWebGpuAllocator(
      WebGpuContext& context, std::shared_ptr<DirectStorageInitializerState>& out_state);
  friend class DirectStorageExternalDataLoader;
  friend class DirectStorageWebGpuAllocator;
};

AllocatorPtr CreateDirectStorageWebGpuAllocator(
    WebGpuContext& context, std::shared_ptr<DirectStorageInitializerState>& out_state);

class DirectStorageExternalDataLoader final : public IExternalDataLoader {
 public:
  DirectStorageExternalDataLoader(
      WebGpuContext& context,
      std::shared_ptr<DirectStorageInitializerState> state,
      WeightLoadAccelerationMode mode);
  ~DirectStorageExternalDataLoader() override;

  bool CanLoad(const OrtMemoryInfo& target_memory_info) const override;
  bool SupportsDataType(int32_t tensor_data_type) const override;
  bool CreatesTensorForDevice(const OrtDevice& target_device) const override;
  bool SupportsPreload() const override;
  common::Status BeginPreload() const override;
  common::Status PreloadTensor(const Env& env,
                               const std::filesystem::path& data_file_path,
                               std::string_view tensor_name,
                               FileOffsetType data_offset,
                               SafeInt<size_t> data_length) const override;
  common::Status FinalizePreload(const std::function<bool()>& is_cancelled) const override;
  common::Status BeginLoad() const override;
  common::Status PrepareTensor(const Env& env,
                               const std::filesystem::path& data_file_path,
                               std::string_view tensor_name,
                               FileOffsetType data_offset,
                               SafeInt<size_t> data_length) const override;
  common::Status FinalizeLoad(const std::function<bool()>& is_cancelled) const override;
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

#endif  // defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
