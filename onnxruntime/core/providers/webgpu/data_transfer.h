// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/common/status.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {
namespace webgpu {

class BufferManager;
struct CommandRecordingState;

// Object layers ("outer" is the IDataTransfer object held by a manager):
// - Built-in Session and plugin-internal adapter:
//     onnxruntime::webgpu::DataTransfer
//       -> onnxruntime::webgpu::DataTransferImpl (impl_ member, stored by value).
// - Built-in Env, plugin Env, and plugin framework Session:
//     onnxruntime::plugin_ep::DataTransfer (core/framework/plugin_data_transfer.h)
//       -> onnxruntime::WebGpuDataTransferImpl (derived from OrtDataTransferImpl)
//       -> onnxruntime::webgpu::DataTransferImpl (local to each CopyTensors call).
//   The outer wrapper holds the C API object through an OrtDataTransferImpl&
//   and releases it via its Release callback. OrtDataTransferImpl is the C API
//   base/function table, not an additional, separately allocated wrapper.
//
// Creation and ownership:
// In both builds, InferenceSession::RegisterExecutionProvider() calls the EP's
// GetDataTransfer(); virtual dispatch selects the Session path described below.
//
// Built-in WebGPU:
// - Env: the internal WebGpuEpFactory::CreateDataTransfer() creates a
//   WebGpuDataTransferImpl via OrtWebGpuCreateDataTransfer().
// - Session: WebGpuExecutionProvider::GetDataTransfer() creates DataTransfer below,
//   bound to the EP's BufferManager and recording. Native kernels use this same
//   Session DataTransferManager rather than creating another transfer.
//
// Plugin WebGPU:
// - Env: library registration calls ep::Factory::CreateDataTransferImpl().
// - Framework Session: PluginExecutionProvider::GetDataTransfer() calls that same
//   factory callback. Both routes use OrtWebGpuCreateDataTransfer(), but create
//   separate WebGpuDataTransferImpl instances for their respective managers.
// - Inside the plugin: onnxruntime::ep::adapter::Ep also calls
//   WebGpuExecutionProvider::GetDataTransfer(), creating an EP-bound DataTransfer
//   for its own adapter DataTransferManager. Adapter kernels use this object,
//   bypassing the factory's C API wrapper.
//
// WebGpuDataTransferImpl, defined in webgpu_provider_factory.cc, uses per-call recording
// for streamless copies, selecting the context from the GPU tensor's memory-device ID. Session-level
// creation alone does not bind it to the requesting EP. In plugin builds, an explicit-stream copy uses
// ep/sync_stream.cc to construct a temporary webgpu::DataTransferImpl with the
// owning EP's BufferManager and recording, rather than the C API wrapper's
// call-local recording.
//
// DataTransferImpl is the shared raw-pointer copy implementation used by both
// DataTransfer (IDataTransfer subclass) and the C API wrapper.
class DataTransferImpl {
 public:
  DataTransferImpl(std::function<const BufferManager&()> buffer_manager_getter, CommandRecordingState& recording)
      : buffer_manager_getter_{std::move(buffer_manager_getter)}, recording_{recording} {}

  common::Status CopyTensor(void const* src_data,
                            bool src_is_gpu,
                            void* dst_data,
                            bool dst_is_gpu,
                            size_t bytes) const;

 private:
  std::function<const BufferManager&()> buffer_manager_getter_;
  CommandRecordingState& recording_;
};

class DataTransfer : public IDataTransfer {
 public:
  DataTransfer(std::function<const BufferManager&()> buffer_manager_getter,
               CommandRecordingState& recording,
               int context_id)
      : impl_{std::move(buffer_manager_getter), recording}, context_id_{context_id} {}
  ~DataTransfer() {};

  // Device-compatibility half of CanCopy, split out because it needs no BufferManager and so can
  // be tested without a live device.
  static bool IsSupportedDevicePair(const OrtDevice& src_device, const OrtDevice& dst_device, int context_id = 0);

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

 private:
  DataTransferImpl impl_;
  int context_id_;
};

}  // namespace webgpu
}  // namespace onnxruntime
