// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "cuda_pch.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {

class GPUDataTransfer : public IDataTransfer {
 public:
  GPUDataTransfer();
  ~GPUDataTransfer() override;

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  // Dumpen MSVC warning about not fully overriding
  using IDataTransfer::CopyTensor;
  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;
  common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const override;

 private:
  struct PinnedStagingState;

  common::Status CopyHostToDeviceWithPinnedStaging(const void* src_data, void* dst_data,
                                                   size_t bytes, int device_id) const;
  common::Status EnsurePinnedStaging(int device_id, PinnedStagingState*& state) const;
  static void ReleasePinnedStaging(PinnedStagingState& state) noexcept;
  void ReleaseAllPinnedStaging() const noexcept;

  mutable std::mutex pinned_staging_mutex_;
  mutable std::unordered_map<int, std::unique_ptr<PinnedStagingState>> pinned_staging_by_device_;
};

}  // namespace onnxruntime
