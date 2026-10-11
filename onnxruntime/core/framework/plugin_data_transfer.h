// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <utility>

#include "core/framework/data_transfer.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/ort_value.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/abi_devices.h"

namespace onnxruntime {
namespace plugin_ep {

struct OrtDataTransferImplDeleter {
  void operator()(OrtDataTransferImpl* data_transfer) const noexcept {
    if (data_transfer != nullptr && data_transfer->Release != nullptr) {
      data_transfer->Release(data_transfer);
    }
  }
};

using OrtDataTransferImplUniquePtr = std::unique_ptr<OrtDataTransferImpl, OrtDataTransferImplDeleter>;

/// <summary>
/// Class to implement IDataTransfer for plugin execution providers.
/// It uses the OrtDataTransferImpl from the plugin EP factory to implement the data transfer functionality.
/// </summary>
class DataTransfer : public IDataTransfer {
 public:
  explicit DataTransfer(OrtDataTransferImplUniquePtr impl)
      : impl_{std::move(impl)} {
    ORT_ENFORCE(impl_ != nullptr, "OrtDataTransferImpl must not be null.");
    ORT_ENFORCE(impl_->Release != nullptr, "OrtDataTransferImpl must provide a Release function.");
    ORT_ENFORCE(impl_->CanCopy != nullptr, "OrtDataTransferImpl must provide a CanCopy function.");
    ORT_ENFORCE(impl_->CopyTensors != nullptr, "OrtDataTransferImpl must provide a CopyTensors function.");
  }

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override {
    const OrtMemoryDevice* src_memory_device = static_cast<const OrtMemoryDevice*>(&src_device);
    const OrtMemoryDevice* dst_memory_device = static_cast<const OrtMemoryDevice*>(&dst_device);

    return impl_->CanCopy(impl_.get(), src_memory_device, dst_memory_device);
  }

  Status CopyTensor(const Tensor& src, Tensor& dst) const override {
    return CopyTensorImpl(src, dst, nullptr);
  }

  Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const override {
    return CopyTensorImpl(src, dst, &stream);
  }

  Status CopyTensors(const std::vector<SrcDstPair>& src_dst_pairs) const override;

 private:
  Status CopyTensorImpl(const Tensor& src, Tensor& dst, onnxruntime::Stream* stream = nullptr) const;

  OrtDataTransferImplUniquePtr impl_;
};
}  // namespace plugin_ep
}  // namespace onnxruntime
