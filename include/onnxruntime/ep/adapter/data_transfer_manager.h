// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include "core/common/status.h"
#include "core/common/common.h"
#include "core/framework/data_transfer.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace ep {
namespace adapter {

/// <summary>
/// An adapter class partially implementing the interface of `onnxruntime::DataTransferManager`.
/// </summary>
struct DataTransferManager {
  explicit DataTransferManager(std::unique_ptr<IDataTransfer> impl) : impl_{std::move(impl)} {}

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const {
    if (src.Shape().Size() != dst.Shape().Size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME,
                             FAIL,
                             "Tensor size mismatch: source tensor size is ",
                             src.Shape().Size(),
                             ", destination tensor size is ",
                             dst.Shape().Size());
    }

    if (impl_->CanCopy(src.Location().device, dst.Location().device)) {
      return impl_->CopyTensor(src, dst);
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME,
                           FAIL,
                           "There's no data transfer registered for copying tensors from ",
                           src.Location().device.ToString(),
                           " to ",
                           dst.Location().device.ToString());
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DataTransferManager);
  std::unique_ptr<IDataTransfer> impl_;
};

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
