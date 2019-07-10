// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {
using namespace common;

Status DataTransferManager::RegisterDataTransfer(std::unique_ptr<IDataTransfer> data_transfer) {
  if (nullptr == data_transfer) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "data_transfer registered is nullptr.");
  }
  datatransfers_.push_back(std::move(data_transfer));
  return Status::OK();
}

const IDataTransfer* DataTransferManager::GetDataTransfer(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  for (auto& data_transfer : datatransfers_) {
    if (!data_transfer->CanCopy(src_device, dst_device)) {
      continue;
    }

    return data_transfer.get();
  }
  return nullptr;
}


Status DataTransferManager::CopyTensor(const Tensor& src, Tensor& dst) const {
  return CopyTensor(src, dst, 0);
}

Status DataTransferManager::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  if (src.Shape().Size() != dst.Shape().Size()) {
    return Status(ONNXRUNTIME, FAIL, "Tensor size mismatch");
  }

  for (auto& data_transfer : datatransfers_) {
    if (!data_transfer->CanCopy(src.Location().device, dst.Location().device)) {
      continue;
    }

    return data_transfer->CopyTensor(src, dst, exec_queue_id);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME,
                         FAIL,
                         "There's no data transfer registered for copying tensors from ",
                         src.Location().device.ToString(),
                         " to ",
                         dst.Location().device.ToString());
}

}  // namespace onnxruntime
