// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "my_ep_data_transfer.h"

namespace onnxruntime {
bool MyEPDataTransfer::CanCopy(const OrtDevice& /*src_device*/, const OrtDevice& /*dst_device*/) const {
  return false;
}

common::Status MyEPDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int /*exec_queue_id*/) const {
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  size_t size = src.SizeInBytes();
  memcpy(dst_data, src_data, size);
  return Status::OK();
}

}  // namespace onnxruntime
