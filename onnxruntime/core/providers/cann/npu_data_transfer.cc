// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/npu_data_transfer.h"
#include "core/providers/cann/cann_call.h"

namespace onnxruntime {
NPUDataTransfer::NPUDataTransfer(aclrtStream stream, bool do_copy_in_default_stream) {
  do_copy_in_default_stream_ = do_copy_in_default_stream;
  streams_[kCannStreamDefault] = stream;
  if (do_copy_in_default_stream) {
    streams_[kCannStreamCopyIn] = stream;
    streams_[kCannStreamCopyOut] = stream;
  } else {
    CANN_CALL_THROW(aclrtCreateStream(&streams_[kCannStreamCopyIn]));
    CANN_CALL_THROW(aclrtCreateStream(&streams_[kCannStreamCopyOut]));
  }
}

NPUDataTransfer::~NPUDataTransfer() {
  if (!do_copy_in_default_stream_ && streams_[kCannStreamCopyIn] != nullptr) {
    CANN_CALL_THROW(aclrtDestroyStream(streams_[kCannStreamCopyIn]));
  }
  if (!do_copy_in_default_stream_ && streams_[kCannStreamCopyOut] != nullptr) {
    CANN_CALL_THROW(aclrtDestroyStream(streams_[kCannStreamCopyOut]));
  }
}

bool NPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::NPU || dst_device.Type() == OrtDevice::NPU;
}

common::Status NPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::NPU) {
    if (src_device.Type() == OrtDevice::CPU && src_device.MemType() == OrtDevice::MemType::CANN_PINNED) {
      CANN_CALL_THROW(aclrtMemcpyAsync(dst_data, bytes, src_data, bytes,
                                       ACL_MEMCPY_HOST_TO_DEVICE, GetStream(exec_queue_id)));
    } else if (src_device.Type() == OrtDevice::NPU) {
      if (dst_data != src_data) {
        CANN_CALL_THROW(aclrtMemcpyAsync(dst_data, bytes, src_data, bytes,
                                         ACL_MEMCPY_DEVICE_TO_DEVICE, GetStream(kCannStreamDefault)));
      }
    } else {
      CANN_CALL_THROW(aclrtMemcpyAsync(dst_data, bytes, src_data, bytes,
                                       ACL_MEMCPY_HOST_TO_DEVICE, GetStream(kCannStreamDefault)));
      CANN_CALL_THROW(aclrtSynchronizeStream(GetStream(kCannStreamDefault)));
    }
  } else {
    if (dst_device.Type() == OrtDevice::CPU && dst_device.MemType() == OrtDevice::MemType::CANN_PINNED) {
      CANN_CALL_THROW(aclrtMemcpyAsync(dst_data, bytes, src_data, bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST, GetStream(exec_queue_id)));
    } else {
      CANN_CALL_THROW(aclrtMemcpyAsync(dst_data, bytes, src_data, bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST, GetStream(kCannStreamDefault)));
      CANN_CALL_THROW(aclrtSynchronizeStream(GetStream(kCannStreamDefault)));
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
