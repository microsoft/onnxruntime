// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/npu_data_transfer.h"

namespace onnxruntime {
NPUDataTransfer::NPUDataTransfer() {}

NPUDataTransfer::~NPUDataTransfer() {}

bool NPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::NPU || dst_device.Type() == OrtDevice::NPU;
}

common::Status NPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  // for the sync version of memcpy, launch to cann default stream
  if (dst_device.Type() == OrtDevice::NPU) {
    if (src_device.Type() == OrtDevice::NPU) {
      // Copy only if the two addresses are different.
      if (dst_data != src_data) {
        CANN_RETURN_IF_ERROR(aclrtMemcpy(dst_data,
                                         bytes,
                                         src_data,
                                         bytes,
                                         ACL_MEMCPY_DEVICE_TO_DEVICE));
        CANN_RETURN_IF_ERROR(aclrtSynchronizeStream(nullptr));
      }
    } else {
      // copy from other CPU memory to NPU, this is blocking
      CANN_RETURN_IF_ERROR(aclrtMemcpy(dst_data,
                                       bytes,
                                       src_data,
                                       bytes,
                                       ACL_MEMCPY_HOST_TO_DEVICE));
      CANN_RETURN_IF_ERROR(aclrtSynchronizeStream(nullptr));
    }
  } else if (src_device.Type() == OrtDevice::NPU) {
    // copying from NPU to CPU memory, this is blocking
    CANN_RETURN_IF_ERROR(aclrtMemcpy(dst_data,
                                     bytes,
                                     src_data,
                                     bytes,
                                     ACL_MEMCPY_DEVICE_TO_HOST));
    CANN_RETURN_IF_ERROR(aclrtSynchronizeStream(nullptr));
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

common::Status NPUDataTransfer::CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::NPU) {
    if (src_device.Type() == OrtDevice::CPU) {
      // copy from pinned memory to NPU, this is non-blocking
      CANN_RETURN_IF_ERROR(aclrtMemcpyAsync(dst_data,
                                            bytes,
                                            src_data,
                                            bytes,
                                            ACL_MEMCPY_HOST_TO_DEVICE,
                                            static_cast<aclrtStream>(stream.GetHandle())));
    } else if (src_device.Type() == OrtDevice::NPU) {
      // copying between NPU, this is non-blocking
      if (dst_data != src_data) {
        CANN_RETURN_IF_ERROR(aclrtMemcpyAsync(dst_data,
                                              bytes,
                                              src_data,
                                              bytes,
                                              ACL_MEMCPY_DEVICE_TO_DEVICE,
                                              static_cast<aclrtStream>(stream.GetHandle())));
      }
    }
  } else if (src_device.Type() == OrtDevice::NPU) {
    if (dst_device.Type() == OrtDevice::CPU) {
      // copying from NPU to pinned memory, this is non-blocking
      CANN_RETURN_IF_ERROR(aclrtMemcpyAsync(dst_data,
                                            bytes,
                                            src_data,
                                            bytes,
                                            ACL_MEMCPY_DEVICE_TO_HOST,
                                            static_cast<aclrtStream>(stream.GetHandle())));
    }
  } else {
    if (src_device.MemType() == OrtDevice::MemType::CANN_PINNED) {
      // sync the stream first to make sure the data arrived
      CANN_RETURN_IF_ERROR(aclrtSynchronizeStream(static_cast<aclrtStream>(stream.GetHandle())));
    }
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

}  // namespace onnxruntime
