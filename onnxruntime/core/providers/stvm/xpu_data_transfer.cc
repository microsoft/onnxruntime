// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"

#include "xpu_data_transfer.h"
#include "stvm_utils.h"

namespace onnxruntime {
GPUDataTransfer::GPUDataTransfer() {
}

GPUDataTransfer::~GPUDataTransfer() {
}

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
    return (src_device.Type() == OrtDevice::GPU || dst_device.Type() == OrtDevice::GPU);
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int _exec_queue_id) const {
  _exec_queue_id = _exec_queue_id + 1;
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  const OrtDevice& src_device = src.Location().device;
  const OrtDevice& dst_device = dst.Location().device;

  if ((src_device.Type() == OrtDevice::CPU) && (dst_device.Type() == OrtDevice::CPU)) {
      memcpy(dst_data, src_data, bytes);
  } else {
    DLTensor tvm_src, tvm_dst;
    DLDataType dl_type{kDLInt, 8, 1};
    std::vector<int64_t> shape{int64_t(bytes)};
    // Construct source DLTensor
    tvm_src.device = GetDLDevice(src_device);
    tvm_src.dtype = dl_type;
    tvm_src.strides = nullptr;
    tvm_src.byte_offset = 0;
    tvm_src.data = const_cast<void*>(src_data);
    tvm_src.ndim = 1;
    tvm_src.shape = shape.data();
    // Construct destination DLTensor
    tvm_dst.device = GetDLDevice(dst_device);
    tvm_dst.dtype = dl_type;
    tvm_dst.strides = nullptr;
    tvm_dst.byte_offset = 0;
    tvm_dst.data = dst_data;
    tvm_dst.ndim = 1;
    tvm_dst.shape = shape.data();
    // Copying from src to dst
    TVMDeviceCopyDataFromTo(&tvm_src, &tvm_dst, nullptr);
  }
  return Status::OK();
}

DLDevice GPUDataTransfer::get_context(const OrtDevice& device) const
{
  DLDevice context;
  switch (device.Type()) {
  case OrtDevice::CPU:
      context = {kDLCPU, 0};
      break;
  case OrtDevice::GPU:
      context = {kDLVulkan, 0};
      break;
  default:
      ORT_NOT_IMPLEMENTED("GPUDataTransfer get_context");
      break;
  }
  return context;
}

}  // namespace onnxruntime
