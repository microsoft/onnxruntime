// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"

#include "xpu_data_transfer.h"
#include "tvm_utils.h"


namespace onnxruntime {
namespace tvm {

XPUDataTransfer::XPUDataTransfer() {
}

XPUDataTransfer::~XPUDataTransfer() {
}

bool XPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::CPU) ||
  (src_device.Type() == OrtDevice::GPU || dst_device.Type() == OrtDevice::GPU);
}

common::Status XPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int _exec_queue_id) const {
  _exec_queue_id = _exec_queue_id + 1;
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  const auto src_device_type = src.Location().device.Type();
  const auto dst_device_type = dst.Location().device.Type();

  if ((src_device_type == OrtDevice::CPU) && (dst_device_type == OrtDevice::CPU)) {
    if (src_data == dst_data) {
      // no need copying as both pointers are referring to same piece of memory.
      return Status::OK();
    }
    memcpy(dst_data, src_data, bytes);
  } else {
    DLTensor tvm_src, tvm_dst;
    DLDataType dl_type{kDLInt, 8, 1};
    std::vector<int64_t> shape{int64_t(bytes)};
    // Construct source DLTensor
    tvm_src.device = GetDLDevice(static_cast<OrtMemoryInfoDeviceType>(src_device_type));
    tvm_src.dtype = dl_type;
    tvm_src.strides = nullptr;
    tvm_src.byte_offset = 0;
    tvm_src.data = const_cast<void*>(src_data);
    tvm_src.ndim = 1;
    tvm_src.shape = shape.data();
    // Construct destination DLTensor
    tvm_dst.device = GetDLDevice(static_cast<OrtMemoryInfoDeviceType>(dst_device_type));
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

DLDevice XPUDataTransfer::get_context(const OrtDevice& device) const
{
  return GetDLDevice(static_cast<OrtMemoryInfoDeviceType>(device.Type()));
}

bool TvmCPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::CPU;
}

common::Status TvmCPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int /*exec_queue_id*/) const {
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  if (src_data == dst_data) {
    // no need copying as both pointers are referring to same piece of memory.
    return Status::OK();
  }
  // Copying only happens between two same size tensors.
  ORT_ENFORCE(src.SizeInBytes() == dst.SizeInBytes());
  memcpy(dst_data, src_data, src.SizeInBytes());
  return Status::OK();
}

}   // namespace tvm
}   // namespace onnxruntime
