// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_data_transfer.h"

#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"

#include "opencl_onnxruntime_utils.h"

namespace onnxruntime {
namespace opencl {

OpenCLDataTransfer::OpenCLDataTransfer(cl::CommandQueue cmd_queue)
    : cmd_queue_{std::move(cmd_queue)} {}

OpenCLDataTransfer::~OpenCLDataTransfer() {}

bool OpenCLDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU);
}

common::Status OpenCLDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  ORT_ENFORCE(exec_queue_id == 0);

  const auto& src_device = src.Location().device;
  const auto& dst_device = dst.Location().device;

  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_ENFORCE(src.ByteOffset() == 0);
    auto dst_buffer = CL_BUFFER_FROM_TENSOR(dst);
    std::cerr << "OpenCL copy host " << src.DataRaw() << " --> device cl::Buffer(" << dst_buffer() << ")\n";
    OPENCL_CHECK_ERROR(cmd_queue_.enqueueWriteBuffer(dst_buffer, CL_TRUE, 0, src.SizeInBytes(), src.DataRaw()));
    return Status::OK();
  }

  if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) {
    ORT_ENFORCE(dst.ByteOffset() == 0);
    auto src_buffer = CL_BUFFER_FROM_TENSOR(src);
    std::cerr << "OpenCL copy host " << src.DataRaw() << " <-- device cl::Buffer(" << src_buffer() << ")\n";
    OPENCL_CHECK_ERROR(cmd_queue_.enqueueReadBuffer(src_buffer, CL_TRUE, 0, dst.SizeInBytes(), dst.MutableDataRaw()));
    return Status::OK();
  }

  return Status(common::ONNXRUNTIME, common::FAIL, "Memcpy opencl: unable to get an allocator.");
}

}  // namespace opencl
}  // namespace onnxruntime
