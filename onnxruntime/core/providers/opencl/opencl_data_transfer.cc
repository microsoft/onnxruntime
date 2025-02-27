// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_utils.h"
#include "opencl_program_manager.h"
#include "opencl_data_transfer.h"

#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace opencl {

OpenCLGPUDataTransfer::OpenCLGPUDataTransfer(const OpenCLExecutionProvider* exec) : exec_(exec) {}

OpenCLGPUDataTransfer::~OpenCLGPUDataTransfer() {}

bool OpenCLGPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU) ;
}

Status OpenCLGPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  // *.Location().mem_type == *.Location().device.MemType() for OpenCL EP
  const auto& src_device = src.Location().device;
  const auto& dst_device = dst.Location().device;

  if (src.SizeInBytes() == 0) {
    VLOGF_DEFAULT(V_COPY, "Skipping copy operation because source tensor size is 0");
    return Status::OK();
  }

  // HOST ==> DEV
  if (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) {
    ORT_RETURN_IF_NOT(src.ByteOffset() == 0, "Copy with byte offset is not supported");
    if (dst_device.MemType() == OrtDevice::MemType::DEFAULT) {
      VLOGF_DEFAULT(V_COPY, "Copy    host(%p) ---> Buffer(%p)", src.DataRaw(), CL_BUFFER_FROM_TENSOR(dst));
      ORT_RETURN_IF_CL_ERROR(clEnqueueWriteBuffer(exec_->GetCommandQueue(), CL_BUFFER_FROM_TENSOR(dst), /*blocking_write=*/CL_TRUE, /*offset=*/0, src.SizeInBytes(), src.DataRaw(), 0, nullptr, nullptr));
      return Status::OK();
    }
  }

  // DEV ==> HOST
  if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU) {
    ORT_RETURN_IF_NOT(dst.ByteOffset() == 0, "Copy with byte offset is not supported");
    if (src_device.MemType() == OrtDevice::MemType::DEFAULT) {
      VLOGF_DEFAULT(V_COPY, "Copy  Buffer(%p) -----> host(%p)", CL_BUFFER_FROM_TENSOR(src), dst.DataRaw());
      ORT_RETURN_IF_CL_ERROR(clEnqueueReadBuffer(exec_->GetCommandQueue(), CL_BUFFER_FROM_TENSOR(src), /*blocking_read=*/CL_TRUE, /*offset=*/0, dst.SizeInBytes(), dst.MutableDataRaw(), 0, nullptr, nullptr));
      return Status::OK();
    }
  }

  // (dev --> dev)
  if (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::GPU) {
      if (src_device.MemType() == OrtDevice::MemType::DEFAULT && dst_device.MemType() == OrtDevice::MemType::DEFAULT) {
          cl_mem src_buffer = CL_BUFFER_FROM_TENSOR(src);
          cl_mem dst_buffer = CL_BUFFER_FROM_TENSOR(dst);
          VLOGF_DEFAULT(V_COPY, "Copy Buffer(%p) ---> Buffer(%p)", src_buffer, dst_buffer);
          ORT_RETURN_IF_CL_ERROR(clEnqueueCopyBuffer(exec_->GetCommandQueue(), src_buffer, dst_buffer, /*src_offset=*/0, /*dst_offset=*/0, src.SizeInBytes(), 0, nullptr, nullptr));

          return Status::OK();
      }
  }

  return ORT_MAKE_STATUS(NONE, EP_FAIL, "Cannot copy tensor from  src_device to  dst_device");
}

}  // namespace opencl
}  // namespace onnxruntime
