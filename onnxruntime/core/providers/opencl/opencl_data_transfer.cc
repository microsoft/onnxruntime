// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/ortdevice.h"
#include "core/framework/tensor.h"

#include "opencl_utils.h"
#include "opencl_data_transfer.h"

namespace onnxruntime {
namespace opencl {

OpenCLDataTransfer::OpenCLDataTransfer() {
}

OpenCLDataTransfer::~OpenCLDataTransfer() {
}

bool OpenCLDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (src_device.Type() == OrtDevice::CPU && dst_device.Type() == OrtDevice::GPU) ||
         (src_device.Type() == OrtDevice::GPU && dst_device.Type() == OrtDevice::CPU);
}

common::Status OpenCLDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  // FIXME: Implement
  std::cerr << "Dummy CopyTensor" << std::endl;;


  return Status::OK();
}

}  // namespace opencl
}  // namespace onnxruntime
