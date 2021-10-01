// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_gpu_data_transfer.h"
#include <sstream>
#include <fstream>
#include <memory>
#include <iostream>

// use default stream for copy for now, to avoid racing in BFC arena as in issue #4829
// note this may cause some models to run slower if there are ops running on CPU
// so we leave it as optional, in case user need the previous behavior
// a full fix to BFC arena is being looked at, and once it's in, we can revert this change
namespace onnxruntime {
OVGPUDataTransfer::OVGPUDataTransfer() {

}

OVGPUDataTransfer::~OVGPUDataTransfer() {
  
}

bool OVGPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU || src_device.MemType() == OrtDevice::MemType::DEFAULT ||
         dst_device.Type() == OrtDevice::GPU || dst_device.MemType() == OrtDevice::MemType::DEFAULT ;
}

common::Status OVGPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;
  
  exec_queue_id ++; 

  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::CPU ) {
     std::cout << "Destination GPU input CPU\n";

    } 
  } else if (src_device.Type() == OrtDevice::GPU) {
    if (dst_device.Type() == OrtDevice::CPU) {
      // copying from GPU to pinned memory, this is non-blocking
       std::cout << "Src GPU Destination CPU\n";

    } else {
      // copying from GPU to GPU memory, this is blocking

      std::cout << "GPU to GPU memory\n";
    }
  } else {
    // copying between cpu memory
    std::cout << "Copy between normal GPU Memory\n";
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}
}  // namespace onnxruntime
