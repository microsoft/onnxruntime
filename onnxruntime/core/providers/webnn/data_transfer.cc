// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webnn/data_transfer.h"

#include <emscripten.h>
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace webnn {

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  // Copying data between MLTensors is not supported by WebNN.
  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  if (!emscripten::val::module_property("shouldTransferToMLTensor").as<bool>()) {
    // We don't need to transfer the buffer to an MLTensor, so we don't need to copy the buffer.
    return Status::OK();
  }

  size_t bytes = src.SizeInBytes();
  if (bytes > 0) {
    const void* src_data = src.DataRaw();
    void* dst_data = dst.MutableDataRaw();

    const auto& dst_device = dst.Location().device;

    if (dst_device.Type() == OrtDevice::GPU) {
      EM_ASM({ Module.jsepUploadTensor($0, HEAPU8.subarray($1, $1 + $2)); }, dst_data, reinterpret_cast<intptr_t>(src_data), bytes);
    } else {
      auto jsepDownloadTensor = emscripten::val::module_property("jsepDownloadTensor");
      auto subarray = emscripten::typed_memory_view(bytes, static_cast<char*>(dst_data));
      jsepDownloadTensor(reinterpret_cast<intptr_t>(src_data), subarray).await();
    }
  }

  return Status::OK();
}

}  // namespace webnn
}  // namespace onnxruntime
