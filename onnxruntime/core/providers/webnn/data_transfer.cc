// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webnn/data_transfer.h"

#include <emscripten.h>
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace webnn {

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  // Copying data between MLBuffers is not supported by WebNN.
  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  if (!emscripten::val::module_property("shouldTransferToMLBuffer").as<bool>()) {
    // We don't need to transfer the buffer to an MLBuffer, so we don't need to copy the buffer.
    return Status::OK();
  }

  size_t bytes = src.SizeInBytes();
  if (bytes > 0) {
    const void* src_data = src.DataRaw();
    void* dst_data = dst.MutableDataRaw();

    const auto& dst_device = dst.Location().device;

    if (dst_device.Type() == OrtDevice::GPU) {
      EM_ASM({ Module.jsepUploadBuffer($0, HEAPU8.subarray($1, $1 + $2)); }, dst_data, reinterpret_cast<intptr_t>(src_data), bytes);
    } else {
      auto jsepDownloadBuffer = emscripten::val::module_property("jsepDownloadBuffer");
      auto buffer = jsepDownloadBuffer(reinterpret_cast<intptr_t>(src_data)).await();
      EM_ASM({
        const buffer = Emval.toValue($0);
        const src_array = new Uint8Array(buffer, 0, $2);
        HEAPU8.set(src_array, $1); }, buffer.as_handle(), reinterpret_cast<intptr_t>(dst_data), bytes);
    }
  }

  return Status::OK();
}

}  // namespace webnn
}  // namespace onnxruntime
