// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <emscripten.h>

#include "core/providers/js/data_transfer.h"

EM_ASYNC_JS(void, jsepDownload, (const void* src_data, void* dst_data, size_t bytes), {
  await Module.jsepCopyAsync(src_data, dst_data, bytes);
});

namespace onnxruntime {
namespace js {

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::GPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::GPU) {
      // copy from GPU to GPU
      EM_ASM({ Module.jsepCopy($0, $1, $2, true); }, src_data, dst_data, bytes);
    } else {
      // copy from CPU to GPU
      EM_ASM({ Module.jsepCopy($0, $1, $2); }, src_data, dst_data, bytes);
    }
  } else /* if (src_device.Type() == OrtDevice::GPU) */ {
    // copy from GPU to CPU
    jsepDownload(src_data, dst_data, bytes);
  }

  return Status::OK();
}

}  // namespace js
}  // namespace onnxruntime
