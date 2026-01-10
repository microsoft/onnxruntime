// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(__wasm__)

#include <emscripten.h>

#include "core/framework/tensor.h"
#include "core/providers/webgpu/external_data_loader.h"

namespace onnxruntime {
namespace webgpu {

bool ExternalDataLoader::CanLoad(const OrtMemoryInfo& target_memory_info) const {
  return target_memory_info.device.Type() == OrtDevice::CPU ||
         (target_memory_info.device.Type() == OrtDevice::GPU && target_memory_info.name == WEBGPU_BUFFER);
}

common::Status ExternalDataLoader::LoadTensor(const Env& env,
                                              const std::filesystem::path& data_file_path,
                                              FileOffsetType data_offset,
                                              SafeInt<size_t> data_length,
                                              Tensor& tensor) const {
  ExternalDataLoadType load_type;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    load_type = ExternalDataLoadType::CPU;
  } else if (tensor.Location().device.Type() == OrtDevice::GPU &&
             tensor.Location().name == WEBGPU_BUFFER) {
    load_type = ExternalDataLoadType::WEBGPU_BUFFER;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported tensor location: ", tensor.Location().ToString());
  }

  return LoadWebAssemblyExternalData(env, data_file_path, data_offset, data_length, load_type, tensor.MutableDataRaw());
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif
