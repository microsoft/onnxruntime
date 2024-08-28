// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/external_data_loader.h"
#ifndef SHARED_PROVIDER
#include "core/framework/tensor.h"
#endif
#if defined(__wasm__)
#include <emscripten.h>
#endif

namespace onnxruntime {

common::Status IExternalDataLoader::LoadTensor([[maybe_unused]] const Env& env,
                                               [[maybe_unused]] const std::filesystem::path& data_file_path,
                                               [[maybe_unused]] FileOffsetType data_offset,
                                               [[maybe_unused]] SafeInt<size_t> data_length,
                                               [[maybe_unused]] Tensor& tensor) const {
  ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
}

#if defined(__wasm__)

common::Status LoadWebAssemblyExternalData(const Env& env,
                                           const std::filesystem::path& data_file_path,
                                           FileOffsetType data_offset,
                                           SafeInt<size_t> data_length,
                                           ExternalDataLoadType load_type,
                                           void* tensor_data) {
  auto err_code = EM_ASM_INT(({
                               // If available, "Module.MountedFiles" is a Map for all preloaded files.
                               if (typeof Module == 'undefined' || !Module.MountedFiles) {
                                 return 1;  // "Module.MountedFiles" is not available.
                               }
                               let fileName = UTF8ToString(Number($0 >>> 0));
                               if (fileName.startsWith('./')) {
                                 fileName = fileName.substring(2);
                               }
                               const fileData = Module.MountedFiles.get(fileName);
                               if (!fileData) {
                                 return 2;  // File not found in preloaded files.
                               }
                               const offset = Number($1 >>> 0);
                               const length = Number($2 >>> 0);
                               const dataIdOrBuffer = Number($3 >>> 0);
                               const loadType = $4;

                               if (offset + length > fileData.byteLength) {
                                 return 3;  // Out of bounds.
                               }

                               try {
                                 const data = fileData.subarray(offset, offset + length);
                                 switch (loadType) {
                                   case 0:
                                     // Load external data to CPU memory.
                                     // Copy the file data (fileData,offset,length) into WebAssembly memory
                                     // (HEAPU8,buffer,length).
                                     HEAPU8.set(data, dataIdOrBuffer);
                                     break;
                                   case 1:
                                     // Load external data to GPU.
                                     Module.jsepUploadExternalBuffer(dataIdOrBuffer, data);
                                     break;
                                   default:
                                     return 4;  // Unknown error occurred in memory copy.
                                 }
                                 return 0;
                               } catch {
                                 return 4;
                               }
                             }),
                             data_file_path.c_str(),
                             static_cast<int32_t>(data_offset),
                             static_cast<int32_t>(data_length),
                             tensor_data,
                             static_cast<int32_t>(load_type));
  const char* err_msg;
  switch (err_code) {
    case 0:
      return Status::OK();
    case 1:
      err_msg = "Module.MountedFiles is not available.";
      break;
    case 2:
      err_msg = "File not found in preloaded files.";
      break;
    case 3:
      err_msg = "Out of bounds.";
      break;
    default:
      err_msg = "Unknown error occurred in memory copy.";
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load external data file \"", data_file_path,
                         "\", error: ", err_msg);
}

#endif

}  // namespace onnxruntime
