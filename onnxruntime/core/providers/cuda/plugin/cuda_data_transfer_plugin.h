// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"

namespace onnxruntime {
namespace cuda_plugin {

/// CUDA data transfer implementation for CPU↔GPU and GPU↔GPU copies.
class CudaDataTransfer : public OrtDataTransferImpl {
 public:
  CudaDataTransfer(const OrtApi& ort_api, const OrtEpApi& ep_api,
                   const OrtMemoryDevice* gpu_device);
  ~CudaDataTransfer() = default;

 private:
  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

  static bool ORT_API_CALL CanCopyImpl(
      const OrtDataTransferImpl* this_ptr,
      const OrtMemoryDevice* src_device,
      const OrtMemoryDevice* dst_device) noexcept;

  static OrtStatus* ORT_API_CALL CopyTensorsImpl(
      OrtDataTransferImpl* this_ptr,
      const OrtValue** src_tensors,
      OrtValue** dst_tensors,
      OrtSyncStream** streams,
      size_t count) noexcept;

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtMemoryDevice* gpu_device_;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
