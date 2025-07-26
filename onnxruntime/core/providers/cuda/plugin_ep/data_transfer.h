// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once

#include "core/session/onnxruntime_c_api.h"

#include "core/providers/cuda/plugin_ep/utils.h"

namespace cuda_plugin_ep {
struct CudaDataTransferImpl : OrtDataTransferImpl {
  CudaDataTransferImpl();

  static bool CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                          const OrtMemoryDevice* src_memory_device,
                          const OrtMemoryDevice* dst_memory_device) noexcept;

  static OrtStatus* CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                    const OrtValue** src_tensors,
                                    OrtValue** dst_tensors,
                                    OrtSyncStream** streams,
                                    size_t num_tensors) noexcept;

  static void ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;
};
}  // namespace cuda_plugin_ep
