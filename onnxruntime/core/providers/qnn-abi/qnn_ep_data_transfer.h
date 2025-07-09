// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test/autoep/library/example_plugin_ep_utils.h"

struct QnnDataTransfer : OrtDataTransferImpl, ApiPtrs {
  QnnDataTransfer(ApiPtrs api_ptrs,
                  const OrtMemoryDevice* device_mem_info_,
                  const OrtMemoryDevice* shared_mem_info_ = nullptr)
      : ApiPtrs(api_ptrs), device_mem_info{device_mem_info_}, shared_mem_info{shared_mem_info_} {
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;
  }

  static bool ORT_API_CALL CanCopyImpl(void* this_ptr,
                                       const OrtMemoryDevice* src_memory_device,
                                       const OrtMemoryDevice* dst_memory_device) noexcept;

  // function to copy one or more tensors.
  // implementation can optionally use async copy if a stream is available for the input.
  static OrtStatus* ORT_API_CALL CopyTensorsImpl(void* this_ptr,
                                                 const OrtValue** src_tensors_ptr,
                                                 OrtValue** dst_tensors_ptr,
                                                 OrtSyncStream** streams_ptr,
                                                 size_t num_tensors) noexcept;
  static void ORT_API_CALL ReleaseImpl(void* this_ptr) noexcept;

 private:
  const OrtMemoryDevice* device_mem_info;
  const OrtMemoryDevice* shared_mem_info;
};
