// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "example_plugin_ep_utils.h"

struct ExampleDataTransfer : OrtDataTransferImpl, ApiPtrs {
  ExampleDataTransfer(ApiPtrs api_ptrs,
                      const OrtMemoryDevice* device_mem_info_)
      : ApiPtrs(api_ptrs), device_mem_info{device_mem_info_} {
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;
  }

  static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                       const OrtMemoryDevice* src_memory_device,
                                       const OrtMemoryDevice* dst_memory_device) noexcept;

  // function to copy one or more tensors.
  // implementation can optionally use async copy if a stream is available for the input.
  static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                 const OrtValue** src_tensors_ptr,
                                                 OrtValue** dst_tensors_ptr,
                                                 OrtSyncStream** streams_ptr,
                                                 size_t num_tensors) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

 private:
  const OrtMemoryDevice* device_mem_info;  // device our EP runs on
};
