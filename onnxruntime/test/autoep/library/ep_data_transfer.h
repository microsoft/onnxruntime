// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

#include "ep_stream_support.h"
#include "utils.h"

struct ExampleDataTransfer : OrtDataTransferImpl, ApiPtrs {
  ExampleDataTransfer(ApiPtrs api_ptrs,
                      const OrtMemoryDevice* device_mem_info_,
                      const OrtMemoryDevice* shared_mem_info_ = nullptr)
      : ApiPtrs(api_ptrs), device_mem_info{device_mem_info_}, shared_mem_info{shared_mem_info_} {
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
  }

  static bool ORT_API_CALL CanCopyImpl(_In_ void* this_ptr,
                                       _In_ const OrtMemoryDevice* src_memory_device,
                                       _In_ const OrtMemoryDevice* dst_memory_device) noexcept;

  // function to copy one or more tensors.
  // implementation can optionally use async copy if a stream is available for the input.
  static OrtStatus* ORT_API_CALL CopyTensorsImpl(_In_ void* this_ptr,
                                                 _In_reads_(num_tensors) const OrtValue** src_tensors_ptr,
                                                 _In_reads_(num_tensors) OrtValue** dst_tensors_ptr,
                                                 _In_reads_(num_tensors) OrtSyncStream** streams_ptr,
                                                 _In_ size_t num_tensors) noexcept;

 private:
  const OrtMemoryDevice* device_mem_info;
  const OrtMemoryDevice* shared_mem_info;
};
