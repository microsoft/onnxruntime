// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdlib.h>

/// <summary>
/// Options for the OpenVINO provider that are passed to SessionOptionsAppendExecutionProvider_OpenVINO_V2.
/// Please note that this struct is *similar* to OrtOpenVINOProviderOptions.
/// Going forward, new OpenVINO provider options are to be supported via this struct and usage of the publicly defined
/// OrtOpenVINOProviderOptions will be deprecated over time.
/// </summary>
struct OrtOpenVINOProviderOptionsV2 {
  /** \brief Device type string
   *
   * Valid settings are one of: "CPU_FP32", "CPU_FP16", "GPU_FP32", "GPU_FP16"
   */
  const char* device_type;
  unsigned char enable_vpu_fast_compile;   ///< 0 = disabled, nonzero = enabled
  const char* device_id="";
  size_t num_of_threads;                   ///< 0 = Use default number of threads
  const char* cache_dir;                   ///path is set to empty by default
  void* context;
  unsigned char enable_opencl_throttling;  ///< 0 = disabled, nonzero = enabled
  unsigned char enable_dynamic_shapes;     ///< 0 = disabled, nonzero = enabled
};
