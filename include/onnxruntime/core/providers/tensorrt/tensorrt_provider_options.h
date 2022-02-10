// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/// <summary>
/// Options for the TensorRT provider that are passed to SessionOptionsAppendExecutionProvider_TensorRT_V2.
/// Please note that this struct is *similar* to OrtTensorRTProviderOptions but only to be used internally. 
/// Going forward, new trt provider options are to be supported via this struct and usage of the publicly defined
/// OrtTensorRTProviderOptions will be deprecated over time.
/// User can only get the instance of OrtTensorRTProviderOptionsV2 via CreateTensorRTProviderOptions.
/// </summary>
struct OrtTensorRTProviderOptionsV2 {
  int device_id;                                // cuda device id.
  int has_user_compute_stream;                  // indicator of user specified CUDA compute stream.
  void* user_compute_stream;                    // user specified CUDA compute stream.
  int trt_max_partition_iterations;             // maximum iterations for TensorRT parser to get capability
  int trt_min_subgraph_size;                    // minimum size of TensorRT subgraphs
  size_t trt_max_workspace_size;                // maximum workspace size for TensorRT.
  int trt_fp16_enable;                          // enable TensorRT FP16 precision. Default 0 = false, nonzero = true
  int trt_int8_enable;                          // enable TensorRT INT8 precision. Default 0 = false, nonzero = true
  const char* trt_int8_calibration_table_name;  // TensorRT INT8 calibration table name.
  int trt_int8_use_native_calibration_table;    // use native TensorRT generated calibration table. Default 0 = false, nonzero = true
  int trt_dla_enable;                           // enable DLA. Default 0 = false, nonzero = true
  int trt_dla_core;                             // DLA core number. Default 0
  int trt_dump_subgraphs;                       // dump TRT subgraph. Default 0 = false, nonzero = true
  int trt_engine_cache_enable;                  // enable engine caching. Default 0 = false, nonzero = true
  const char* trt_engine_cache_path;            // specify engine cache path
  int trt_engine_decryption_enable;             // enable engine decryption. Default 0 = false, nonzero = true
  const char* trt_engine_decryption_lib_path;   // specify engine decryption library path
  int trt_force_sequential_engine_build;        // force building TensorRT engine sequentially. Default 0 = false, nonzero = true
};
