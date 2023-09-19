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
  int device_id{0};                                      // cuda device id.
  int has_user_compute_stream{0};                        // indicator of user specified CUDA compute stream.
  void* user_compute_stream{nullptr};                    // user specified CUDA compute stream.
  int trt_max_partition_iterations{1000};                // maximum iterations for TensorRT parser to get capability
  int trt_min_subgraph_size{1};                          // minimum size of TensorRT subgraphs
  size_t trt_max_workspace_size{1 << 30};                // maximum workspace size for TensorRT.
  int trt_fp16_enable{0};                                // enable TensorRT FP16 precision. Default 0 = false, nonzero = true
  int trt_int8_enable{0};                                // enable TensorRT INT8 precision. Default 0 = false, nonzero = true
  const char* trt_int8_calibration_table_name{nullptr};  // TensorRT INT8 calibration table name.
  int trt_int8_use_native_calibration_table{0};          // use native TensorRT generated calibration table. Default 0 = false, nonzero = true
  int trt_dla_enable{0};                                 // enable DLA. Default 0 = false, nonzero = true
  int trt_dla_core{0};                                   // DLA core number. Default 0
  int trt_dump_subgraphs{0};                             // dump TRT subgraph. Default 0 = false, nonzero = true
  int trt_engine_cache_enable{0};                        // enable engine caching. Default 0 = false, nonzero = true
  const char* trt_engine_cache_path{nullptr};            // specify engine cache path
  int trt_engine_decryption_enable{0};                   // enable engine decryption. Default 0 = false, nonzero = true
  const char* trt_engine_decryption_lib_path{nullptr};   // specify engine decryption library path
  int trt_force_sequential_engine_build{0};              // force building TensorRT engine sequentially. Default 0 = false, nonzero = true
  int trt_context_memory_sharing_enable{0};              // enable context memory sharing between subgraphs. Default 0 = false, nonzero = true
  int trt_layer_norm_fp32_fallback{0};                   // force Pow + Reduce ops in layer norm to FP32. Default 0 = false, nonzero = true
  int trt_timing_cache_enable{0};                        // enable TensorRT timing cache. Default 0 = false, nonzero = true
  int trt_force_timing_cache{0};                         // force the TensorRT cache to be used even if device profile does not match. Default 0 = false, nonzero = true
  int trt_detailed_build_log{0};                         // Enable detailed build step logging on TensorRT EP with timing for each engine build. Default 0 = false, nonzero = true
  int trt_build_heuristics_enable{0};                    // Build engine using heuristics to reduce build time. Default 0 = false, nonzero = true
  int trt_sparsity_enable{0};                            // Control if sparsity can be used by TRT. Default 0 = false, 1 = true
  int trt_builder_optimization_level{3};                 // Set the builder optimization level. WARNING: levels below 3 do not guarantee good engine performance, but greatly improve build time.  Default 3, valid range [0-5]
  int trt_auxiliary_streams{-1};                         // Set maximum number of auxiliary streams per inference stream. Setting this value to 0 will lead to optimal memory usage. Default -1 = heuristics
  const char* trt_tactic_sources{nullptr};               // pecify the tactics to be used by adding (+) or removing (-) tactics from the default
                                                         // tactic sources (default = all available tactics) e.g. "-CUDNN,+CUBLAS" available keys: "CUBLAS"|"CUBLAS_LT"|"CUDNN"|"EDGE_MASK_CONVOLUTIONS"
  const char* trt_extra_plugin_lib_paths{nullptr};       // specify extra TensorRT plugin library paths
  const char* trt_profile_min_shapes{nullptr};           // Specify the range of the input shapes to build the engine with
  const char* trt_profile_max_shapes{nullptr};           // Specify the range of the input shapes to build the engine with
  const char* trt_profile_opt_shapes{nullptr};           // Specify the range of the input shapes to build the engine with
  int trt_cuda_graph_enable{0};                          // Enable CUDA graph in ORT TRT
};
