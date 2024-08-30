// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/// <summary>
/// Options for the WebGPU provider that are passed to SessionOptionsAppendExecutionProvider_WGPU.
/// User can only get the instance of OrtWGPUProviderOptions via CreateWGPUProviderOptions.
/// </summary>
struct OrtWGPUProviderOptions {
  OrtWGPUProviderOptions& operator=(const OrtWGPUProviderOptions& other);  // copy assignment operator

  int device_id{0};  // device id.

  void* instance_handle{nullptr};  // WebGPU instance handle.
  void* adapter_handle{nullptr};   // WebGPU adapter handle.
  void* device_handle{nullptr};    // WebGPU device handle.
  void* dawn_proc_table{nullptr};  // DawnProcTable pointer.

  const char* const* provider_options_keys{nullptr};    // Provider options keys.
  const char* const* provider_options_values{nullptr};  // Provider options values.
  size_t num_keys{0};                                   // Number of provider options.
};
