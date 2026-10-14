// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "core/framework/provider_options.h"
#include "core/providers/providers.h"

#include "core/providers/webgpu/math/matmul_algorithm.h"
#include "core/providers/webgpu/webgpu_provider_options.h"

struct OrtDataTransferImpl;

namespace onnxruntime {
struct ConfigOptions;

struct WebGpuExecutionProviderTestOptions {
  uint64_t max_storage_buffer_binding_size{0};
  std::optional<webgpu::MatMulAlgorithm> forced_matmul_algorithm;
};

struct WebGpuProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const ConfigOptions& config_options);
  static std::shared_ptr<IExecutionProviderFactory> CreateForTesting(
      const ConfigOptions& config_options, const WebGpuExecutionProviderTestOptions& test_options);
};

// Each copy uses its session stream or the GPU tensor's registered context.
// Caller takes ownership of the returned OrtDataTransferImpl*
OrtDataTransferImpl* OrtWebGpuCreateDataTransfer();

}  // namespace onnxruntime
