// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "providers.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options);

namespace test {

// unique_ptr providers with default values for session registration
std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultNupharExecutionProvider(bool allow_unaligned_buffers = true);
std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultMIGraphXExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultRknpuExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultAclExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultArmNNExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultRocmExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultCoreMLExecutionProvider();

// EP for internal testing
std::unique_ptr<IExecutionProvider> DefaultInternalTestingExecutionProvider(
    const std::unordered_set<std::string>& supported_ops);

}  // namespace test
}  // namespace onnxruntime
