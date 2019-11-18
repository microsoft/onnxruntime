// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace test {

// unique_ptr providers with default values for session registration
std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultMkldnnExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultNGraphExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultNupharExecutionProvider(bool allow_unaligned_buffers = true);
std::unique_ptr<IExecutionProvider> DefaultBrainSliceExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultAclExecutionProvider(bool enable_arena = true);

}  // namespace test
}  // namespace onnxruntime
