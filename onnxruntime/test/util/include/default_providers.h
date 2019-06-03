// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace test {

// unique_ptr providers with default values for session registration
std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultMkldnnExecutionProvider(bool enable_arena = true);
std::unique_ptr<IExecutionProvider> DefaultNGraphExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultNupharExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultBrainSliceExecutionProvider();
std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider();

}  // namespace test
}  // namespace onnxruntime
