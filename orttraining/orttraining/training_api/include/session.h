// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/session_options.h"

namespace onnxruntime {
namespace training {
namespace api {

/**
 * @brief Get the global static session options.
 *
 * @return SessionOptions&
 */
SessionOptions& GetSessionOptions();

/**
 * @brief Get the registered global static execution provider store.
 *
 * @return std::unordered_map<std::string, std::shared_ptr<IExecutionProvider>>&
 */
std::unordered_map<std::string, std::shared_ptr<IExecutionProvider>>& GetRegisteredExecutionProviders();

/**
 * @brief Set the traing run on CUDA execution provider.
 *
 * @param cuda_options CUDA execution provider user config.
 */
void SetExecutionProvider(OrtCUDAProviderOptionsV2* cuda_options);

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
