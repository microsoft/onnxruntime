// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace training {
namespace api {

class Module;    //forward declaration
class Optimizer; //forward declaration

// Sets the Execution provider for train, eval and optimizer sessions
Status SetExecutionProvider(const Module& module, const Optimizer& optimizer, const std::shared_ptr<IExecutionProvider>& p_exec_provider);

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
