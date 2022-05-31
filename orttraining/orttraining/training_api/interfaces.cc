// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/include/module.h"
#include "orttraining/training_api/include/optimizer.h"

#include "orttraining/training_api/include/interfaces.h"

namespace onnxruntime {
namespace training {
namespace api {

// Sets the Execution provider for train, eval and optimizer sessions
Status SetExecutionProvider(const Module& module, const Optimizer& optimizer, const std::shared_ptr<IExecutionProvider>& p_exec_provider) {
  ORT_THROW_IF_ERROR(module.train_sess_->RegisterExecutionProvider(p_exec_provider));
  if (nullptr != module.eval_sess_) {
    ORT_THROW_IF_ERROR(module.eval_sess_->RegisterExecutionProvider(p_exec_provider));
  }
  ORT_THROW_IF_ERROR(optimizer.optim_sess_->RegisterExecutionProvider(p_exec_provider));
  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
