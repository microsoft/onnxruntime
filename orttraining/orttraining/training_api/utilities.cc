// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "orttraining/training_api/interfaces.h"
#include "orttraining/training_api/utilities.h"

namespace onnxruntime {
namespace training {
namespace api_test {
namespace utils {

void SetExecutionProvider(const Module& /*module*/, const Optimizer& /*optimizer*/, IExecutionProvider* /*provider*/) {
  ORT_NOT_IMPLEMENTED("Not implemented.");
}

}  // namespace utils
}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime
