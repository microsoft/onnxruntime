// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <thread>
#include <future>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/framework_common.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"

namespace onnxruntime {
namespace training {
class IOBinding;

class TrainingAgent {
 public:
  explicit TrainingAgent(InferenceSession& session);
  ~TrainingAgent();
  // For ORTModule.forward()
  common::Status RunForward(const onnxruntime::RunOptions& run_options, onnxruntime::IOBinding& io_binding) ORT_MUST_USE_RESULT;
  // For ORTModule.backward()
  common::Status RunBackward(const onnxruntime::RunOptions& run_options, onnxruntime::IOBinding& io_binding) ORT_MUST_USE_RESULT;

 private:
  // TrainingAgent runs on a InferenceSession under the hood
  InferenceSession& inference_session_;
};

}  // namespace training
}  // namespace onnxruntime
