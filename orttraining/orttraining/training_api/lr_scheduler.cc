// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/lr_scheduler.h"

namespace onnxruntime {
namespace training {
namespace api {

float LinearLRScheduler::ComputeLRMultiplicativeFactorInternal(int64_t step) {
  if (step < warmup_step_count_) {
    return static_cast<float>(step) / std::max(1.0f, static_cast<float>(warmup_step_count_));
  }

  int64_t remain_step_count = total_step_count_ - step;
  int64_t post_warmup_step_count = total_step_count_ - warmup_step_count_;

  return std::max(0.f, static_cast<float>(remain_step_count) /
                           (std::max(1.f, static_cast<float>(post_warmup_step_count))));
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
