// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/include/lr_scheduler.h"

namespace onnxruntime {
namespace training {
namespace api {

float LinearLRScheduler::ComputeLRMultiplicativeFactorInternal(int64_t step) {
  float step_flt = static_cast<float>(step);
  if (step_flt < warmup_step_count_flt_) {
    return step_flt / std::max(1.f, warmup_step_count_flt_);
  }

  float remain_step_count = total_step_count_flt_ - step_flt;
  static float post_warmup_step_count = total_step_count_flt_ - warmup_step_count_flt_;

  return std::max(0.f, remain_step_count / (std::max(1.f, post_warmup_step_count)));
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
