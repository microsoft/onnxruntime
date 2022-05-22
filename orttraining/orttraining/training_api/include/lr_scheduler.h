// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/training_api/include/optimizer.h"

namespace onnxruntime {
namespace training {
namespace api {

/**
 * @brief Base class for learning rate scheduler.
 */
struct LRSchedulerBase {
 public:
  LRSchedulerBase(Optimizer& optimizer) : optim_(optimizer) {}
  virtual ~LRSchedulerBase() = default;

  Status Step() {
    return optim_.SetLearningRate(ComputeLearningRateInternal());
  }

 protected:
  int64_t GetStepInternal() {
    return optim_.optimizer_state_.step;
  }

  float GetInitialLRInternal() {
    return optim_.optimizer_state_.learning_rate;
  }

 private:
  virtual float ComputeLearningRateInternal() = 0;
  Optimizer& optim_;
};

/**
 * @brief Base class for multiplicative learning rate scheduler.
 *
 * The learning rate computed this way:
 *   multiplicative_factor = ComputeLRMultiplicativeFactorInternal();
 *   lr = multiplicative_factor * initial_lr;
 */
struct MultiplicativeLRSchedulerBase : public LRSchedulerBase {
 public:
  MultiplicativeLRSchedulerBase(Optimizer& optimizer)
      : LRSchedulerBase(optimizer) {
  }

 private:
  float ComputeLearningRateInternal() override {
    return GetInitialLRInternal() * ComputeLRMultiplicativeFactorInternal(GetStepInternal());
  }

  /**
   * @brief A function which computes a multiplicative factor given training step.
   * Must be implemented by sub classes.
   */
  virtual float ComputeLRMultiplicativeFactorInternal(int64_t step) = 0;
};

/**
 * @brief Decays learning rate by linearly updated multiplicative factor.
 *
 */
struct LinearLRScheduler : public MultiplicativeLRSchedulerBase {
 public:
  explicit LinearLRScheduler(Optimizer& optimizer, int64_t warmup_step_count, int64_t total_step_count)
      : MultiplicativeLRSchedulerBase(optimizer),
        warmup_step_count_flt_(static_cast<float>(warmup_step_count)),
        total_step_count_flt_(static_cast<float>(total_step_count)) {
  }

 private:
  float ComputeLRMultiplicativeFactorInternal(int64_t step) override;

  float warmup_step_count_flt_;
  float total_step_count_flt_;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
