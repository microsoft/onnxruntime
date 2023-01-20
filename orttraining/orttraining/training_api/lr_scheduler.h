// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/training_api/optimizer.h"

namespace onnxruntime {
namespace training {
namespace api {

/**
 * @brief Base class for learning rate scheduler.
 */
struct LRSchedulerBase {
 public:
  LRSchedulerBase(std::shared_ptr<Optimizer> optimizer) : optim_(optimizer) {}
  virtual ~LRSchedulerBase() = default;

  /**
   * @brief Compute learning rate taking step and initial learning rate as inputs, then update
   * the adapted learning rate into optimizer.
   *
   * This should be called every time optimizer update states related to learning rate calculations,
   * for example, initial learning and step.
   */
  Status Step() {
    return optim_->SetLearningRate(ComputeLearningRateInternal());
  }

 protected:
  int64_t GetStepInternal() {
    return optim_->optimizer_state_.step;
  }

  float GetInitialLRInternal() {
    return optim_->optimizer_state_.initial_lr;
  }

 private:
  virtual float ComputeLearningRateInternal() = 0;
  std::shared_ptr<Optimizer> optim_;
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
  MultiplicativeLRSchedulerBase(std::shared_ptr<Optimizer> optimizer)
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
  explicit LinearLRScheduler(std::shared_ptr<Optimizer> optimizer, int64_t warmup_step_count, int64_t total_step_count)
      : MultiplicativeLRSchedulerBase(optimizer),
        warmup_step_count_(warmup_step_count),
        total_step_count_(total_step_count) {
    ORT_THROW_IF_ERROR(Step());
  }

 private:
  float ComputeLRMultiplicativeFactorInternal(int64_t step) override;

  int64_t warmup_step_count_;
  int64_t total_step_count_;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
