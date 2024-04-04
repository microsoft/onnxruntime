// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "orttraining/training_api/module.h"
#include "orttraining/training_api/optimizer.h"
#include "orttraining/training_api/lr_scheduler.h"
#include "orttraining/training_api/checkpoint.h"
#include "orttraining/training_api/utils.h"

namespace onnxruntime {
namespace training {
namespace api {
using namespace common;

// Wrapper on top of module and optimizer classes and is the only class exposed via capis
class TrainingSession {
 public:
  TrainingSession(const Environment& session_env,
                  const SessionOptions& session_options,
                  const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
                  CheckpointState* state,
                  const ModelIdentifiers& model_identifiers,
                  gsl::span<OrtCustomOpDomain* const> custom_op_domains = gsl::span<OrtCustomOpDomain* const>());

  Status RegisterScheduler(const std::function<
                               std::unique_ptr<LRSchedulerBase>(std::shared_ptr<Optimizer>)>& get_scheduler,
                           float initial_lr);

  size_t GetTrainingModelOutputCount() const noexcept;

  size_t GetEvalModelOutputCount() const noexcept;

  std::string GetTrainingModelOutputName(size_t index) const noexcept;

  std::string GetEvalModelOutputName(size_t index) const noexcept;

  size_t GetTrainingModelInputCount() const noexcept;

  size_t GetEvalModelInputCount() const noexcept;

  std::string GetTrainingModelInputName(size_t index) const noexcept;

  std::string GetEvalModelInputName(size_t index) const noexcept;

  Status TrainStep(const RunOptions& run_options,
                   const std::vector<OrtValue>& inputs,
                   std::vector<OrtValue>& fetches);

  Status EvalStep(const RunOptions& run_options,
                  const std::vector<OrtValue>& inputs,
                  std::vector<OrtValue>& fetches) const;

  Status LazyResetGrad();

  Status OptimizerStep(const RunOptions& run_options);

  Status SetLearningRate(float learning_rate) noexcept;

  float GetLearningRate() const;

  Status SchedulerStep() noexcept;

  size_t GetParametersSize(const bool trainable_only = true) const;

  Status CopyParametersToBuffer(OrtValue& parameters_buffer, const bool trainable_only = true);

  Status CopyBufferToParameters(OrtValue& parameters_buffer, const bool trainable_only = true);

#if !defined(ORT_MINIMAL_BUILD)
  Status ExportModelForInferencing(const std::string& inference_model_path,
                                   gsl::span<const std::string> graph_output_names) const;
#endif

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TrainingSession);

  CheckpointState* state_;  // Non owning pointer to the checkpoint state. It must outlive the training session.
  std::unique_ptr<Module> module_;
  std::shared_ptr<Optimizer> optimizer_;
  std::unique_ptr<LRSchedulerBase> scheduler_;
};
}  // namespace api
}  // namespace training
}  // namespace onnxruntime
