// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * @brief Temporary classes bridging between public C interfaces and internal ones.
 * Currently implementation is not covering all targeted public training APIs, more changes are expected to add.
 *
 * TODO: For longer-term, we should re-arange following interfaces following the ways of exiting APIs' definitions and exposures
 * in include/onnxruntime/core/session/onnxruntime_c_api.h and include/onnxruntime/core/session/onnxruntime_cxx_api.h.
 *
 * This is an intermediate header file for our training api internal development usage.
 *
 */

#pragma once

#include <onnxruntime_cxx_api.h>

#include "core/session/inference_session.h"
#include "core/session/environment.h"

#include "orttraining/training_api/include/module.h"
#include "orttraining/training_api/include/optimizer.h"
#include "orttraining/training_api/include/lr_scheduler.h"
#include "orttraining/training_api/include/checkpoint_property.h"
#include "orttraining/training_api/include/checkpoint.h"

using onnxruntime::training::api::LinearLRScheduler;
using onnxruntime::training::api::Module;
using onnxruntime::training::api::ModuleCheckpointState;
using onnxruntime::training::api::Optimizer;
using onnxruntime::training::api::OptimizerCheckpointState;
using onnxruntime::training::api::Parameter;

namespace Ort {

namespace {

void ToOrtValue(const std::vector<Ort::Value>& ort_value_list, std::vector<OrtValue>& ortvalue_list) {
  size_t input_len = ort_value_list.size();
  ortvalue_list.clear();
  ortvalue_list.reserve(input_len);
  const Ort::Value* ort_value_inputs_ptr = ort_value_list.data();
  auto ortvalue_inputs_ptr = reinterpret_cast<const OrtValue**>(const_cast<Ort::Value*>(ort_value_inputs_ptr));
  for (size_t i = 0; i < input_len; ++i) {
    auto& ort_value = *reinterpret_cast<const ::OrtValue*>(ortvalue_inputs_ptr[i]);
    ortvalue_list.push_back(ort_value);
  }
}

void FromOrtValue(std::vector<OrtValue>& ortvalue_list, std::vector<Ort::Value>& ort_value_list) {
  // Clean the output.
  ort_value_list.clear();
  size_t output_names_len = ortvalue_list.size();
  for (size_t i = 0; i < output_names_len; ++i)
    ort_value_list.emplace_back(nullptr);

  Ort::Value* ort_value_outputs_ptr = ort_value_list.data();
  auto ortvalue_outputs_ptr = reinterpret_cast<OrtValue**>(ort_value_outputs_ptr);
  for (size_t i = 0; i != output_names_len; ++i) {
    OrtValue& value = ortvalue_list[i];
    // Ort::Value will release the pointer once it goes out of scope.
    ortvalue_outputs_ptr[i] = new OrtValue(value);
  }
}

}  // namespace

struct OrtModule {
 public:
  OrtModule(
      OrtEnv* env, OrtSessionOptions* session_options,
      const std::string& train_model_path_or_bytes,
      std::unordered_map<std::string, std::shared_ptr<onnxruntime::training::api::Parameter>>& parameters,
      const std::optional<std::string>& eval_model_path_or_bytes = std::nullopt) {
    module_ = std::make_unique<onnxruntime::training::api::Module>(
        train_model_path_or_bytes,
        parameters,
        *reinterpret_cast<::onnxruntime::SessionOptions*>(session_options),
        *reinterpret_cast<::onnxruntime::Environment*>(env),
        eval_model_path_or_bytes);
  }

  std::unordered_map<std::string, std::shared_ptr<onnxruntime::training::api::Parameter>> NamedParameters() const {
    return module_->NamedParameters();
  }

  bool ResetGrad() {
    return module_->ResetGrad().IsOK();
  }

  bool TrainStep(const std::vector<Ort::Value>& inputs, std::vector<Ort::Value>& outputs) {
    std::vector<OrtValue> feeds;
    ToOrtValue(inputs, feeds);

    std::vector<OrtValue> fetches;
    if (!module_->TrainStep(feeds, fetches).IsOK()) {
      return false;
    }

    // Clean the output.
    outputs.clear();
    FromOrtValue(fetches, outputs);

    return true;
  }

  bool EvalStep(const std::vector<Ort::Value>& inputs, std::vector<Ort::Value>& outputs) {
    std::vector<OrtValue> feeds;
    ToOrtValue(inputs, feeds);

    std::vector<OrtValue> fetches;
    if (!module_->EvalStep(feeds, fetches).IsOK()) {
      return false;
    }

    // Clean the output.
    outputs.clear();
    FromOrtValue(fetches, outputs);
    return true;
  }

  bool GetStateDict(onnxruntime::training::api::ModuleCheckpointState& module_checkpoint_states) {
    return module_->GetStateDict(module_checkpoint_states).IsOK();
  }

 private:
  std::unique_ptr<onnxruntime::training::api::Module> module_;
};

struct OrtOptimizer {
  friend struct OrtLinearLRScheduler;

  OrtOptimizer(
      OrtEnv* env, OrtSessionOptions* session_options,
      const std::string& optim_path_or_bytes,
      const std::unordered_map<std::string, std::shared_ptr<onnxruntime::training::api::Parameter>>& parameters) {
    optimizer_ = std::make_shared<onnxruntime::training::api::Optimizer>(
        optim_path_or_bytes,
        parameters,
        *reinterpret_cast<::onnxruntime::SessionOptions*>(session_options),
        *reinterpret_cast<::onnxruntime::Environment*>(env));
  }

  bool Step() {
    return optimizer_->Step().IsOK();
  }

  bool GetStateDict(onnxruntime::training::api::OptimizerCheckpointState& optimizer_checkpoint_states) {
    return optimizer_->GetStateDict(optimizer_checkpoint_states).IsOK();
  }

 private:
  std::shared_ptr<onnxruntime::training::api::Optimizer> optimizer_;
};

struct OrtLinearLRScheduler {
  OrtLinearLRScheduler(OrtOptimizer& optimizer, int64_t warmup_step_count, int64_t total_step_count)
      : optim_(optimizer.optimizer_) {
    linear_lr_scheduler_ = std::make_unique<LinearLRScheduler>(optim_, warmup_step_count, total_step_count);
  }

  bool Step() {
    return linear_lr_scheduler_->Step().IsOK();
  }

 private:
  std::unique_ptr<onnxruntime::training::api::LinearLRScheduler> linear_lr_scheduler_;
  std::shared_ptr<onnxruntime::training::api::Optimizer> optim_;
};

}  // namespace Ort
