// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "onnx/defs/tensor_proto_util.h"
#include "orttraining/training_api/interfaces.h"
#include "orttraining/training_api/checkpoint_property.h"

namespace onnxruntime {
namespace training {
namespace api {

/**
 * @brief A data class representing traing states, which include:
 * 1). parameter states,
 * 2). optimizer states,
 * 3). user defined training properties, for example 'epoch', 'best_score'.
 */
struct CheckpointStates {
 public:
  ModuleCheckpointStates module_checkpoint_states;
  OptimizerCheckpointStates optimizer_checkpoint_states;
  PropertyBag custom_properties;
};

/**
 * @brief The single entry for checkpoint utilities.
 *
 * A checkpoint is a directory of files:
 * checkpoint/
 *   param_tensors.pbseq - parameter tensor protobuf messages
 *   optim_group0_momentum0_tensors.pbseq - optimizer momentum state tensor protobuf messages
 *   optim_group0_momentum1_tensors.pbseq - optimizer momentum state tensor protobuf messages
 *   optim_group0_properties.pbseq - group-wise optimizer property tensor protobuf messages
 *   custom_properties.pbseq - custom property protobuf messages
 */
struct CheckpointUtils {
 public:
  /**
   * @brief Save ONNX initializers as ORT checkpoint.
   *
   * @param model_uri ONNX model file path.
   * @param trainable_param_names trainable parameter names.
   * @param checkpoint_path folder where checkpoint is saved.
   * @return Status
   */
  static Status SaveORTCheckpoint(const std::string& model_uri,
                                  const std::vector<std::string>& trainable_param_names,
                                  const PathString& checkpoint_path) {
    return OrtSaveInternal(model_uri, trainable_param_names, checkpoint_path);
  }

  /**
   * @brief Save training states as ORT checkpoint.
   *
   * @param states parameter/optimizer and other user defined training states.
   * @param checkpoint_path folder where checkpoint is saved.
   * @return Status
   */
  static Status SaveORTCheckpoint(CheckpointStates& states, const PathString& checkpoint_path) {
    return OrtSaveInternal(states, checkpoint_path);
  }

  /**
   * @brief Load training states from ORT checkpoint.
   *
   * @param checkpoint_path folder where checkpoint is stored.
   * @param checkpoint_states parameter/optimizer and other user defined training states.
   * @return Status
   */
  static Status LoadORTCheckpoint(const PathString& checkpoint_path, CheckpointStates& checkpoint_states) {
    return OrtLoadInternal(checkpoint_path, checkpoint_states);
  }

 private:
  CheckpointUtils() {}

  static Status OrtSaveInternal(const std::string& model_uri,
                                const std::vector<std::string>& trainable_param_names,
                                const PathString& checkpoint_path);

  static Status OrtSaveModuleStatesInternal(ModuleCheckpointStates& module_states, const PathString& parameter_folder_path);
  static Status OrtSaveOptimizerStatesInternal(OptimizerCheckpointStates& optimizer_states, const PathString& optimizer_folder_path);
  static Status OrtSaveInternal(CheckpointStates& states, const PathString& checkpoint_path);

  static Status OrtLoadModuleStatesInternal(const PathString& parameter_folder_path, ModuleCheckpointStates& module_states);
  static Status OrtLoadOptimizerStatesInternal(const PathString& optimizer_folder_path, OptimizerCheckpointStates& optimizer_states);
  static Status OrtLoadInternal(const PathString& checkpoint_path, CheckpointStates& checkpoint_states);
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
