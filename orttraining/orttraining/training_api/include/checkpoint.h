// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "onnx/defs/tensor_proto_util.h"

#include "orttraining/training_api/include/module.h"
#include "orttraining/training_api/include/optimizer.h"
#include "orttraining/training_api/include/checkpoint_property.h"

/**
 * There are two representation for checkpoint respectively in memory and files:
 *
 * 1. CheckpointStates. A data class representing traing states in memory, which include:
 *    i. module state:
 *        a instance of data class `ModuleCheckpointStates` managed along with Module/Parameter classes,
 *    ii. optimizer state:
 *        a instance of data class `OptimizerCheckpointStates` managed along with Optimizer class,
 *    iii. user defined training properties, for example 'epoch', 'best_score':
 *        a instance of data class `PropertyBag` managed along with CheckpointProperty classes.
 *
 *    In terms of class dependencies, Checkpoint implementations are dependent on (and on top of)
 *        Parameter/Module/Optimizer/CheckpointProperty, NOT vice versa.
 *
 * 2. A directory of files:
 *    checkpoint/
 *       paramtrain_tensors.pbseq - trainable parameter tensor protobuf messages
 *       paramfrozen_tensors.pbseq - non_trainable parameter tensor protobuf messages
 *       optim_group0_momentum0_tensors.pbseq - optimizer momentum state tensor protobuf messages
 *       optim_group0_momentum1_tensors.pbseq - optimizer momentum state tensor protobuf messages
 *       optim_group0_properties.pbseq - group-wise optimizer property tensor protobuf messages
 *       custom_properties.pbseq - custom property protobuf messages
 *
 *    LoadCheckpoint takes CheckpointStates as outputs, loading from a directory of checkpoint.
 *    SaveCheckpoint takes CheckpointStates as inputs, saving checkpoint files into a directory.
 */

namespace onnxruntime {
namespace training {
namespace api {

struct CheckpointStates {
 public:
  ModuleCheckpointStates module_checkpoint_states;
  OptimizerCheckpointStates optimizer_checkpoint_states;
  PropertyBag custom_properties;
};

/**
 * @brief Save ONNX initializers as ORT checkpoint.
 *
 * @param model_uri ONNX model file path.
 * @param trainable_param_names trainable parameter names.
 * @param checkpoint_path folder where checkpoint is saved.
 * @return Status
 */
Status SaveCheckpoint(const std::string& model_uri,
                      const std::vector<std::string>& trainable_param_names,
                      const PathString& checkpoint_path);

/**
 * @brief Save training states as ORT checkpoint.
 *
 * @param states parameter/optimizer and other user defined training states.
 * @param checkpoint_path folder where checkpoint is saved.
 * @return Status
 */
Status SaveCheckpoint(CheckpointStates& states,
                      const PathString& checkpoint_path);

/**
 * @brief Load training states from ORT checkpoint.
 *
 * @param checkpoint_path folder where checkpoint is stored.
 * @param checkpoint_states parameter/optimizer and other user defined training states.
 * @return Status
 */
Status LoadCheckpoint(const PathString& checkpoint_path,
                      CheckpointStates& checkpoint_states);

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
