// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "onnx/defs/tensor_proto_util.h"

#include "orttraining/training_api/module.h"
#include "orttraining/training_api/optimizer.h"
#include "orttraining/training_api/checkpoint_property.h"

/**
 * There are two representation for checkpoint respectively in memory and files:
 *
 * 1. CheckpointState. A data class representing traing states in memory, which include:
 *    i. module state:
 *        a instance of data class `ModuleCheckpointState` managed along with Module/Parameter classes,
 *    ii. optimizer state:
 *        a instance of data class `OptimizerCheckpointState` managed along with Optimizer class,
 *    iii. user defined training properties, for example 'epoch', 'best_score':
 *        a instance of data class `PropertyBag`.
 *
 *    In terms of class dependencies, Checkpoint implementations are dependent on (and on top of)
 *        Parameter/Module/Optimizer, NOT vice versa.
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
 *    LoadCheckpoint takes CheckpointState as outputs, loading from a directory of checkpoint.
 *    SaveCheckpoint takes CheckpointState as inputs, saving checkpoint files into a directory.
 */

namespace onnxruntime {
namespace training {
namespace api {

struct CheckpointState {
 public:
  ModuleCheckpointState module_checkpoint_state;
  OptimizerCheckpointState optimizer_checkpoint_state;
  PropertyBag property_bag;
};

/**
 * @brief Save training states as ORT checkpoint.
 *
 * @param state parameter/optimizer and other user defined training states.
 * @param checkpoint_path folder where checkpoint is saved.
 * @return Status
 * TODO: change state to const ref
 */
Status SaveCheckpoint(CheckpointState& state, const PathString& checkpoint_path,
                      const bool include_optimizer_state);

/**
 * @brief Save ONNX initializers as ORT checkpoint.
 *
 * @param trainable_tensor_protos trainable parameters in TensorProto format.
 * @param non_trainable_tensor_protos non-trainable parameters in TensorProto format.
 * @param checkpoint_path folder where checkpoint is saved.
 * @return Status
 */
Status SaveCheckpoint(const std::vector<ONNX_NAMESPACE::TensorProto>& trainable_tensor_protos,
                      const std::vector<ONNX_NAMESPACE::TensorProto>& non_trainable_tensor_protos,
                      const PathString& checkpoint_path);

/**
 * @brief Load training states from ORT checkpoint.
 *
 * @param checkpoint_path folder where checkpoint is stored.
 * @param checkpoint_states parameter/optimizer and other user defined training states.
 * @return Status
 */
Status LoadCheckpoint(const PathString& checkpoint_path,
                      CheckpointState& checkpoint_state);

/**
 * @brief Load training states from ORT checkpoint into a ModelProto.
 * @param checkpoint_path folder where checkpoint is stored.
 * @param model_proto Model Proto.
 * @return Status
 */
Status LoadCheckpointToModel(const PathString& checkpoint_path,
                             ONNX_NAMESPACE::ModelProto& model_proto);

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
