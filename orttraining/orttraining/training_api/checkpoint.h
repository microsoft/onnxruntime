// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/path_lib.h"
#include "orttraining/training_api/checkpoint_property.h"
#include "orttraining/training_api/module.h"
#include "orttraining/training_api/optimizer.h"

/**
 * The CheckpointState is a data class representing traning states in memory, which include:
 *  - Module state: Contains the model's trainable and non-trainable parameters.
 *  - Optimizer state: Contains the optimizer's state (for example learning rate, step, first
 *                     and second order momentums ...).
 *  - User defined properties: For example 'epoch', 'best_score' ...
 *
 * These states can be used to begin or resume training from a checkpoint. In order to pause training,
 * the user can save the CheckpointState to a checkpoint file.
 *
 * The checkpoint file is a single flatbuffer file containing all the states highlighted above.
 * The flatbuffer schema is defined in onnxruntime/core/flatbuffers/schema/ort_training_checkpoint.fbs
 *
 */

namespace onnxruntime::training::api {

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
 * @param checkpoint_path file where checkpoint is saved.
 * @return Status
 */
Status SaveCheckpoint(const CheckpointState& state, const PathString& checkpoint_path,
                      const bool include_optimizer_state);

#if !defined(ORT_MINIMAL_BUILD)
/**
 * @brief Save ONNX initializers as ORT checkpoint.
 *
 * @param trainable_tensor_protos trainable parameters in TensorProto format.
 * @param non_trainable_tensor_protos non-trainable parameters in TensorProto format.
 * @param checkpoint_path file where checkpoint is saved.
 * @param nominal_checkpoint flag indicating whether to save the complete checkpoint or the nominal checkpoint.
 * @return Status
 */
Status SaveCheckpoint(gsl::span<const ONNX_NAMESPACE::TensorProto> trainable_tensor_protos,
                      gsl::span<const ONNX_NAMESPACE::TensorProto> non_trainable_tensor_protos,
                      const PathString& checkpoint_path, const bool nominal_checkpoint);
#endif

/**
 * @brief Load training states from ORT checkpoint.
 *
 * @param checkpoint_path file where checkpoint is stored.
 * @param checkpoint_states parameter/optimizer and other user defined training states.
 * @return Status
 */
Status LoadCheckpoint(const PathString& checkpoint_path,
                      CheckpointState& checkpoint_state);

/**
 * @brief Load training states from ORT checkpoint bytes buffer.
 * @param checkpoint_bytes bytes buffer of the checkpoint.
 * @param checkpoint_state parameter/optimizer and other user defined training states.
 * @return Status
 */
Status LoadCheckpointFromBuffer(gsl::span<const uint8_t> checkpoint_bytes,
                                CheckpointState& checkpoint_state);

#if !defined(ORT_MINIMAL_BUILD)
/**
 * @brief Load training states from ORT checkpoint into a ModelProto.
 * @param checkpoint_path file where checkpoint is stored.
 * @param model_proto ONNX model to load the checkpoint to.
 * @return Status
 */
Status LoadCheckpointToModel(const PathString& checkpoint_path,
                             ONNX_NAMESPACE::ModelProto& model_proto);
#endif

}  // namespace onnxruntime::training::api
