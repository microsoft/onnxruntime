// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "orttraining/core/framework/checkpointing.h"
#include "orttraining/training_api/interfaces.h"
#include <type_traits>
#include "onnx/defs/tensor_proto_util.h"

namespace onnxruntime {
namespace training {

// TODO: Rename to api after all major classes implemented.
namespace api_test {

/**
 * @brief Base class for user defined checkpoint property.
 */
struct CheckpointProperty {
 public:
  CheckpointProperty(const std::string& prop_name)
      : prop_name_(prop_name) {
  }

  virtual ~CheckpointProperty() {}
  virtual ONNX_NAMESPACE::TensorProto ToTensorProto() = 0;

  std::string GetName() const {
    return prop_name_;
  }

 protected:
  std::string prop_name_;
};

/**
 * @brief User defined checkpoint property.
 * Only support int64_t, std::string and float data types.
 */
template <typename T>
struct TypedCheckpointProperty : public CheckpointProperty {
 public:
  TypedCheckpointProperty(const std::string& prop_name, const T& prop_value)
      : CheckpointProperty(prop_name), prop_value_(prop_value) {
  }

  ONNX_NAMESPACE::TensorProto ToTensorProto() override {
    auto t_proto = ONNX_NAMESPACE::ToTensor<T>(prop_value_);

    // We did not add dims assuming currently we only support scalars.
    return t_proto;
  }

  T GetData() const {
    return prop_value_;
  }

 private:
  T prop_value_;
};

/**
 * @brief A data class representing traing states.
 * Including:
 *     > Parameter states.
 *     > Optimizer states.
 *     > User defined training properties, for example 'epoch', 'best_score'.
 */
struct CheckpointStates {
 public:
  ModuleCheckpointStates module_checkpoint_states;
  OptimizerCheckpointStates optimizer_checkpoint_states;
  std::unordered_map<std::string, std::shared_ptr<CheckpointProperty>> named_properties;
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
 *   properties.pbseq - custom property protobuf messages
 */
struct CheckpointUtils {
 public:
  /**
   * @brief Save ONNX initializers as ORT checkpoint.
   *
   * @param tensor_protos parameters in TensorProto format.
   * @param trainable_param_names trainable parameter names.
   * @param checkpoint_path folder where checkpoint is saved.
   * @param model_location onnx model path.
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

Status CreateOrtValuesFromTensorProtos(
    const std::vector<const ONNX_NAMESPACE::TensorProto*>& tensor_protos,
    NameMLValMap& name_to_ort_value);

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime
