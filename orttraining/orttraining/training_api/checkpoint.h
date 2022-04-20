// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)

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

PathString CreateFolderIfNotExists(const PathString& path, const std::string& folder_name);

Status CreateOrtValuesFromTensorProtos(const PathString& model_location,
                                       const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
                                       NameMLValMap& name_to_ort_value);

/**
 * @brief A single entry for checkpoint utilities.
 *
 * A checkpoint is a directory of files:
 * checkpoint/
 *   parameters/
 *       tensors.pbseq - tensor protobuf messages
 *       tensors.bin - tensor binary data
 *   optimizers/
 *       group_0/
 *           momentum_0/
 *               tensors.pbseq - tensor protobuf messages
 *               tensors.bin - tensor binary data
 *           momentum_1/
 *               tensors.pbseq - tensor protobuf messages
 *               tensors.bin - tensor binary data
 *           properties.pbseq - group-wise property protobuf messages
 *   properties.pbseq - property protobuf messages
 */
struct CheckpointUtils {
 public:
  static CheckpointUtils GetInstance() {
    static CheckpointUtils ckpt;
    return ckpt;
  }

  /**
   * @brief Save ONNX initializers as ORT checkpoint.
   *
   * @param tensor_protos trainable parameters in TensorProto format.
   * @param checkpoint_path folder where checkpoint is saved.
   * @param model_location onnx model path.
   * @return Status
   */
  static Status Ort_Save(const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
                         const PathString& checkpoint_path) {
    ORT_RETURN_IF_ERROR(GetInstance().Ort_Save_Internal(tensor_protos, checkpoint_path));
    return Status::OK();
  }

  /**
   * @brief Save training states as ORT checkpoint.
   *
   * @param states parameter/optimizer and other user defined training states.
   * @param checkpoint_path folder where checkpoint is saved.
   * @return Status
   */
  static Status Ort_Save(CheckpointStates& states, const PathString& checkpoint_path) {
    ORT_RETURN_IF_ERROR(GetInstance().Ort_Save_Internal(states, checkpoint_path));
    return Status::OK();
  }

  /**
   * @brief Load training states from ORT checkpoint.
   *
   * @param checkpoint_path folder where checkpoint is stored.
   * @param checkpoint_states parameter/optimizer and other user defined training states.
   * @return Status
   */
  static Status Ort_Load(const PathString& checkpoint_path, CheckpointStates& checkpoint_states) {
    ORT_RETURN_IF_ERROR(GetInstance().Ort_Load_Internal(checkpoint_path, checkpoint_states));
    return Status::OK();
  }

  Status Ort_Save_Internal(const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
                           const PathString& checkpoint_path);
  Status Ort_Save_Internal(CheckpointStates& states, const PathString& checkpoint_path);
  Status Ort_Load_Internal(const PathString& checkpoint_path, CheckpointStates& checkpoint_states);

 private:
  CheckpointUtils() {}
};

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime

#endif
