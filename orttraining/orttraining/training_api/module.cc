// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "orttraining/training_api/include/module.h"

namespace onnxruntime {
namespace training {
namespace api {

Status Module::GetStateDict(ModuleCheckpointStates& module_checkpoint_states) {
  module_checkpoint_states.named_parameters = named_parameters();

  // Pass the training session data transfer manager for data copying when saving.
  // An alternative is, we can do copy at this stage.
  ORT_RETURN_IF_NOT(train_sess_, "training session not initialized");
  const DataTransferManager& sess_data_transfer_manager = train_sess_->GetDataTransferManager();
  module_checkpoint_states.train_session_data_transfer_mgr = &sess_data_transfer_manager;
  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
