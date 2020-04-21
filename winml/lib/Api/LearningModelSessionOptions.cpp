// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "LearningModelSessionOptions.h"

namespace WINMLP {
LearningModelSessionOptions::LearningModelSessionOptions(const LearningModelSessionOptions& options) : batch_size_override_(options.batch_size_override_),
                                                                                                       close_model_on_session_creation_(options.close_model_on_session_creation_) {}

uint32_t LearningModelSessionOptions::BatchSizeOverride() {
  return batch_size_override_;
}

void LearningModelSessionOptions::BatchSizeOverride(uint32_t value) {
  batch_size_override_ = value;
}

bool LearningModelSessionOptions::CloseModelOnSessionCreation() {
  return close_model_on_session_creation_;
}

void LearningModelSessionOptions::CloseModelOnSessionCreation(bool value) {
  close_model_on_session_creation_ = value;
}
}  // namespace WINMLP
