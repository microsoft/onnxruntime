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

STDMETHODIMP LearningModelSessionOptions::GetIntraOpNumThreads(uint32_t* numThreads) noexcept {
  if (numThreads == NULL) return E_INVALIDARG;
  else
  {
    *numThreads = intra_op_num_threads_override_;
    return S_OK;
  }
}

STDMETHODIMP LearningModelSessionOptions::OverrideIntraOpNumThreads(uint32_t intraOpNumThreads) noexcept {
  intra_op_num_threads_override_ = intraOpNumThreads;
  return S_OK;
}
}  // namespace WINMLP