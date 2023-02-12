// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/tunable/rocm_tuning_context.h"

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/tuning_context.h"
#define TUNING_CONTEXT_IMPL
#include "core/framework/tuning_context_impl.h"
#undef TUNING_CONTEXT_IMPL
#include "core/providers/rocm/rocm_execution_provider.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

RocmTuningContext::RocmTuningContext(ROCMExecutionProvider*, TunableOpInfo* info) : info_(info) {}

void RocmTuningContext::EnableTunableOp() {
  LOGS_DEFAULT(INFO) << "Enable TunableOp for ROCm Execution Provider";
  info_->enabled = true;
}

void RocmTuningContext::DisableTunableOp() {
  LOGS_DEFAULT(INFO) << "Disable TunableOp for ROCm Execution Provider";
  info_->enabled = false;
}

bool RocmTuningContext::IsTunableOpEnabled() const {
  return info_->enabled;
}

TuningResultsManager& RocmTuningContext::GetTuningResultsManager() {
  return manager_;
}

const TuningResultsManager& RocmTuningContext::GetTuningResultsManager() const {
  return manager_;
}

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
