// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/snpe/snpe_runtime_wrapper.h"

namespace onnxruntime {
namespace contrib {
namespace snpe {

void SnpeRuntimeWrapper::Set(const std::string& runtime) {
  if (runtime.empty()) {
    return;
  }

  if (runtime == "DSP" || runtime == "DSP_FIXED8_TF") {
    runtime_ = zdl::DlSystem::Runtime_t::DSP;
    return;
  }

  if (runtime == "CPU" || runtime == "CPU_FLOAT32") {
    runtime_ = zdl::DlSystem::Runtime_t::CPU;
    return;
  }

  if (runtime == "GPU" || runtime == "GPU_FLOAT32_16_HYBRID") {
    runtime_ = zdl::DlSystem::Runtime_t::GPU;
    return;
  }

  if (runtime == "GPU_FLOAT16") {
    runtime_ = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
    return;
  }

  if (runtime == "AIP_FIXED_TF" || runtime == "AIP_FIXED8_TF") {
    runtime_ = zdl::DlSystem::Runtime_t::AIP_FIXED_TF;
    return;
  }
}

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
