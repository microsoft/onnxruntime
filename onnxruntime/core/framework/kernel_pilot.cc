// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_pilot.h"

namespace onnxruntime {

IKernelPilotMoeExpertSelection& KernelPilot::Moe() noexcept {
  return moe_;
}

const IKernelPilotMoeExpertSelection& KernelPilot::Moe() const noexcept {
  return moe_;
}

}  // namespace onnxruntime
