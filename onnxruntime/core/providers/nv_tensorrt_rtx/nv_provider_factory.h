// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {
struct ProviderInfo_Nv {
  virtual OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) = 0;

 protected:
  ~ProviderInfo_Nv() = default;  // Can only be destroyed through a subclass instance
};
}  // namespace onnxruntime
