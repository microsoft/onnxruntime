// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <string>

#include "core/framework/arena_extend_strategy.h"
#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
struct CANNExecutionProviderInfo {
  OrtDevice::DeviceId device_id{0};
  size_t npu_mem_limit{std::numeric_limits<size_t>::max()};
  ArenaExtendStrategy arena_extend_strategy{ArenaExtendStrategy::kNextPowerOfTwo};
  bool enable_cann_graph{true};
  bool dump_graphs{false};
  std::string precision_mode;
  std::string op_select_impl_mode;
  std::string optypelist_for_implmode;
  OrtArenaCfg* default_memory_arena_cfg{nullptr};

  static CANNExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  static ProviderOptions ToProviderOptions(const CANNExecutionProviderInfo& info);
  static ProviderOptions ToProviderOptions(const OrtCANNProviderOptions& info);
};

}  // namespace onnxruntime
