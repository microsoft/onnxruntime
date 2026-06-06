// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

#include "core/common/common.h"
#include "core/framework/config_options.h"

namespace onnxruntime {

enum class CpuAttentionImpl {
  kUnfused,
  kFlashSpecialized,
  kFlashFlex,
  kAuto,
};

struct CpuAttentionSelection {
  CpuAttentionImpl impl = CpuAttentionImpl::kAuto;
  bool strict = false;
};

Status ResolveCpuAttentionSelection(const ConfigOptions& config_options,
                                    CpuAttentionSelection& selection);

std::string_view CpuAttentionImplToString(CpuAttentionImpl impl);

}  // namespace onnxruntime
