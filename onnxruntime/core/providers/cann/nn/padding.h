// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/common.h"

namespace onnxruntime {
namespace cann {

constexpr const char* GetConvAutoPadMode(AutoPadType auto_pad) {
  switch (auto_pad) {
    case AutoPadType::NOTSET:
      return "NOTSET";
    case AutoPadType::VALID:
      return "VALID";
    case AutoPadType::SAME_UPPER:
      return "SAME_UPPER";
    case AutoPadType::SAME_LOWER:
      return "SAME_LOWER";
  }

  return nullptr;
}

constexpr const char* GetPoolAutoPadMode(AutoPadType auto_pad) {
  switch (auto_pad) {
    case AutoPadType::NOTSET:
      return "CALCULATED";
    case AutoPadType::VALID:
      return "VALID";
    case AutoPadType::SAME_UPPER:
      return "SAME";
    case AutoPadType::SAME_LOWER:
      return "SAME";
  }

  return nullptr;
}

}  // namespace cann
}  // namespace onnxruntime
