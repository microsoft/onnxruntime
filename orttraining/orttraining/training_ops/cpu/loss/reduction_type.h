// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/common/exceptions.h"
#endif

namespace onnxruntime {

enum class ReductionType {
  MEAN = 0,
  SUM = 1,
  NONE = 2
};

inline ReductionType StringToReductionType(const std::string& str) {
  if (str == "mean") {
    return ReductionType::MEAN;
  }
  if (str == "sum") {
    return ReductionType::SUM;
  }
  if (str == "none") {
    return ReductionType::NONE;
  }
  ORT_ENFORCE(false, "Unknown ReductionType String");
}

}  // namespace onnxruntime
