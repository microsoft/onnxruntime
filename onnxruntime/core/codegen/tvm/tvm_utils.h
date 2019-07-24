// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/tvm.h>

#include "core/framework/data_types.h"

namespace onnxruntime {

constexpr const char* TVM_STACKVM = "TvmStackVm";

namespace tvm_codegen {
  // Helper function that converts a onnxruntime MLDataType to TVM DLDataType
  DLDataType ToTvmDLDataType(MLDataType ml_type);
}  //  namespace tvm
}  //  namespace onnxruntime
