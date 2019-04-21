// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/common.h"

namespace onnxruntime {
constexpr const char* kNoOp = "NoOp";
constexpr const char* kConstant = "Constant";
constexpr const char* kFunctionOp = "_kFunctionOp";
constexpr const char* kConstantValue = "value";
constexpr const char* kOnnxDomain = "";
constexpr const char* kOnnxDomainAlias = "ai.onnx";
constexpr const char* kMLDomain = "ai.onnx.ml";
constexpr const char* kMSDomain = "com.microsoft";
constexpr const char* kNGraphDomain = "com.intel.ai";
constexpr const char* kCpuExecutionProvider = "CPUExecutionProvider";
constexpr const char* kCudaExecutionProvider = "CUDAExecutionProvider";
constexpr const char* kMklDnnExecutionProvider = "MKLDNNExecutionProvider";
constexpr const char* kNGraphExecutionProvider = "NGRAPHExecutionProvider";
constexpr const char* kNupharExecutionProvider = "NupharExecutionProvider";
constexpr const char* kBrainSliceExecutionProvider = "BrainSliceExecutionProvider";
constexpr const char* kTensorrtExecutionProvider = "TensorrtExecutionProvider";
}  // namespace onnxruntime

