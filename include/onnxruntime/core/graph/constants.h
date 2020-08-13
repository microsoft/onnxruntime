// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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
constexpr const char* kMSNchwcDomain = "com.microsoft.nchwc";
constexpr const char* kMSFeaturizersDomain = "com.microsoft.mlfeaturizers";
constexpr const char* kMSDmlDomain = "com.microsoft.dml";
constexpr const char* kNGraphDomain = "com.intel.ai";
constexpr const char* kMIGraphXDomain = "";
constexpr const char* kVitisAIDomain = "com.xilinx";
constexpr const char* kCpuExecutionProvider = "CPUExecutionProvider";
constexpr const char* kCudaExecutionProvider = "CUDAExecutionProvider";
constexpr const char* kDnnlExecutionProvider = "DnnlExecutionProvider";
constexpr const char* kNGraphExecutionProvider = "NGRAPHExecutionProvider";
constexpr const char* kOpenVINOExecutionProvider = "OpenVINOExecutionProvider";
constexpr const char* kNupharExecutionProvider = "NupharExecutionProvider";
constexpr const char* kVitisAIExecutionProvider = "VitisAIExecutionProvider";
constexpr const char* kTensorrtExecutionProvider = "TensorrtExecutionProvider";
constexpr const char* kNnapiExecutionProvider = "NnapiExecutionProvider";
constexpr const char* kRknpuExecutionProvider = "RknpuExecutionProvider";
constexpr const char* kDmlExecutionProvider = "DmlExecutionProvider";
constexpr const char* kMIGraphXExecutionProvider = "MIGraphXExecutionProvider";
constexpr const char* kAclExecutionProvider = "ACLExecutionProvider";
constexpr const char* kArmNNExecutionProvider = "ArmNNExecutionProvider";
constexpr const char *providers_available[] = {
  kCpuExecutionProvider,
#ifdef USE_CUDA
  kCudaExecutionProvider,
#endif
#ifdef USE_DNNL
  kDnnlExecutionProvider,
#endif
#ifdef USE_NGRAPH
  kNGraphExecutionProvider,
#endif
#ifdef USE_OPENVINO
  kOpenVINOExecutionProvider,
#endif
#ifdef USE_NUPHAR
  kNupharExecutionProvider,
#endif
#ifdef USE_VITISAI
  kVitisAIExecutionProvider,
#endif
#ifdef USE_TENSORRT
  kTensorrtExecutionProvider,
#endif
#ifdef USE_NNAPI
  kNnapiExecutionProvider,
#endif
#ifdef USE_RKNPU
  kRknpuExecutionProvider,
#endif
#ifdef USE_DML
  kDmlExecutionProvider,
#endif
#ifdef USE_MIGRAPHX
  kMIGraphXExecutionProvider,
#endif
#ifdef USE_ACL
  kAclExecutionProvider,
#endif
#ifdef USE_ARMNN
  kArmNNExecutionProvider,
#endif
};
}  // namespace onnxruntime
