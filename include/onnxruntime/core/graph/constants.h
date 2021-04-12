// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

constexpr const char* kNoOp = "NoOp";
constexpr const char* kConstant = "Constant";
constexpr const char* kFunctionOp = "_kFunctionOp";
constexpr const char* kConstantValue = "value";
constexpr const char* kOnnxDomain = "";
constexpr const char* kOnnxDomainAlias = "ai.onnx";
constexpr const char* kMLDomain = "ai.onnx.ml";
constexpr const char* kMSDomain = "com.microsoft";
constexpr const char* kMSExperimentalDomain = "com.microsoft.experimental";
constexpr const char* kMSNchwcDomain = "com.microsoft.nchwc";
constexpr const char* kMSFeaturizersDomain = "com.microsoft.mlfeaturizers";
constexpr const char* kMSDmlDomain = "com.microsoft.dml";
constexpr const char* kNGraphDomain = "com.intel.ai";
constexpr const char* kMIGraphXDomain = "";
constexpr const char* kVitisAIDomain = "com.xilinx";

constexpr const char* kCpuExecutionProvider = "CPUExecutionProvider";
constexpr const char* kCudaExecutionProvider = "CUDAExecutionProvider";
constexpr const char* kDnnlExecutionProvider = "DnnlExecutionProvider";
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
constexpr const char* kRocmExecutionProvider = "ROCMExecutionProvider";
constexpr const char* kCoreMLExecutionProvider = "CoreMLExecutionProvider";

}  // namespace onnxruntime
