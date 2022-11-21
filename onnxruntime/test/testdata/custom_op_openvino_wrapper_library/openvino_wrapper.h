// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#ifdef _MSC_VER
#pragma warning(disable: 4100)
#endif
#include <openvino/openvino.hpp>
#ifdef _MSC_VER
#pragma warning(default : 4100)
#endif

#include <string>

struct KernelOpenVINO {
  KernelOpenVINO(const OrtApi& api, const OrtKernelInfo* info,
                 const std::unordered_map<std::string, std::string>& session_configs);

  void Compute(OrtKernelContext* context);

 private:
  ov::CompiledModel compiled_model_;
  ov::OutputVector ov_inputs_;
  ov::OutputVector ov_outputs_;
  std::string weights_;
  std::string device_type_;
};

struct CustomOpOpenVINO : Ort::CustomOpBase<CustomOpOpenVINO, KernelOpenVINO> {
  CustomOpOpenVINO(Ort::ConstSessionOptions session_options) : session_options_(session_options){};

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t index) const;
  const char* GetExecutionProviderType() const;
  std::vector<std::string> GetSessionConfigKeys() const;

 private:
  Ort::ConstSessionOptions session_options_;
};
