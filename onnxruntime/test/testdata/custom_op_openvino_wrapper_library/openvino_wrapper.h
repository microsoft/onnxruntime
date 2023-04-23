// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#ifdef _MSC_VER
#pragma warning(disable : 4100)
#endif
#include <openvino/openvino.hpp>
#ifdef _MSC_VER
#pragma warning(default : 4100)
#endif

#include <string>

struct KernelOpenVINO {
  KernelOpenVINO(const OrtApi& api, const OrtKernelInfo* info,
                 const std::unordered_map<std::string, std::string>& session_configs);

  KernelOpenVINO(const KernelOpenVINO&) = delete;
  KernelOpenVINO& operator=(const KernelOpenVINO&) = delete;

  ~KernelOpenVINO() = default;

  void Compute(OrtKernelContext* context);

 private:
  ov::CompiledModel compiled_model_;
  ov::OutputVector ov_inputs_;
  ov::OutputVector ov_outputs_;
  Ort::Value weights_;
  std::string device_type_;
  Ort::Logger logger_;
};

struct CustomOpOpenVINO : Ort::CustomOpBase<CustomOpOpenVINO, KernelOpenVINO> {
  explicit CustomOpOpenVINO(Ort::ConstSessionOptions session_options);

  CustomOpOpenVINO(const CustomOpOpenVINO&) = delete;
  CustomOpOpenVINO& operator=(const CustomOpOpenVINO&) = delete;

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;

  constexpr const char* GetName() const noexcept {
    return "OpenVINO_Wrapper";
  }

  constexpr const char* GetExecutionProviderType() const noexcept {
    return "CPUExecutionProvider";
  }

  constexpr size_t GetInputTypeCount() const noexcept {
    return 1;
  }

  constexpr size_t GetOutputTypeCount() const noexcept {
    return 1;
  }

  constexpr ONNXTensorElementDataType GetInputType(size_t /* index */) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  constexpr ONNXTensorElementDataType GetOutputType(size_t /* index */) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  constexpr OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t /* index */) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }

  constexpr OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /* index */) const noexcept {
    return INPUT_OUTPUT_VARIADIC;
  }

  constexpr bool GetVariadicInputHomogeneity() const noexcept {
    return false;  // heterogenous
  }

  constexpr bool GetVariadicOutputHomogeneity() const noexcept {
    return false;  // heterogeneous
  }

  std::vector<std::string> GetSessionConfigKeys() const;

 private:
  std::unordered_map<std::string, std::string> session_configs_;
};
