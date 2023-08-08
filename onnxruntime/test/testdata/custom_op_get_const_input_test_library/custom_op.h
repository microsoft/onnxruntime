// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"

struct TestCustomKernel {
  TestCustomKernel(const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);
};

struct TestCustomOp : Ort::CustomOpBase<TestCustomOp, TestCustomKernel> {
  explicit TestCustomOp();

  void* CreateKernel(const OrtApi&, const OrtKernelInfo*) const;

  const char* GetName() const { return "custom op"; };

  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; };

  size_t GetInputTypeCount() const { return 2; };

  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const { return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL; };

  size_t GetOutputTypeCount() const { return 1; };

  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const { return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL; };
};
