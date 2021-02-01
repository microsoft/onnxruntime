// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"

struct Input {
  const char* name = nullptr;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct MyCustomKernel {
  MyCustomKernel(Ort::CustomOpApi ort, const OrtKernelInfo* /*info*/) : ort_(ort) {
  }

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;
};

struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomKernel> {
  explicit MyCustomOp(const char* provider) : provider_(provider) {}

  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new MyCustomKernel(api, info); };
  const char* GetName() const { return "Foo"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

 private:
  const char* provider_;
};
