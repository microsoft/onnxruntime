// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

struct Input {
  const char* name = nullptr;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

struct MyCustomKernel {
  MyCustomKernel(const OrtKernelInfo* /*info*/, void* compute_stream)
      : compute_stream_(compute_stream) {
  }

  void Compute(OrtKernelContext* context);

 private:
  void* compute_stream_;
};

struct MyCustomKernelSecondInputOnCpu {
  MyCustomKernelSecondInputOnCpu(const OrtKernelInfo* /*info*/, void* compute_stream)
      : compute_stream_(compute_stream) {
  }

  void Compute(OrtKernelContext* context);

 private:
  void* compute_stream_;
};

struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomKernel> {
  explicit MyCustomOp(const char* provider, void* compute_stream) : provider_(provider), compute_stream_(compute_stream) {}

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const { return new MyCustomKernel(info, compute_stream_); };
  const char* GetName() const { return "Foo"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    // Both the inputs need to be necessarily of float type
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

 private:
  const char* provider_{"CPUExecutionProvider"};
  void* compute_stream_;
};

struct MyCustomOpSecondInputOnCpu : Ort::CustomOpBase<MyCustomOpSecondInputOnCpu, MyCustomKernelSecondInputOnCpu> {
  explicit MyCustomOpSecondInputOnCpu(const char* provider, void* compute_stream) : provider_(provider), compute_stream_(compute_stream) {}

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const { return new MyCustomKernelSecondInputOnCpu(info, compute_stream_); };
  const char* GetName() const { return "Foo"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    // Both the inputs need to be necessarily of float type
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  OrtMemType GetInputMemoryType(size_t i) const {
    if (i == 1) { return OrtMemTypeCPUInput; }
    return OrtMemTypeDefault;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

 private:
  const char* provider_{"CUDAExecutionProvider"};
  void* compute_stream_;
};

struct MyCustomKernelMultipleDynamicInputs {
  MyCustomKernelMultipleDynamicInputs(const OrtKernelInfo* /*info*/, void* compute_stream)
      : compute_stream_(compute_stream) {
  }

  void Compute(OrtKernelContext* context);

 private:
  void* compute_stream_;
};

struct MyCustomOpMultipleDynamicInputs : Ort::CustomOpBase<MyCustomOpMultipleDynamicInputs, MyCustomKernelMultipleDynamicInputs> {
  explicit MyCustomOpMultipleDynamicInputs(const char* provider, void* compute_stream) : provider_(provider), compute_stream_(compute_stream) {}

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const { 
    return new MyCustomKernelMultipleDynamicInputs(info, compute_stream_); 
  };
  const char* GetName() const { return "Foo"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    // Both the inputs are dynamic typed (i.e.) they can be any type and need not be
    // homogeneous
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

 private:
  const char* provider_;
  void* compute_stream_;
};

struct MyCustomKernelWithOptionalInput {
  MyCustomKernelWithOptionalInput(const OrtKernelInfo* /*info*/) {
  }
  void Compute(OrtKernelContext* context);
};

struct MyCustomOpWithOptionalInput : Ort::CustomOpBase<MyCustomOpWithOptionalInput, MyCustomKernelWithOptionalInput> {
  explicit MyCustomOpWithOptionalInput(const char* provider) : provider_(provider) {}

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const { return new MyCustomKernelWithOptionalInput(info); };
  const char* GetName() const { return "FooBar"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
    // The second input (index == 1) is optional
    if (index == 1)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

 private:
  const char* provider_;
};

struct MyCustomKernelWithAttributes {
  MyCustomKernelWithAttributes(const OrtKernelInfo* kernel_info) {
    Ort::ConstKernelInfo info{kernel_info};
    int_attr_ = info.GetAttribute<int64_t>("int_attr");
    float_attr_ = info.GetAttribute<float>("float_attr");

    ints_attr_ = info.GetAttributes<int64_t>("ints_attr");
    floats_attr_ = info.GetAttributes<float>("floats_attr");

    string_arr_ = info.GetAttribute<std::string>("string_attr");
  }

  void Compute(OrtKernelContext* context);

 private:
  int64_t int_attr_;
  float float_attr_;

  std::vector<int64_t> ints_attr_;
  std::vector<float> floats_attr_;

  std::string string_arr_;
};

struct MyCustomOpWithAttributes : Ort::CustomOpBase<MyCustomOpWithAttributes, MyCustomKernelWithAttributes> {
  explicit MyCustomOpWithAttributes(const char* provider) : provider_(provider) {}

  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const { return new MyCustomKernelWithAttributes(info); };
  const char* GetName() const { return "FooBar_Attr"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

 private:
  const char* provider_;
};

//Slice array of floats or doubles between [from, to) and save to output
struct SliceCustomOpKernel {
  SliceCustomOpKernel(const OrtKernelInfo* /*info*/) {
  }

  void Compute(OrtKernelContext* context);
};

struct SliceCustomOp : Ort::CustomOpBase<SliceCustomOp, SliceCustomOpKernel> {
  explicit SliceCustomOp(const char* provider) : provider_(provider) {}
  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const {
    return new SliceCustomOpKernel(info);
  };

  const char* GetName() const { return "Slice"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;  // input array of float or double
    else if (index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;  // slice from

    // index 2 (keep compiler happy on Linux)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;  // slice to
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

 private:
  const char* provider_;
};

struct StandaloneCustomKernel {
  StandaloneCustomKernel(const OrtKernelInfo* info, void*);
  ~StandaloneCustomKernel();
  void Compute(OrtKernelContext* context);

 private:
  void InitTopK();
  void InvokeTopK(OrtKernelContext* context);

  void InitGru();
  void InvokeGru(OrtKernelContext* context);

  void InitInvokeConv(OrtKernelContext* context);  // create Conv and invoke in Compute(...)

  Ort::KernelInfo info_copy_{nullptr};
  Ort::Op op_add_{nullptr};
  Ort::Op op_topk_{nullptr};
  Ort::Op op_gru_{nullptr};
};

struct StandaloneCustomOp : Ort::CustomOpBase<StandaloneCustomOp, StandaloneCustomKernel> {
  explicit StandaloneCustomOp(const char* provider, void* compute_stream) : provider_(provider), compute_stream_(compute_stream) {}

  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const { return new StandaloneCustomKernel(info, compute_stream_); };
  const char* GetName() const { return "Foo"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

 private:
  const char* provider_;
  void* compute_stream_;
};
