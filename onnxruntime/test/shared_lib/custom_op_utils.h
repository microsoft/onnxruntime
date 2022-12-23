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
  MyCustomKernel(const OrtApi& ort_api, const OrtKernelInfo* /*info*/)
      : ort_(ort_api) {
  }

  void Compute(OrtKernelContext* context);

 private:
  const OrtApi& ort_;
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
  explicit MyCustomOp(const char* provider) : provider_(provider) {}
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { return new MyCustomKernel(api, info); };
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
  MyCustomKernelMultipleDynamicInputs(const OrtApi& ort_api, const OrtKernelInfo* /*info*/)
      : ort_(ort_api) {
  }

  void Compute(OrtKernelContext* context);

 private:
  const OrtApi& ort_;
};

struct MyCustomOpMultipleDynamicInputs : Ort::CustomOpBase<MyCustomOpMultipleDynamicInputs, MyCustomKernelMultipleDynamicInputs> {
  explicit MyCustomOpMultipleDynamicInputs(const char* provider) : provider_(provider) {}
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new MyCustomKernelMultipleDynamicInputs(api, info);
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

// Custom kernel that outputs the lengths of all input strings.
struct MyCustomStringLengthsKernel {
  explicit MyCustomStringLengthsKernel(const OrtKernelInfo* /* info */) {}

  void Compute(OrtKernelContext* context);
};

// Utility function to be used when testing with MyCustomStringLengthsKernel.
// Creates an input tensor from the provided input string and adds it to `ort_inputs`.
// Also initializes the corresponding expected output and I/O names.
void AddInputForCustomStringLengthsKernel(std::string input_str, OrtAllocator* allocator,
                                          std::vector<Ort::Value>& ort_inputs, std::vector<std::string>& input_names,
                                          std::vector<std::string>& output_names,
                                          std::vector<std::vector<int64_t>>& expected_dims,
                                          std::vector<std::vector<int64_t>>& expected_outputs);

// Custom kernel that echos input arguments (shape [1]) in reversed order.
// Used to test variadic custom ops with heterogenous input types.
struct MyCustomEchoReversedArgsKernel {
  explicit MyCustomEchoReversedArgsKernel(const OrtKernelInfo* /* info */) {}
  void Compute(OrtKernelContext* context);
};

// Utility custom op class that can be configured with a kernel class (T) and input/output
// configurations.
template <typename T>
struct TemplatedCustomOp : Ort::CustomOpBase<TemplatedCustomOp<T>, T> {
  TemplatedCustomOp(const char* op_name, std::vector<ONNXTensorElementDataType> input_types,
                    std::vector<OrtCustomOpInputOutputCharacteristic> input_characs, int input_min_arity,
                    bool input_homogeneity, std::vector<ONNXTensorElementDataType> output_types,
                    std::vector<OrtCustomOpInputOutputCharacteristic> output_characs, int output_min_arity,
                    bool output_homogeneity)
      : op_name_(op_name), input_types_(std::move(input_types)),
      input_characs_(std::move(input_characs)), input_min_arity_(input_min_arity),
      input_homogeneity_(input_homogeneity), output_types_(std::move(output_types)),
      output_characs_(std::move(output_characs)), output_min_arity_(output_min_arity),
      output_homogeneity_(output_homogeneity) {}

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const {
    return new T(info);
  }

  const char* GetName() const noexcept { return op_name_; }

  size_t GetInputTypeCount() const noexcept { return input_types_.size(); }

  ONNXTensorElementDataType GetInputType(size_t index) const noexcept {
    return input_types_[index];
  }

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const noexcept {
    return input_characs_[index];
  }

  int GetVariadicInputMinArity() const noexcept {
    return input_min_arity_;
  }

  bool GetVariadicInputHomogeneity() const noexcept {
    return input_homogeneity_;
  }

  size_t GetOutputTypeCount() const noexcept { return output_types_.size(); }

  ONNXTensorElementDataType GetOutputType(size_t index) const noexcept {
    return output_types_[index];
  }

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t index) const noexcept {
    return output_characs_[index];
  }

  int GetVariadicOutputMinArity() const noexcept {
    return output_min_arity_;
  }

  bool GetVariadicOutputHomogeneity() const noexcept {
    return output_homogeneity_;
  }

 private:
  const char* op_name_;

  std::vector<ONNXTensorElementDataType> input_types_;
  std::vector<OrtCustomOpInputOutputCharacteristic> input_characs_;
  int input_min_arity_;
  bool input_homogeneity_;

  std::vector<ONNXTensorElementDataType> output_types_;
  std::vector<OrtCustomOpInputOutputCharacteristic> output_characs_;
  int output_min_arity_;
  bool output_homogeneity_;
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

// Slice array of floats or doubles between [from, to) and save to output
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
  StandaloneCustomKernel(const OrtKernelInfo* info);
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
  explicit StandaloneCustomOp(const char* provider) : provider_(provider) {}

  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const { return new StandaloneCustomKernel(info); };
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
};
