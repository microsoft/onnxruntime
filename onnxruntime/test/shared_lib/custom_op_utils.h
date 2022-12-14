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

// Custom kernel that outputs the lengths of all input strings.
struct MyCustomStringLengthsKernel {
  explicit MyCustomStringLengthsKernel(const OrtKernelInfo* /* info */) {}

  void Compute(OrtKernelContext* context);
};

// Utility function to be used when testing with MyCustomStringLengthsKernel.
// Creates an input tensor from the provided input string and adds it to `ort_inputs`.
// Also initializes the correspoinding expected output and I/O names.
void AddInputForCustomStringLengthsKernel(std::string input_str, OrtAllocator* allocator,
                                          std::vector<Ort::Value>& ort_inputs, std::vector<std::string>& input_names,
                                          std::vector<std::string>& output_names,
                                          std::vector<std::vector<int64_t>>& expected_dims,
                                          std::vector<std::vector<int64_t>>& expected_outputs);

// Custom op with 1 variadic input (string) and 1 variadic output (int64_t)
// The input and output minimum arity can be configured to test for enforcement.
struct MyCustomOpWithVariadicIO : Ort::CustomOpBase<MyCustomOpWithVariadicIO, MyCustomStringLengthsKernel> {
  MyCustomOpWithVariadicIO(int input_min_arity, int output_min_arity) : input_min_arity_(input_min_arity),
                                                                        output_min_arity_(output_min_arity) {}

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const {
    return new MyCustomStringLengthsKernel(info);
  }
  constexpr const char* GetName() const noexcept { return "VariadicNode"; }

  constexpr size_t GetInputTypeCount() const noexcept { return 1; }
  constexpr ONNXTensorElementDataType GetInputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }
  constexpr OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t /* index */) const noexcept {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }

  constexpr size_t GetOutputTypeCount() const noexcept { return 1; }
  constexpr ONNXTensorElementDataType GetOutputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  };
  constexpr OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const noexcept {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }

  int GetVariadicInputMinArity() const noexcept {
    return input_min_arity_;  // At least one variadic arg.
  }

  constexpr bool GetVariadicInputHomogeneity() const noexcept {
    return true;  // All inputs are of the same type.
  }

  int GetVariadicOutputMinArity() const noexcept {
    return output_min_arity_;  // At least one variadic output value.
  }

  constexpr bool GetVariadicOutputHomogeneity() const noexcept {
    return true;  // All outputs are of the same type.
  }

 private:
  int input_min_arity_;
  int output_min_arity_;
};

// Custom op with 2 inputs (required, variadic) and 2 outputs (required, variadic)
struct MyCustomOpWithMixedVariadicIO : Ort::CustomOpBase<MyCustomOpWithMixedVariadicIO, MyCustomStringLengthsKernel> {
  MyCustomOpWithMixedVariadicIO() = default;

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const {
    return new MyCustomStringLengthsKernel(info);
  }
  constexpr const char* GetName() const noexcept { return "VariadicNode"; }

  constexpr size_t GetInputTypeCount() const noexcept { return 2; }
  constexpr ONNXTensorElementDataType GetInputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const noexcept {
    return index == 0 ? OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED :
                        OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }

  constexpr size_t GetOutputTypeCount() const noexcept { return 2; }
  constexpr ONNXTensorElementDataType GetOutputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  };
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t index) const noexcept {
    return index == 0 ? OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED :
                        OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }
};

// An empty kernel to use with invalid custom op classes.
struct MyStubKernel {
  explicit MyStubKernel(const OrtKernelInfo* /* info */) {}
  void Compute(OrtKernelContext* /* context */) {}
};

// Custom op with an invalid variadic input specification.
struct MyInvalidVariadicInputCustomOp : Ort::CustomOpBase<MyInvalidVariadicInputCustomOp, MyStubKernel> {
  MyInvalidVariadicInputCustomOp() = default;

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const { return new MyStubKernel(info); }
  constexpr const char* GetName() const noexcept { return "VariadicNode"; }

  constexpr size_t GetInputTypeCount() const noexcept { return 2; }
  constexpr ONNXTensorElementDataType GetInputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const noexcept {
    // Incorrectly specify that the first input is variadic. This will generate an error because only the last
    // input can be marked variadic.
    return index == 0 ? OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC :
                        OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

  constexpr size_t GetOutputTypeCount() const noexcept { return 1; };
  constexpr ONNXTensorElementDataType GetOutputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  };
  constexpr OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const noexcept {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }
};

// Custom op with an invalid variadic output specification.
struct MyInvalidVariadicOutputCustomOp : Ort::CustomOpBase<MyInvalidVariadicOutputCustomOp, MyStubKernel> {
  MyInvalidVariadicOutputCustomOp() = default;

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const { return new MyStubKernel(info); }
  constexpr const char* GetName() const noexcept { return "VariadicNode"; }

  constexpr size_t GetInputTypeCount() const noexcept { return 1; }
  constexpr ONNXTensorElementDataType GetInputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }
  constexpr OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t /* index */) const noexcept {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }

  constexpr size_t GetOutputTypeCount() const noexcept { return 2; }
  constexpr ONNXTensorElementDataType GetOutputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t index) const noexcept {
    // Incorrectly specify that the first output is variadic. This will generate an error because only the last
    // output can be marked variadic.
    return index == 0 ? OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC :
                        OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }
};

// Stub custom op that specifies a homogeneous variadic input.
// Used to test the enforcement of input homogeneity (should error if used in a model with heterogeneous input types)
struct StubCustomOpWithHomogeneousVariadicInput : Ort::CustomOpBase<StubCustomOpWithHomogeneousVariadicInput, MyStubKernel> {
  StubCustomOpWithHomogeneousVariadicInput() = default;

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const {
    return new MyStubKernel(info);
  }
  constexpr const char* GetName() const noexcept { return "VariadicNode"; }

  constexpr size_t GetInputTypeCount() const noexcept { return 1; }
  constexpr ONNXTensorElementDataType GetInputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  constexpr OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t /* index */) const noexcept {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }

  constexpr bool GetVariadicInputHomogeneity() const noexcept {
    return true;
  }

  constexpr size_t GetOutputTypeCount() const noexcept { return 1; }
  constexpr ONNXTensorElementDataType GetOutputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
  constexpr OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const noexcept {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }

  constexpr bool GetVariadicOutputHomogeneity() const noexcept {
    return false;  // Not all outputs are of the same type.
  }
};

// Custom kernel that echos input arguments (shape [1]) in reversed order.
// Used to test variadic custom ops with heterogenous input types.
struct MyCustomEchoReversedArgsKernel {
  explicit MyCustomEchoReversedArgsKernel(const OrtKernelInfo* /* info */) {}
  void Compute(OrtKernelContext* context);
};

// Custom op with 1 variadic input (undefined elem type) and 1 variadic output (undefined elem type)
struct MyCustomOpWithVariadicUndefIO : Ort::CustomOpBase<MyCustomOpWithVariadicUndefIO, MyCustomEchoReversedArgsKernel> {
  MyCustomOpWithVariadicUndefIO() = default;

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const {
    return new MyCustomEchoReversedArgsKernel(info);
  }
  constexpr const char* GetName() const noexcept { return "VariadicNode"; }

  constexpr size_t GetInputTypeCount() const noexcept { return 1; }
  constexpr ONNXTensorElementDataType GetInputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  constexpr OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t /* index */) const noexcept {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }

  constexpr size_t GetOutputTypeCount() const noexcept { return 1; }
  constexpr ONNXTensorElementDataType GetOutputType(size_t /*index*/) const noexcept {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
  constexpr OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const noexcept {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }

  constexpr int GetVariadicInputMinArity() const noexcept {
    return 1;
  }

  constexpr bool GetVariadicInputHomogeneity() const noexcept {
    return false;  // Not all inputs are of the same type.
  }

  constexpr int GetVariadicOutputMinArity() const noexcept {
    return 1;
  }

  constexpr bool GetVariadicOutputHomogeneity() const noexcept {
    return false;  // Not all outputs are of the same type.
  }
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
