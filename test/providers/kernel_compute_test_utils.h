// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_STRIDED_TENSORS

#pragma once

#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// OpTester creates a graph and run it using InferenceSession, the inputs and outputs are graph inputs and outputs,
// which cannot be strided tensors for now. This KernelComputeTester creates the input and output OrtValues
// and call the OpKernel->Compute() directly, here input and output OrtValues can be strided tensors.
class KernelComputeTester {
 public:
  explicit KernelComputeTester(const char* op, const char* provider = kCpuExecutionProvider, int opset_version = 14,
                               const char* domain = kOnnxDomain)
      : op_(op), provider_(provider), opset_version_(opset_version), domain_(domain) {
    if (opset_version_ < 0) {
      static int latest_onnx_version =
          ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange().Map().at(ONNX_NAMESPACE::ONNX_DOMAIN).second;
      opset_version_ = latest_onnx_version;
    }
  }

  struct Data {
    NodeArg def_;
    OrtValue value_;
    bool is_cpu_data_;
    Data(onnxruntime::NodeArg&& def, OrtValue&& value, bool is_cpu_data)
        : def_(std::move(def)), value_(std::move(value)), is_cpu_data_(is_cpu_data) {}
    Data(Data&&) = default;
    Data& operator=(Data&&) = default;
  };

  template <typename T>
  void AddInput(const char* name, std::initializer_list<int64_t> dims, std::initializer_list<T> values,
                std::initializer_list<int64_t> strides = {}, bool is_cpu_data = false) {
    AddData(input_data_, name, dims, values.begin(), strides, is_cpu_data);
  }

  template <typename T>
  void AddInput(const char* name, std::initializer_list<int64_t> dims, const std::vector<T>& values,
                std::initializer_list<int64_t> strides = {}, bool is_cpu_data = false) {
    AddData(input_data_, name, dims, values.data(), strides, is_cpu_data);
  }

  template <typename T>
  void AddOutput(const char* name, std::initializer_list<int64_t> dims, std::initializer_list<T> values,
                 std::initializer_list<int64_t> strides = {}) {
    AddData(output_data_, name, dims, values.begin(), strides, false);
  }

  template <typename T>
  void AddOutput(const char* name, std::initializer_list<int64_t> dims, const std::vector<T> values,
                 std::initializer_list<int64_t> strides = {}) {
    AddData(output_data_, name, dims, values.data(), strides, false);
  }

  template <typename T>
  void AddAttribute(std::string name, T value) {
    // Generate a the proper AddAttribute call for later.
    add_attribute_funcs_.emplace_back(
        [name = std::move(name), value = std::move(value)](Node& node) { node.AddAttribute(name, value); });
  }

  void Run(std::unordered_set<int> strided_outputs = {});

 private:
  template <typename T>
  void AddData(std::vector<Data>& data, const char* name, const std::vector<int64_t>& dims, const T* values,
               const std::vector<int64_t>& strides, bool is_cpu_data) {
    OrtValue value;
    TensorShape shape(dims);
    auto allocator = AllocatorManager::Instance().GetAllocator(CPU);
    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), shape, std::move(allocator), value);

    if (!strides.empty()) {
      value.GetMutable<Tensor>()->SetShapeAndStrides(shape, strides);
    }

    if (values) {
      Tensor* tensor = value.GetMutable<Tensor>();
      auto* p_data = tensor->MutableData<T>();
      memcpy(p_data, values, tensor->SizeInBytes());
    }

    TTypeProto<T> tensor_type_proto(&dims);
    auto node_arg = NodeArg(name, &tensor_type_proto.proto);
    data.emplace_back(Data(std::move(node_arg), std::move(value), is_cpu_data));
  }

  const char* op_;
  std::string provider_;
  int opset_version_;
  const char* domain_;

  std::vector<Data> input_data_;
  std::vector<Data> output_data_;
  std::vector<std::function<void(onnxruntime::Node& node)>> add_attribute_funcs_;
};

}  // namespace test
}  // namespace onnxruntime

#endif
