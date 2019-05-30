// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "providers.h"
#include <memory>
#include <vector>
#include <iostream>
#include <atomic>
#include <gtest/gtest.h>
#include "test_allocator.h"
#include "test_fixture.h"
#include "onnx_protobuf.h"

struct Input {
  const char* name;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

void RunSession(OrtAllocator* allocator, Ort::Session& session_object,
                const std::vector<Input>& inputs,
                const char* output_name,
                const std::vector<int64_t>& dims_y,
                const std::vector<float>& values_y,
                Ort::Value* output_tensor) {
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(allocator->Info(allocator), const_cast<float*>(inputs[i].values.data()), inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }

  std::vector<Ort::Value> ort_outputs;
  if (output_tensor)
    session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(), &output_name, output_tensor, 1);
  else {
    ort_outputs = session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(), &output_name, 1);
    ASSERT_EQ(ort_outputs.size(), 1);
    output_tensor = &ort_outputs[0];
  }

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), dims_y);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), total_len);

  float* f = output_tensor->GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }
}

template <typename T>
void TestInference(Ort::Env& env, T model_uri,
                   const std::vector<Input>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<float>& expected_values_y,
                   int provider_type, OrtCustomOpDomain* custom_op_domain_ptr) {
  Ort::SessionOptions session_options;

  if (provider_type == 1) {
#ifdef USE_CUDA
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    std::cout << "Running simple inference with cuda provider" << std::endl;
#else
    return;
#endif
  } else if (provider_type == 2) {
#ifdef USE_MKLDNN
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(session_options, 1));
    std::cout << "Running simple inference with mkldnn provider" << std::endl;
#else
    return;
#endif
  } else if (provider_type == 3) {
#ifdef USE_NUPHAR
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Nuphar(session_options, 0, ""));
    std::cout << "Running simple inference with nuphar provider" << std::endl;
#else
    return;
#endif
  } else {
    std::cout << "Running simple inference with default provider" << std::endl;
  }
  if (custom_op_domain_ptr) {
    ORT_THROW_ON_ERROR(OrtAddCustomOpDomain(session_options, custom_op_domain_ptr));
  }

  Ort::Session session(env, model_uri, session_options);
  auto default_allocator = std::make_unique<MockedOrtAllocator>();
  // Now run
  //without preallocated output tensor
  RunSession(default_allocator.get(),
             session,
             inputs,
             output_name,
             expected_dims_y,
             expected_values_y,
             nullptr);
  //with preallocated output tensor
  Ort::Value value_y = Ort::Value::CreateTensor<float>(default_allocator.get(), expected_dims_y.data(), expected_dims_y.size());

  //test it twice
  for (int i = 0; i != 2; ++i)
    RunSession(default_allocator.get(),
               session,
               inputs,
               output_name,
               expected_dims_y,
               expected_values_y,
               &value_y);
}

static constexpr PATH_TYPE MODEL_URI = TSTR("testdata/mul_1.pb");
static constexpr PATH_TYPE CUSTOM_OP_MODEL_URI = TSTR("testdata/foo_1.pb");

class CApiTestWithProvider : public CApiTest,
                             public ::testing::WithParamInterface<int> {
};

// Tests that the Foo::Bar() method does Abc.
TEST_P(CApiTestWithProvider, simple) {
  // simple inference test
  // prepare inputs
  std::vector<Input> inputs(1);
  Input& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  TestInference<PATH_TYPE>(env_, MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, GetParam(), nullptr);
}

INSTANTIATE_TEST_CASE_P(CApiTestWithProviders,
                        CApiTestWithProvider,
                        ::testing::Values(0, 1, 2, 3, 4));

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

// Once we use C++17 this could be replaced with std::size
template <typename T, size_t N>
constexpr size_t countof(T (&)[N]) { return N; }

struct MyCustomKernel {
  MyCustomKernel(Ort::CustomOpApi ort, const OrtKernelInfo* /*info*/) : ort_(ort) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
    const float* X = ort_.GetTensorData<float>(input_X);
    const float* Y = ort_.GetTensorData<float>(input_Y);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    float* out = ort_.GetTensorMutableData<float>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    for (int64_t i = 0; i < size; i++) {
      out[i] = X[i] + Y[i];
    }
  }

 private:
  Ort::CustomOpApi ort_;
};

struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomKernel> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) { return new MyCustomKernel(api, info); };
  const char* GetName() const { return "Foo"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

TEST_F(CApiTest, custom_op_handler) {
  std::cout << "Running custom op inference" << std::endl;

  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  MyCustomOp custom_op;
  Ort::CustomOpDomain custom_op_domain("");
  custom_op_domain.Add(&custom_op);

  TestInference<PATH_TYPE>(env_, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0, custom_op_domain);
}

#ifdef ORT_RUN_EXTERNAL_ONNX_TESTS
TEST_F(CApiTest, create_session_without_session_option) {
  constexpr PATH_TYPE model_uri = TSTR("../models/opset8/test_squeezenet/model.onnx");
  Ort::Session ret(env_, model_uri, Ort::SessionOptions{nullptr});
  ASSERT_NE(nullptr, ret);
}
#endif

TEST_F(CApiTest, create_tensor) {
  const char* s[] = {"abc", "kmp"};
  int64_t expected_len = 2;
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  Ort::Value tensor = Ort::Value::CreateTensor(default_allocator.get(), &expected_len, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  ORT_THROW_ON_ERROR(OrtFillStringTensor(tensor, s, expected_len));
  auto shape_info = tensor.GetTensorTypeAndShapeInfo();

  int64_t len = shape_info.GetElementCount();
  ASSERT_EQ(len, expected_len);
  std::vector<int64_t> shape_array(len);

  size_t data_len = tensor.GetStringTensorDataLength();
  std::string result(data_len, '\0');
  std::vector<size_t> offsets(len);
  tensor.GetStringTensorContent((void*)result.data(), data_len, offsets.data(), offsets.size());
}

TEST_F(CApiTest, create_tensor_with_data) {
  float values[] = {3.0f, 1.0f, 2.f, 0.f};
  constexpr size_t values_length = sizeof(values) / sizeof(values[0]);

  Ort::AllocatorInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<int64_t> dims = {4};
  Ort::Value tensor = Ort::Value::CreateTensor<float>(info, values, values_length, dims.data(), dims.size());

  float* new_pointer = tensor.GetTensorMutableData<float>();
  ASSERT_EQ(new_pointer, values);

  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1, tensor_info.GetDimensionsCount());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  //TODO: Linker on Mac OS X is kind of strange. The next line of code will trigger a crash
#ifndef __APPLE__
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return ret;
}
