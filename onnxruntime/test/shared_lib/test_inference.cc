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
using namespace onnxruntime;

void RunSession(OrtAllocator* env, OrtSession* session_object,
                const std::vector<size_t>& dims_x,
                const std::vector<float>& values_x,
                const std::vector<int64_t>& dims_y,
                const std::vector<float>& values_y,
                OrtValue* output_tensor) {
  std::unique_ptr<OrtValue, decltype(&OrtReleaseValue)> value_x(nullptr, OrtReleaseValue);
  std::vector<OrtValue*> inputs(1);
  inputs[0] = OrtCreateTensorAsOrtValue(env, dims_x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  value_x.reset(inputs[0]);
  void* raw_data;
  ORT_THROW_ON_ERROR(OrtGetTensorMutableData(inputs[0], &raw_data));
  memcpy(raw_data, values_x.data(), values_x.size() * sizeof(values_x[0]));
  std::vector<const char*> input_names{"X"};
  const char* output_names[] = {"Y"};
  bool is_output_allocated_by_ort = output_tensor == nullptr;
  OrtValue* old_output_ptr = output_tensor;
  ORT_THROW_ON_ERROR(OrtRun(session_object, NULL, input_names.data(), inputs.data(), inputs.size(), output_names, 1, &output_tensor));
  ASSERT_NE(output_tensor, nullptr);
  if (!is_output_allocated_by_ort)
    ASSERT_EQ(output_tensor, old_output_ptr);
  std::unique_ptr<OrtTensorTypeAndShapeInfo> shape_info;
  {
    OrtTensorTypeAndShapeInfo* shape_info_ptr;
    ORT_THROW_ON_ERROR(OrtGetTensorShapeAndType(output_tensor, &shape_info_ptr));
    shape_info.reset(shape_info_ptr);
  }
  size_t rtensor_dims = OrtGetNumOfDimensions(shape_info.get());
  std::vector<int64_t> shape_array(rtensor_dims);
  OrtGetDimensions(shape_info.get(), shape_array.data(), shape_array.size());
  ASSERT_EQ(shape_array, dims_y);
  size_t total_len = 1;
  for (size_t i = 0; i != rtensor_dims; ++i) {
    total_len *= shape_array[i];
  }
  ASSERT_EQ(values_y.size(), total_len);
  float* f;
  ORT_THROW_ON_ERROR(OrtGetTensorMutableData(output_tensor, (void**)&f));
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }
  if (is_output_allocated_by_ort) OrtReleaseValue(output_tensor);
}

template <typename T>
void TestInference(OrtEnv* env, T model_uri,
                   const std::vector<size_t>& dims_x,
                   const std::vector<float>& values_x,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<float>& expected_values_y,
                   int provider_type, bool custom_op) {
  SessionOptionsWrapper sf(env);

  if (provider_type == 1) {
#ifdef USE_CUDA
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, 0));
    std::cout << "Running simple inference with cuda provider" << std::endl;
#else
    return;
#endif
  } else if (provider_type == 2) {
#ifdef USE_MKLDNN
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(sf, 1));
    std::cout << "Running simple inference with mkldnn provider" << std::endl;
#else
    return;
#endif
  } else if (provider_type == 3) {
#ifdef USE_NUPHAR
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Nuphar(sf, 0, ""));
    std::cout << "Running simple inference with nuphar provider" << std::endl;
#else
    return;
#endif
  } else {
    std::cout << "Running simple inference with default provider" << std::endl;
  }
  if (custom_op) {
    sf.AppendCustomOpLibPath("libonnxruntime_custom_op_shared_lib_test.so");
  }
  std::unique_ptr<OrtSession, decltype(&OrtReleaseSession)>
      inference_session(sf.OrtCreateSession(model_uri), OrtReleaseSession);
  std::unique_ptr<MockedOrtAllocator> default_allocator(std::make_unique<MockedOrtAllocator>());
  // Now run
  //without preallocated output tensor
  RunSession(default_allocator.get(),
             inference_session.get(),
             dims_x,
             values_x,
             expected_dims_y,
             expected_values_y,
             nullptr);
  //with preallocated output tensor
  std::unique_ptr<OrtValue, decltype(&OrtReleaseValue)> value_y(nullptr, OrtReleaseValue);
  {
    std::vector<OrtValue*> allocated_outputs(1);
    std::vector<size_t> dims_y(expected_dims_y.size());
    for (size_t i = 0; i != expected_dims_y.size(); ++i) {
      dims_y[i] = static_cast<size_t>(expected_dims_y[i]);
    }

    allocated_outputs[0] =
        OrtCreateTensorAsOrtValue(default_allocator.get(), dims_y, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    value_y.reset(allocated_outputs[0]);
  }
  //test it twice
  for (int i = 0; i != 2; ++i)
    RunSession(default_allocator.get(),
               inference_session.get(),
               dims_x,
               values_x,
               expected_dims_y,
               expected_values_y,
               value_y.get());
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
  std::vector<size_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  TestInference<PATH_TYPE>(env, MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, GetParam(), false);
}

INSTANTIATE_TEST_CASE_P(CApiTestWithProviders,
                        CApiTestWithProvider,
                        ::testing::Values(0, 1, 2, 3, 4));

#ifndef _WIN32
//doesn't work, failed in type comparison
TEST_F(CApiTest, DISABLED_custom_op) {
  std::cout << "Running custom op inference" << std::endl;
  std::vector<size_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  TestInference<PATH_TYPE>(env, CUSTOM_OP_MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, false, true);
}
#endif

#ifdef ORT_RUN_EXTERNAL_ONNX_TESTS
TEST_F(CApiTest, create_session_without_session_option) {
  constexpr PATH_TYPE model_uri = TSTR("../models/opset8/test_squeezenet/model.onnx");
  OrtSession* ret;
  ORT_THROW_ON_ERROR(::OrtCreateSession(env, model_uri, nullptr, &ret));
  ASSERT_NE(nullptr, ret);
  OrtReleaseSession(ret);
}
#endif
TEST_F(CApiTest, create_tensor) {
  const char* s[] = {"abc", "kmp"};
  size_t expected_len = 2;
  std::unique_ptr<MockedOrtAllocator> default_allocator(std::make_unique<MockedOrtAllocator>());
  {
    std::unique_ptr<OrtValue, decltype(&OrtReleaseValue)> tensor(
        OrtCreateTensorAsOrtValue(default_allocator.get(), {expected_len}, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
        OrtReleaseValue);
    ORT_THROW_ON_ERROR(OrtFillStringTensor(tensor.get(), s, expected_len));
    std::unique_ptr<OrtTensorTypeAndShapeInfo> shape_info;
    {
      OrtTensorTypeAndShapeInfo* shape_info_ptr;
      ORT_THROW_ON_ERROR(OrtGetTensorShapeAndType(tensor.get(), &shape_info_ptr));
      shape_info.reset(shape_info_ptr);
    }
    size_t len = static_cast<size_t>(OrtGetTensorShapeElementCount(shape_info.get()));
    ASSERT_EQ(len, expected_len);
    std::vector<int64_t> shape_array(len);

    size_t data_len;
    ORT_THROW_ON_ERROR(OrtGetStringTensorDataLength(tensor.get(), &data_len));
    std::string result(data_len, '\0');
    std::vector<size_t> offsets(len);
    ORT_THROW_ON_ERROR(OrtGetStringTensorContent(tensor.get(), (void*)result.data(), data_len, offsets.data(),
                                                 offsets.size()));
  }
}

TEST_F(CApiTest, create_tensor_with_data) {
  float values[] = {3.0f, 1.0f, 2.f, 0.f};
  constexpr size_t values_length = sizeof(values) / sizeof(values[0]);
  OrtAllocatorInfo* info;
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &info));
  std::vector<size_t> dims = {4};
  std::unique_ptr<OrtValue, decltype(&OrtReleaseValue)> tensor(
      OrtCreateTensorWithDataAsOrtValue(info, values, values_length * sizeof(float), dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT), OrtReleaseValue);
  OrtReleaseAllocatorInfo(info);
  void* new_pointer;
  ORT_THROW_ON_ERROR(OrtGetTensorMutableData(tensor.get(), &new_pointer));
  ASSERT_EQ(new_pointer, values);
  struct OrtTypeInfo* type_info;
  ORT_THROW_ON_ERROR(OrtGetTypeInfo(tensor.get(), &type_info));
  const struct OrtTensorTypeAndShapeInfo* tensor_info = OrtCastTypeInfoToTensorInfo(type_info);
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1, OrtGetNumOfDimensions(tensor_info));
  OrtReleaseTypeInfo(type_info);
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
