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

using namespace onnxruntime;

template <bool with_target_names>
void RunSession(ONNXRuntimeAllocator* env, ONNXSession* session_object,
                const std::vector<size_t>& dims_x,
                const std::vector<float>& values_x,
                const std::vector<int64_t>& dims_y,
                const std::vector<float>& values_y) {
  std::unique_ptr<ONNXValue, decltype(&ReleaseONNXValue)> value_x(nullptr, ReleaseONNXValue);
  std::vector<ONNXValuePtr> inputs(1);
  inputs[0] = ONNXRuntimeCreateTensorAsONNXValue(env, dims_x, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  value_x.reset(inputs[0]);
  void* raw_data;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeGetTensorMutableData(inputs[0], &raw_data));
  memcpy(raw_data, values_x.data(), values_x.size() * sizeof(values_x[0]));
  std::vector<const char*> input_names{"X"};
  ONNXValuePtr rtensor = nullptr;
  std::unique_ptr<ONNXValueList, decltype(&ReleaseONNXValueList)> output(nullptr, ReleaseONNXValueList);
  if (with_target_names) {
    const char* output_names[] = {"Y"};
    std::vector<ONNXValuePtr> t(1);
    ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeRunInference(session_object, input_names.data(), inputs.data(), inputs.size(), output_names, 1, t.data()));
    rtensor = t[0];
  } else {
    size_t output_len;
    {
      ONNXValueListPtr t;
      ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeRunInferenceAndFetchAll(session_object, input_names.data(), inputs.data(), inputs.size(), &t, &output_len));
      output.reset(t);
    }

    ASSERT_EQ(static_cast<size_t>(1), output_len);
    rtensor = ONNXRuntimeONNXValueListGetNthValue(output.get(), 0);
  }
  ASSERT_NE(rtensor, nullptr);
  std::unique_ptr<ONNXRuntimeTensorTypeAndShapeInfo> shape_info;
  {
    ONNXRuntimeTensorTypeAndShapeInfo* shape_info_ptr;
    ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeGetTensorShapeAndType(rtensor, &shape_info_ptr));
    shape_info.reset(shape_info_ptr);
  }
  size_t rtensor_dims = ONNXRuntimeGetNumOfDimensions(shape_info.get());
  std::vector<int64_t> shape_array(rtensor_dims);
  ONNXRuntimeGetDimensions(shape_info.get(), shape_array.data(), shape_array.size());
  ASSERT_EQ(shape_array, dims_y);
  size_t total_len = 1;
  for (size_t i = 0; i != rtensor_dims; ++i) {
    total_len *= shape_array[i];
  }
  ASSERT_EQ(values_y.size(), total_len);
  float* f;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeGetTensorMutableData(rtensor, (void**)&f));
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }
  if (with_target_names) {
    ReleaseONNXValue(rtensor);
  }
}

template <typename T, bool with_target_names>
void TestInference(ONNXEnv* env, const T& model_uri,
                   const std::vector<size_t>& dims_x,
                   const std::vector<float>& values_x,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<float>& expected_values_y,
                   int provider_type, bool custom_op) {
  SessionOptionsWrapper sf(env);

  if (provider_type == 1) {
#ifdef USE_CUDA
    ONNXRuntimeProviderFactoryPtr* f;
    ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateCUDAExecutionProviderFactory(0, &f));
    sf.AppendExecutionProvider(f);
    ONNXRuntimeReleaseObject(f);
    std::cout << "Running simple inference with cuda provider" << std::endl;
#else
    return;
#endif
  } else if (provider_type == 2) {
#ifdef USE_MKLDNN
    ONNXRuntimeProviderFactoryPtr* f;
    ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateMkldnnExecutionProviderFactory(1, &f));
    sf.AppendExecutionProvider(f);
    ONNXRuntimeReleaseObject(f);
    std::cout << "Running simple inference with mkldnn provider" << std::endl;
#else
    return;
#endif
  } else if (provider_type == 3) {
#ifdef USE_NUPHAR
    ONNXRuntimeProviderFactoryPtr* f;
    ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateNupharExecutionProviderFactory(0, "", &f));
    sf.AppendExecutionProvider(f);
    ONNXRuntimeReleaseObject(f);
    std::cout << "Running simple inference with nuphar provider" << std::endl;
#else
    return;
#endif
  } else {
    std::cout << "Running simple inference with default provider" << std::endl;
  }
  if (custom_op) {
    sf.AddCustomOp("libonnxruntime_custom_op_shared_lib_test.so");
  }
  std::unique_ptr<ONNXSession, decltype(&ReleaseONNXSession)> inference_session(sf.ONNXRuntimeCreateInferenceSession(model_uri.c_str()), ReleaseONNXSession);
  std::unique_ptr<ONNXRuntimeAllocator> default_allocator(MockedONNXRuntimeAllocator::Create());
  // Now run
  RunSession<with_target_names>(default_allocator.get(), inference_session.get(), dims_x, values_x, expected_dims_y, expected_values_y);
}

#ifdef _WIN32
typedef std::wstring PATH_TYPE;
#define TSTR(X) L##X
#else
#define TSTR(X) (X)
typedef std::string PATH_TYPE;
#endif

static const PATH_TYPE MODEL_URI = TSTR("testdata/mul_1.pb");
static const PATH_TYPE CUSTOM_OP_MODEL_URI = TSTR("testdata/foo_1.pb");

class CApiTestWithProvider : public CApiTest,
                             public ::testing::WithParamInterface<int> {
};

// Tests that the Foo::Bar() method does Abc.
TEST_P(CApiTestWithProvider, simple) {
  const PATH_TYPE input_filepath = TSTR("this/package/testdata/myinputfile.dat");
  const PATH_TYPE output_filepath = TSTR("this/package/testdata/myoutputfile.dat");
  // simple inference test
  // prepare inputs
  std::vector<size_t> dims_x = {3, 2};
  std::vector<float> values_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  TestInference<PATH_TYPE, true>(env, MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, GetParam(), false);
  TestInference<PATH_TYPE, false>(env, MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, GetParam(), false);
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

  TestInference<PATH_TYPE, true>(env, CUSTOM_OP_MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, false, true);
  TestInference<PATH_TYPE, false>(env, CUSTOM_OP_MODEL_URI, dims_x, values_x, expected_dims_y, expected_values_y, false, true);
}
#endif

TEST_F(CApiTest, create_tensor) {
  const char* s[] = {"abc", "kmp"};
  size_t expected_len = 2;
  std::unique_ptr<ONNXRuntimeAllocator> default_allocator(MockedONNXRuntimeAllocator::Create());
  {
    std::unique_ptr<ONNXValue, decltype(&ReleaseONNXValue)> tensor(
        ONNXRuntimeCreateTensorAsONNXValue(default_allocator.get(), {expected_len}, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING), ReleaseONNXValue);
    ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeFillStringTensor(tensor.get(), s, expected_len));
    std::unique_ptr<ONNXRuntimeTensorTypeAndShapeInfo> shape_info;
    {
      ONNXRuntimeTensorTypeAndShapeInfo* shape_info_ptr;
      ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeGetTensorShapeAndType(tensor.get(), &shape_info_ptr));
      shape_info.reset(shape_info_ptr);
    }
    size_t len = static_cast<size_t>(ONNXRuntimeGetTensorShapeElementCount(shape_info.get()));
    ASSERT_EQ(len, expected_len);
    std::vector<int64_t> shape_array(len);

    size_t data_len;
    ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeGetStringTensorDataLength(tensor.get(), &data_len));
    std::string result(data_len, '\0');
    std::vector<size_t> offsets(len);
    ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeGetStringTensorContent(tensor.get(), (void*)result.data(), data_len, offsets.data(), offsets.size()));
  }
}

TEST_F(CApiTest, create_tensor_with_data) {
  float values[] = {3.0f, 1.0f, 2.f, 0.f};
  constexpr size_t values_length = sizeof(values) / sizeof(values[0]);
  ONNXRuntimeAllocatorInfo* info;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateAllocatorInfo("Cpu", ONNXRuntimeDeviceAllocator, 0, ONNXRuntimeMemTypeDefault, &info));
  std::vector<size_t> dims = {3};
  std::unique_ptr<ONNXValue, decltype(&ReleaseONNXValue)> tensor(
      ONNXRuntimeCreateTensorWithDataAsONNXValue(info, values, values_length * sizeof(float), dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT), ReleaseONNXValue);
  ReleaseONNXRuntimeAllocatorInfo(info);
  void* new_pointer;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeGetTensorMutableData(tensor.get(), &new_pointer));
  ASSERT_EQ(new_pointer, values);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
