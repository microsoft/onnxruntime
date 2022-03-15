// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <thread>

#include <gtest/gtest.h>

#include "core/common/common.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/util/thread_utils.h"
#include "providers.h"
#include "test_allocator.h"
#include "test_fixture.h"
#include "utils.h"
#include "custom_op_utils.h"
#include <gsl/gsl>

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// Once we use C++17 this could be replaced with std::size
template <typename T, size_t N>
constexpr size_t countof(T (&)[N]) { return N; }

extern std::unique_ptr<Ort::Env> ort_env;

template <typename OutT>
void RunSession(OrtAllocator* allocator, Ort::Session& session_object,
                const std::vector<Input>& inputs,
                const char* output_name,
                const std::vector<int64_t>& dims_y,
                const std::vector<OutT>& values_y,
                Ort::Value* output_tensor) {
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(
        Ort::Value::CreateTensor<float>(allocator->Info(allocator), const_cast<float*>(inputs[i].values.data()),
                                        inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }

  std::vector<Ort::Value> ort_outputs;
  if (output_tensor)
    session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                       &output_name, output_tensor, 1);
  else {
    ort_outputs = session_object.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                     &output_name, 1);
    ASSERT_EQ(ort_outputs.size(), 1u);
    output_tensor = &ort_outputs[0];
  }

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), dims_y);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), total_len);

  OutT* f = output_tensor->GetTensorMutableData<OutT>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }
}

template <typename OutT>
static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
                          const std::vector<Input>& inputs,
                          const char* output_name,
                          const std::vector<int64_t>& expected_dims_y,
                          const std::vector<OutT>& expected_values_y,
                          int provider_type,
                          OrtCustomOpDomain* custom_op_domain_ptr,
                          const char* custom_op_library_filename,
                          void** library_handle = nullptr,
                          bool test_session_creation_only = false,
                          void* cuda_compute_stream = nullptr) {
  Ort::SessionOptions session_options;

  if (provider_type == 1) {
#ifdef USE_CUDA
    std::cout << "Running simple inference with cuda provider" << std::endl;
    auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(cuda_compute_stream);
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
    ORT_UNUSED_PARAMETER(cuda_compute_stream);
    return;
#endif
  } else if (provider_type == 2) {
#ifdef USE_DNNL
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options, 1));
    std::cout << "Running simple inference with dnnl provider" << std::endl;
#else
    return;
#endif
  } else if (provider_type == 3) {
#ifdef USE_NUPHAR
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nuphar(session_options,
                                                                      /*allow_unaligned_buffers*/ 1, ""));
    std::cout << "Running simple inference with nuphar provider" << std::endl;
#else
    return;
#endif
  } else {
    std::cout << "Running simple inference with default provider" << std::endl;
  }
  if (custom_op_domain_ptr) {
    session_options.Add(custom_op_domain_ptr);
  }

  if (custom_op_library_filename) {
    Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(session_options,
                                                             custom_op_library_filename, library_handle));
  }

  // if session creation passes, model loads fine
  Ort::Session session(env, model_uri.c_str(), session_options);

  // caller wants to test running the model (not just loading the model)
  if (!test_session_creation_only) {
    // Now run
    auto default_allocator = std::make_unique<MockedOrtAllocator>();

    //without preallocated output tensor
    RunSession<OutT>(default_allocator.get(),
                     session,
                     inputs,
                     output_name,
                     expected_dims_y,
                     expected_values_y,
                     nullptr);
    //with preallocated output tensor
    Ort::Value value_y = Ort::Value::CreateTensor<float>(default_allocator.get(),
                                                         expected_dims_y.data(), expected_dims_y.size());

    //test it twice
    for (int i = 0; i != 2; ++i)
      RunSession<OutT>(default_allocator.get(),
                       session,
                       inputs,
                       output_name,
                       expected_dims_y,
                       expected_values_y,
                       &value_y);
  }
}

static constexpr PATH_TYPE MODEL_URI = TSTR("testdata/mul_1.onnx");
static constexpr PATH_TYPE MATMUL_MODEL_URI = TSTR("testdata/matmul_1.onnx");
#ifndef ORT_NO_RTTI
static constexpr PATH_TYPE SEQUENCE_MODEL_URI = TSTR("testdata/sequence_length.onnx");
#endif
#if !defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
static constexpr PATH_TYPE SEQUENCE_MODEL_URI_2 = TSTR("testdata/optional_sequence_tensor.onnx");
#endif
static constexpr PATH_TYPE CUSTOM_OP_MODEL_URI = TSTR("testdata/foo_1.onnx");
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_TEST_MODEL_URI = TSTR("testdata/custom_op_library/custom_op_test.onnx");
static constexpr PATH_TYPE OVERRIDABLE_INITIALIZER_MODEL_URI = TSTR("testdata/overridable_initializer.onnx");
static constexpr PATH_TYPE NAMED_AND_ANON_DIM_PARAM_URI = TSTR("testdata/capi_symbolic_dims.onnx");
static constexpr PATH_TYPE MODEL_WITH_CUSTOM_MODEL_METADATA = TSTR("testdata/model_with_valid_ort_config_json.onnx");
static constexpr PATH_TYPE VARIED_INPUT_CUSTOM_OP_MODEL_URI = TSTR("testdata/VariedInputCustomOp.onnx");
static constexpr PATH_TYPE VARIED_INPUT_CUSTOM_OP_MODEL_URI_2 = TSTR("testdata/foo_3.onnx");
static constexpr PATH_TYPE OPTIONAL_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI = TSTR("testdata/foo_bar_1.onnx");
static constexpr PATH_TYPE OPTIONAL_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI_2 = TSTR("testdata/foo_bar_2.onnx");
static constexpr PATH_TYPE CUSTOM_OP_MODEL_WITH_ATTRIBUTES_URI = TSTR("testdata/foo_bar_3.onnx");
#if !defined(DISABLE_SPARSE_TENSORS)
static constexpr PATH_TYPE SPARSE_OUTPUT_MODEL_URI = TSTR("testdata/sparse_initializer_as_output.onnx");
#ifndef DISABLE_CONTRIB_OPS
static constexpr PATH_TYPE SPARSE_INPUT_MATMUL_MODEL_URI = TSTR("testdata/sparse_to_dense_matmul.onnx");
#endif
#endif  // !defined(DISABLE_SPARSE_TENSORS)

#ifdef ENABLE_EXTENSION_CUSTOM_OPS
static constexpr PATH_TYPE ORT_CUSTOM_OPS_MODEL_URI = TSTR("testdata/custom_op_string_lower.onnx");
static constexpr PATH_TYPE ORT_CUSTOM_OPS_MODEL_URI_2 = TSTR("testdata/custom_op_negpos.onnx");
#endif

#ifdef ENABLE_LANGUAGE_INTEROP_OPS
static constexpr PATH_TYPE PYOP_FLOAT_MODEL_URI = TSTR("testdata/pyop_1.onnx");
static constexpr PATH_TYPE PYOP_MULTI_MODEL_URI = TSTR("testdata/pyop_2.onnx");
static constexpr PATH_TYPE PYOP_KWARG_MODEL_URI = TSTR("testdata/pyop_3.onnx");
#endif

#ifndef REDUCED_OPS_BUILD
static constexpr PATH_TYPE RESIZE_AND_CROP_MODEL_URI = TSTR("testdata/crop_and_resize.onnx");
#endif

class CApiTestWithProvider : public testing::Test, public ::testing::WithParamInterface<int> {
};

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

  TestInference<float>(*ort_env, MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, GetParam(),
                       nullptr, nullptr);
}

TEST(CApiTest, dim_param) {
  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, NAMED_AND_ANON_DIM_PARAM_URI, session_options);

  auto in0 = session.GetInputTypeInfo(0);
  auto in0_ttsi = in0.GetTensorTypeAndShapeInfo();

  auto num_input_dims = in0_ttsi.GetDimensionsCount();
  ASSERT_GE(num_input_dims, 1u);
  // reading 1st dimension only so don't need to malloc int64_t* or const char** values for the Get*Dimensions calls
  int64_t dim_value = 0;
  const char* dim_param = nullptr;
  in0_ttsi.GetDimensions(&dim_value, 1);
  in0_ttsi.GetSymbolicDimensions(&dim_param, 1);
  ASSERT_EQ(dim_value, -1) << "symbolic dimension should be -1";
  ASSERT_EQ(strcmp(dim_param, "n"), 0) << "Expected 'n'. Got: " << dim_param;

  auto out0 = session.GetOutputTypeInfo(0);
  auto out0_ttsi = out0.GetTensorTypeAndShapeInfo();
  auto num_output_dims = out0_ttsi.GetDimensionsCount();
  ASSERT_EQ(num_output_dims, 1u);

  out0_ttsi.GetDimensions(&dim_value, 1);
  out0_ttsi.GetSymbolicDimensions(&dim_param, 1);
  ASSERT_EQ(dim_value, -1) << "symbolic dimension should be -1";
  ASSERT_EQ(strcmp(dim_param, ""), 0);
}

INSTANTIATE_TEST_SUITE_P(CApiTestWithProviders,
                         CApiTestWithProvider,
                         ::testing::Values(0, 1, 2, 3, 4));

#if !defined(DISABLE_SPARSE_TENSORS)
TEST(CApiTest, SparseOutputModel) {
  std::vector<int64_t> dense_shape{3, 3};
  std::vector<float> values{1.764052391052246, 0.40015721321105957, 0.978738009929657};
  std::vector<int64_t> values_shape{3};
  std::vector<int64_t> coo_indices{2, 3, 5};
  std::vector<int64_t> indices_shape{3};

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  const char* const output_names[] = {"values"};
  Ort::Session session(*ort_env, SPARSE_OUTPUT_MODEL_URI, Ort::SessionOptions{});
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);
  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& sparse_output = ort_outputs[0];
  auto ti = sparse_output.GetTypeInfo();
  ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
  auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, tensor_type_shape.GetElementType());

  ASSERT_EQ(ORT_SPARSE_COO, sparse_output.GetSparseFormat());
  auto values_ts = sparse_output.GetSparseTensorValuesTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, values_ts.GetElementType());
  ASSERT_EQ(values_shape, values_ts.GetShape());

  const auto* values_fetch = sparse_output.GetSparseTensorValues<float>();
  auto val_span = gsl::make_span(values_fetch, values.size());
  ASSERT_TRUE(std::equal(values.cbegin(), values.cend(), val_span.cbegin(), val_span.cend()));

  auto indices_ts = sparse_output.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_COO_INDICES);
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
  ASSERT_EQ(indices_shape, indices_ts.GetShape());

  size_t num_indices = 0;
  const int64_t* indices = sparse_output.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_COO_INDICES, num_indices);
  ASSERT_EQ(num_indices, static_cast<size_t>(indices_shape[0]));
  auto ind_span = gsl::make_span(indices, num_indices);
  ASSERT_TRUE(std::equal(coo_indices.cbegin(), coo_indices.cend(), ind_span.cbegin(), ind_span.cend()));
}

#ifndef DISABLE_CONTRIB_OPS
TEST(CApiTest, SparseInputModel) {
  std::vector<int64_t> common_shape{9, 9};  // inputs and outputs same shape
  std::vector<float> A_values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                              10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                              18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                              26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
                              34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0,
                              42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0,
                              50.0, 51.0, 52.0, 53.0};

  // 2 - D index
  std::vector<int64_t> indices_shape{gsl::narrow<int64_t>(A_values.size()), 2};
  std::vector<int64_t> A_indices{0, 1, 0, 2, 0, 6, 0, 7, 0, 8, 1, 0, 1,
                                 1, 1, 2, 1, 6, 1, 7, 1, 8, 2, 0, 2, 1,
                                 2, 2, 2, 6, 2, 7, 2, 8, 3, 3, 3, 4, 3,
                                 5, 3, 6, 3, 7, 3, 8, 4, 3, 4, 4, 4, 5,
                                 4, 6, 4, 7, 4, 8, 5, 3, 5, 4, 5, 5, 5,
                                 6, 5, 7, 5, 8, 6, 0, 6, 1, 6, 2, 6, 3,
                                 6, 4, 6, 5, 7, 0, 7, 1, 7, 2, 7, 3, 7,
                                 4, 7, 5, 8, 0, 8, 1, 8, 2, 8, 3, 8, 4,
                                 8, 5};

  std::vector<float> B_data{0, 1, 2, 0, 0, 0, 3, 4, 5,
                            6, 7, 8, 0, 0, 0, 9, 10, 11,
                            12, 13, 14, 0, 0, 0, 15, 16, 17,
                            0, 0, 0, 18, 19, 20, 21, 22, 23,
                            0, 0, 0, 24, 25, 26, 27, 28, 29,
                            0, 0, 0, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 0, 0, 0,
                            42, 43, 44, 45, 46, 47, 0, 0, 0,
                            48, 49, 50, 51, 52, 53, 0, 0, 0};

  std::vector<float> Y_result{546, 561, 576, 552, 564, 576, 39, 42, 45,
                              1410, 1461, 1512, 1362, 1392, 1422, 201, 222, 243,
                              2274, 2361, 2448, 2172, 2220, 2268, 363, 402, 441,
                              2784, 2850, 2916, 4362, 4485, 4608, 1551, 1608, 1665,
                              3540, 3624, 3708, 5604, 5763, 5922, 2037, 2112, 2187,
                              4296, 4398, 4500, 6846, 7041, 7236, 2523, 2616, 2709,
                              678, 789, 900, 2892, 3012, 3132, 4263, 4494, 4725,
                              786, 915, 1044, 3324, 3462, 3600, 4911, 5178, 5445,
                              894, 1041, 1188, 3756, 3912, 4068, 5559, 5862, 6165};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::Value::Shape ort_dense_shape{common_shape.data(), common_shape.size()};
  Ort::Value::Shape ort_values_shape{&indices_shape[0], 1U};
  auto a_st = Ort::Value::CreateSparseTensor(info, A_values.data(), ort_dense_shape, ort_values_shape);
  a_st.UseCooIndices(A_indices.data(), A_indices.size());

  auto b_tensor = Ort::Value::CreateTensor(info, B_data.data(), B_data.size(), common_shape.data(), common_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(a_st));
  ort_inputs.push_back(std::move(b_tensor));
  const char* input_names[] = {"sparse_A", "dense_B"};
  const char* const output_names[] = {"dense_Y"};
  Ort::Session session(*ort_env, SPARSE_INPUT_MATMUL_MODEL_URI, Ort::SessionOptions{});
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);
  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& dense_Y = ort_outputs[0];
  ASSERT_TRUE(dense_Y.IsTensor());

  auto result_ts = dense_Y.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, result_ts.GetElementType());
  ASSERT_EQ(common_shape, result_ts.GetShape());

  const auto* result_vals = dense_Y.GetTensorData<float>();
  auto result_span = gsl::make_span(result_vals, Y_result.size());
  ASSERT_TRUE(std::equal(Y_result.cbegin(), Y_result.cend(), result_span.cbegin(), result_span.cend()));
}
#endif  // DISABLE_CONTRIB_OPS
#endif  // !defined(DISABLE_SPARSE_TENSORS)

TEST(CApiTest, custom_op_handler) {
  std::cout << "Running custom op inference" << std::endl;

  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

#ifdef USE_CUDA
  cudaStream_t compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  MyCustomOp custom_op{onnxruntime::kCudaExecutionProvider, compute_stream};
#else
  MyCustomOp custom_op{onnxruntime::kCpuExecutionProvider, nullptr};
#endif

  Ort::CustomOpDomain custom_op_domain("");
  custom_op_domain.Add(&custom_op);

#ifdef USE_CUDA
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 1,
                       custom_op_domain, nullptr, nullptr, false, compute_stream);
  cudaStreamDestroy(compute_stream);
#else
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       custom_op_domain, nullptr);
#endif
}

#ifdef ENABLE_EXTENSION_CUSTOM_OPS
// test enabled ort-customops negpos
TEST(CApiTest, test_enable_ort_customops_negpos) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto allocator = std::make_unique<MockedOrtAllocator>();

  // Create Inputs
  std::vector<Ort::Value> ort_inputs;
  std::vector<float> input_data = {-1.1f, 2.2f, 4.4f, -5.5f};
  std::vector<int64_t> input_dims = {2, 2};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_data.data()), input_data.size(), input_dims.data(), input_dims.size()));

  // Create Session with ORT CustomOps
  Ort::SessionOptions session_options;
  session_options.EnableOrtCustomOps();
  Ort::Session session(*ort_env, ORT_CUSTOM_OPS_MODEL_URI_2, session_options);

  // Create Input and Output Names
  std::vector<const char*> input_names = {"X"};
  const char* output_names[] = {"out0", "out1"};

  // Run Session
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, countof(output_names));

  // Validate Results
  ASSERT_EQ(ort_outputs.size(), 2u);

  std::vector<int64_t> out_dims = {2, 2};
  std::vector<float> values_out0 = {-1.1f, 0.0f, 0.0f, -5.5f};
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), out_dims);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_out0.size(), total_len);

  float* f = ort_outputs[0].GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_out0[i], f[i]);
  }
}

// test enabled ort-customops stringlower
TEST(CApiTest, test_enable_ort_customops_stringlower) {
  auto allocator = std::make_unique<MockedOrtAllocator>();

  // Create Inputs
  std::vector<Ort::Value> ort_inputs;
  std::string input_data{"HI, This is ENGINEER from Microsoft."};
  const char* const input_strings[] = {input_data.c_str()};
  std::vector<int64_t> input_dims = {1, 1};

  Ort::Value input_tensor = Ort::Value::CreateTensor(allocator.get(), input_dims.data(), input_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
  input_tensor.FillStringTensor(input_strings, 1U);
  ort_inputs.push_back(std::move(input_tensor));

  // Create Session with ORT CustomOps
  Ort::SessionOptions session_options;
  session_options.EnableOrtCustomOps();
  Ort::Session session(*ort_env, ORT_CUSTOM_OPS_MODEL_URI, session_options);

  // Create Input and Output Names
  std::vector<const char*> input_names = {"input_1"};
  const char* output_names[] = {"customout"};

  // Run Session
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, countof(output_names));

  // Validate Results
  ASSERT_EQ(ort_outputs.size(), 1u);

  std::vector<int64_t> out_dims = {1, 1};
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), out_dims);
  ASSERT_EQ(type_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  std::string output_data{"hi, this is engineer from microsoft."};
  auto expected_string = output_data.c_str();
  size_t expected_string_len = strlen(expected_string);
  auto data_length = ort_outputs[0].GetStringTensorDataLength();
  ASSERT_EQ(expected_string_len, data_length);

  std::string result(data_length, '\0');
  std::vector<size_t> offsets(type_info.GetElementCount());
  ort_outputs[0].GetStringTensorContent((void*)result.data(), data_length, offsets.data(), offsets.size());
  ASSERT_STREQ(result.c_str(), expected_string);
}
#endif

//test custom op which accepts float and double as inputs
TEST(CApiTest, varied_input_custom_op_handler) {
  std::vector<Input> inputs(2);
  inputs[0].name = "X";
  inputs[0].dims = {3};
  inputs[0].values = {2.0f, 3.0f, 4.0f};
  inputs[1].name = "Y";
  inputs[1].dims = {3};
  inputs[1].values = {5.0f, 6.0f, 7.0f};
  std::vector<int64_t> expected_dims_z = {1};
  std::vector<float> expected_values_z = {10.0f};

#ifdef USE_CUDA
  SliceCustomOp slice_custom_op{onnxruntime::kCudaExecutionProvider};
#else
  SliceCustomOp slice_custom_op{onnxruntime::kCpuExecutionProvider};
#endif

  Ort::CustomOpDomain custom_op_domain("abc");
  custom_op_domain.Add(&slice_custom_op);

#ifdef USE_CUDA
  TestInference<float>(*ort_env, VARIED_INPUT_CUSTOM_OP_MODEL_URI, inputs, "Z",
                       expected_dims_z, expected_values_z, 1, custom_op_domain, nullptr);
#else
  TestInference<float>(*ort_env, VARIED_INPUT_CUSTOM_OP_MODEL_URI, inputs, "Z",
                       expected_dims_z, expected_values_z, 0, custom_op_domain, nullptr);
#endif
}

TEST(CApiTest, multiple_varied_input_custom_op_handler) {
#ifdef USE_CUDA
  cudaStream_t compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  MyCustomOpMultipleDynamicInputs custom_op{onnxruntime::kCudaExecutionProvider, compute_stream};
#else
  MyCustomOpMultipleDynamicInputs custom_op{onnxruntime::kCpuExecutionProvider, nullptr};
#endif

  Ort::CustomOpDomain custom_op_domain("");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;

#ifdef USE_CUDA
  auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(compute_stream);
  session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif

  session_options.Add(custom_op_domain);
  Ort::Session session(*ort_env, VARIED_INPUT_CUSTOM_OP_MODEL_URI_2, session_options);

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;

  // input 0 (float type)
  input_names.emplace_back("X");
  std::vector<float> input_0_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<int64_t> input_0_dims = {3, 2};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_0_data.data()),
                                      input_0_data.size(), input_0_dims.data(), input_0_dims.size()));

  // input 1 (double type)
  input_names.emplace_back("W");
  std::vector<double> input_1_data = {2, 3, 4, 5, 6, 7};
  std::vector<int64_t> input_1_dims = {3, 2};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<double>(info, const_cast<double*>(input_1_data.data()),
                                       input_1_data.size(), input_1_dims.data(), input_1_dims.size()));

  // Run
  const char* output_name = "Y";
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 &output_name, 1);
  ASSERT_EQ(ort_outputs.size(), 1u);

  // Validate results
  std::vector<int64_t> y_dims = {3, 2};
  std::vector<float> values_y = {3.f, 5.f, 7.f, 9.f, 11.f, 13.f};
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), y_dims);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), total_len);

  float* f = ort_outputs[0].GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }

#ifdef USE_CUDA
  cudaStreamDestroy(compute_stream);
#endif
}

TEST(CApiTest, optional_input_output_custom_op_handler) {
  MyCustomOpWithOptionalInput custom_op{onnxruntime::kCpuExecutionProvider};

  // `MyCustomOpFooBar` defines a custom op with atmost 3 inputs and the second input is optional.
  // In this test, we are going to try and run 2 models - one with the optional input and one without
  // the optional input.
  Ort::CustomOpDomain custom_op_domain("");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  std::vector<Ort::Value> ort_inputs;

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  // input 0
  std::vector<float> input_0_data = {1.f};
  std::vector<int64_t> input_0_dims = {1};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_0_data.data()),
                                      input_0_data.size(), input_0_dims.data(), input_0_dims.size()));

  // input 1
  std::vector<float> input_1_data = {1.f};
  std::vector<int64_t> input_1_dims = {1};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_1_data.data()),
                                      input_1_data.size(), input_1_dims.data(), input_1_dims.size()));

  // input 2
  std::vector<float> input_2_data = {1.f};
  std::vector<int64_t> input_2_dims = {1};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_2_data.data()),
                                      input_2_data.size(), input_2_dims.data(), input_2_dims.size()));

  const char* output_name = "Y";

  // Part 1: Model with optional input present
  {
    std::vector<const char*> input_names = {"X1", "X2", "X3"};
    Ort::Session session(*ort_env, OPTIONAL_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                   &output_name, 1);
    ASSERT_EQ(ort_outputs.size(), 1u);

    // Validate results
    std::vector<int64_t> y_dims = {1};
    std::vector<float> values_y = {3.f};
    auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), y_dims);
    size_t total_len = type_info.GetElementCount();
    ASSERT_EQ(values_y.size(), total_len);

    float* f = ort_outputs[0].GetTensorMutableData<float>();
    for (size_t i = 0; i != total_len; ++i) {
      ASSERT_EQ(values_y[i], f[i]);
    }
  }

  // Part 2: Model with optional input absent
  {
    std::vector<const char*> input_names = {"X1", "X2"};
    ort_inputs.erase(ort_inputs.begin() + 2);  // remove the last input in the container
    Ort::Session session(*ort_env, OPTIONAL_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI_2, session_options);
    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                   &output_name, 1);
    ASSERT_EQ(ort_outputs.size(), 1u);

    // Validate results
    std::vector<int64_t> y_dims = {1};
    std::vector<float> values_y = {2.f};
    auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), y_dims);
    size_t total_len = type_info.GetElementCount();
    ASSERT_EQ(values_y.size(), total_len);

    float* f = ort_outputs[0].GetTensorMutableData<float>();
    for (size_t i = 0; i != total_len; ++i) {
      ASSERT_EQ(values_y[i], f[i]);
    }
  }
}
TEST(CApiTest, custom_op_with_attributes_handler) {
  MyCustomOpWithAttributes custom_op{onnxruntime::kCpuExecutionProvider};

  Ort::CustomOpDomain custom_op_domain("");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  Ort::Session session(*ort_env, CUSTOM_OP_MODEL_WITH_ATTRIBUTES_URI, session_options);

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;

  // input 0 (float type)
  input_names.emplace_back("X");
  std::vector<float> input_0_data = {1.f};
  std::vector<int64_t> input_0_dims = {1};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_0_data.data()),
                                      input_0_data.size(), input_0_dims.data(), input_0_dims.size()));

  // Run
  const char* output_name = "Y";
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 &output_name, 1);
  ASSERT_EQ(ort_outputs.size(), 1u);

  // Validate results
  std::vector<int64_t> y_dims = {1};
  std::vector<float> values_y = {15.f};
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), y_dims);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), total_len);

  float* f = ort_outputs[0].GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }
}

// Tests registration of a custom op of the same name for both CPU and CUDA EPs
#ifdef USE_CUDA
TEST(CApiTest, RegisterCustomOpForCPUAndCUDA) {
  std::cout << "Tests registration of a custom op of the same name for both CPU and CUDA EPs" << std::endl;

  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  MyCustomOp custom_op_cpu{onnxruntime::kCpuExecutionProvider, nullptr};
  // We are going to test session creation only - hence it is not a problem to use the default stream as the compute stream for the custom op
  MyCustomOp custom_op_cuda{onnxruntime::kCudaExecutionProvider, nullptr};
  Ort::CustomOpDomain custom_op_domain("");
  custom_op_domain.Add(&custom_op_cpu);
  custom_op_domain.Add(&custom_op_cuda);

  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y,
                       expected_values_y, 1, custom_op_domain, nullptr, nullptr, true);
}
#endif

//It has memory leak. The OrtCustomOpDomain created in custom_op_library.cc:RegisterCustomOps function was not freed
#if defined(__ANDROID__)
TEST(CApiTest, DISABLED_test_custom_op_library) {
//To accomodate a reduced op build pipeline
#elif defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
TEST(CApiTest, DISABLED_test_custom_op_library) {
#else
TEST(CApiTest, test_custom_op_library) {
#endif
  std::cout << "Running inference using custom op shared library" << std::endl;

  std::vector<Input> inputs(2);
  inputs[0].name = "input_1";
  inputs[0].dims = {3, 5};
  inputs[0].values = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                      6.6f, 7.7f, 8.8f, 9.9f, 10.0f,
                      11.1f, 12.2f, 13.3f, 14.4f, 15.5f};
  inputs[1].name = "input_2";
  inputs[1].dims = {3, 5};
  inputs[1].values = {15.5f, 14.4f, 13.3f, 12.2f, 11.1f,
                      10.0f, 9.9f, 8.8f, 7.7f, 6.6f,
                      5.5f, 4.4f, 3.3f, 2.2f, 1.1f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 5};
  std::vector<int32_t> expected_values_y =
      {17, 17, 17, 17, 17,
       17, 18, 18, 18, 17,
       17, 17, 17, 17, 17};

  std::string lib_name;
#if defined(_WIN32)
  lib_name = "custom_op_library.dll";
#elif defined(__APPLE__)
  lib_name = "libcustom_op_library.dylib";
#else
  lib_name = "./libcustom_op_library.so";
#endif

  void* library_handle = nullptr;
#ifdef USE_CUDA
  TestInference<int32_t>(*ort_env, CUSTOM_OP_LIBRARY_TEST_MODEL_URI, inputs, "output", expected_dims_y,
                         expected_values_y, 1, nullptr, lib_name.c_str(), &library_handle);
#else
  TestInference<int32_t>(*ort_env, CUSTOM_OP_LIBRARY_TEST_MODEL_URI, inputs, "output", expected_dims_y,
                         expected_values_y, 0, nullptr, lib_name.c_str(), &library_handle);
#endif

#ifdef _WIN32
  bool success = ::FreeLibrary(reinterpret_cast<HMODULE>(library_handle));
  ORT_ENFORCE(success, "Error while closing custom op shared library");
#else
  int retval = dlclose(library_handle);
  ORT_ENFORCE(retval == 0, "Error while closing custom op shared library");
#endif
}

#if defined(ENABLE_LANGUAGE_INTEROP_OPS)
std::once_flag my_module_flag;

void PrepareModule() {
  std::ofstream module("mymodule.py");
  module << "class MyKernel:" << std::endl;
  module << "\t"
         << "def __init__(self,A,B,C):" << std::endl;
  module << "\t\t"
         << "self.a,self.b,self.c = A,B,C" << std::endl;
  module << "\t"
         << "def compute(self,x):" << std::endl;
  module << "\t\t"
         << "return x*2" << std::endl;
  module << "class MyKernel_2:" << std::endl;
  module << "\t"
         << "def __init__(self,A,B):" << std::endl;
  module << "\t\t"
         << "self.a,self.b = A,B" << std::endl;
  module << "\t"
         << "def compute(self,x):" << std::endl;
  module << "\t\t"
         << "return x*4" << std::endl;
  module << "class MyKernel_3:" << std::endl;
  module << "\t"
         << "def __init__(self,A,B):" << std::endl;
  module << "\t\t"
         << "self.a,self.b = A,B" << std::endl;
  module << "\t"
         << "def compute(self,*kwargs):" << std::endl;
  module << "\t\t"
         << "return kwargs[0]*5" << std::endl;
  module.close();
}

TEST(CApiTest, test_pyop) {
  std::call_once(my_module_flag, PrepareModule);
  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {2, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> expected_dims_y = {2, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f};
  TestInference<float>(*ort_env, PYOP_FLOAT_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       nullptr, nullptr);
}

TEST(CApiTest, test_pyop_multi) {
  std::call_once(my_module_flag, PrepareModule);
  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {2, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> expected_dims_y = {2, 2};
  std::vector<float> expected_values_y = {8.0f, 16.0f, 24.0f, 32.0f};
  TestInference<float>(*ort_env, PYOP_MULTI_MODEL_URI, inputs, "Z", expected_dims_y, expected_values_y, 0,
                       nullptr, nullptr);
}

TEST(CApiTest, test_pyop_kwarg) {
  std::call_once(my_module_flag, PrepareModule);
  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {2, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> expected_dims_y = {2, 2};
  std::vector<float> expected_values_y = {25.0f, 50.0f, 75.0f, 100.0f};
  TestInference<float>(*ort_env, PYOP_KWARG_MODEL_URI, inputs, "Z", expected_dims_y, expected_values_y, 0,
                       nullptr, nullptr);
}
#endif

#ifdef ORT_RUN_EXTERNAL_ONNX_TESTS
TEST(CApiTest, create_session_without_session_option) {
  constexpr PATH_TYPE model_uri = TSTR("../models/opset8/test_squeezenet/model.onnx");
  Ort::Session ret(*ort_env, model_uri, Ort::SessionOptions{nullptr});
  ASSERT_NE(nullptr, ret);
}
#endif

#ifdef REDUCED_OPS_BUILD
TEST(ReducedOpsBuildTest, test_excluded_ops) {
  // In reduced ops build, test a model containing ops not included in required_ops.config cannot be loaded.
  // See onnxruntime/test/testdata/reduced_build_test.readme.txt for more details of the setup
  constexpr PATH_TYPE model_uri = TSTR("testdata/reduced_build_test.onnx_model_with_excluded_ops");
  std::vector<Input> inputs = {{"X", {3}, {-1.0f, 2.0f, -3.0f}}};
  std::vector<int64_t> expected_dims_y = {3};
  std::vector<float> expected_values_y = {0.1f, 0.1f, 0.1f};
  bool failed = false;
  try {
    //only test model loading, exception expected
    TestInference<float>(*ort_env, model_uri, inputs, "Y", expected_dims_y, expected_values_y, 0,
                         nullptr, nullptr, nullptr, true);
  } catch (const Ort::Exception& e) {
    failed = e.GetOrtErrorCode() == ORT_NOT_IMPLEMENTED;
  }
  ASSERT_EQ(failed, true);
}
#endif

TEST(CApiTest, get_allocator_cpu) {
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 1));
  Ort::Session session(*ort_env, NAMED_AND_ANON_DIM_PARAM_URI, session_options);
  Ort::MemoryInfo info_cpu = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator cpu_allocator(session, info_cpu);

  // CPU OrtMemoryInfo does not return OrtArenaAllocator on x86 but rather a device allocator
  // which causes MemoryInfo that is used to request the allocator and the actual instance
  // of MemoryInfo returned from the allocator exactly match, although they are functionally equivalent.
  auto allocator_info = cpu_allocator.GetInfo();
  ASSERT_EQ(info_cpu.GetAllocatorName(), allocator_info.GetAllocatorName());
  ASSERT_EQ(info_cpu.GetDeviceId(), allocator_info.GetDeviceId());
  ASSERT_EQ(info_cpu.GetMemoryType(), allocator_info.GetDeviceId());
  void* p = cpu_allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  cpu_allocator.Free(p);

  auto mem_allocation = cpu_allocator.GetAllocation(1024);
  ASSERT_NE(nullptr, mem_allocation.get());
  ASSERT_EQ(1024U, mem_allocation.size());
}

#ifdef USE_CUDA
TEST(CApiTest, get_allocator_cuda) {
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
  Ort::Session session(*ort_env, NAMED_AND_ANON_DIM_PARAM_URI, session_options);

  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator cuda_allocator(session, info_cuda);

  auto allocator_info = cuda_allocator.GetInfo();
  ASSERT_TRUE(info_cuda == allocator_info);
  void* p = cuda_allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  cuda_allocator.Free(p);

  auto mem_allocation = cuda_allocator.GetAllocation(1024);
  ASSERT_NE(nullptr, mem_allocation.get());
  ASSERT_EQ(1024U, mem_allocation.size());
}
#endif

TEST(CApiTest, io_binding) {
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 1));
  Ort::Session session(*ort_env, MODEL_URI, session_options);

  Ort::MemoryInfo info_cpu = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Ort::Value bound_x = Ort::Value::CreateTensor(info_cpu, x_values.data(), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  const std::array<int64_t, 2> y_shape = {3, 2};
  std::array<float, 3 * 2> y_values;
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cpu, y_values.data(), y_values.size(),
                                                y_shape.data(), y_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  session.Run(Ort::RunOptions(), binding);
  // Check the values against the bound raw memory
  ASSERT_TRUE(std::equal(std::begin(y_values), std::end(y_values), std::begin(expected_y)));

  // Now compare values via GetOutputValues
  {
    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    ASSERT_EQ(output_values.size(), 1U);
    const Ort::Value& Y_value = output_values[0];
    ASSERT_TRUE(Y_value.IsTensor());
    Ort::TensorTypeAndShapeInfo type_info = Y_value.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_info.GetElementType());
    auto count = type_info.GetElementCount();
    ASSERT_EQ(expected_y.size(), count);
    const float* values = Y_value.GetTensorData<float>();
    ASSERT_TRUE(std::equal(values, values + count, std::begin(expected_y)));
  }

  {
    std::vector<std::string> output_names = binding.GetOutputNames();
    ASSERT_EQ(1U, output_names.size());
    ASSERT_EQ(output_names[0].compare("Y"), 0);
  }

  // Now replace binding of Y with an on device binding instead of pre-allocated memory.
  // This is when we can not allocate an OrtValue due to unknown dimensions
  {
    Ort::MemoryInfo info_cpu_dev("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
    binding.BindOutput("Y", info_cpu_dev);
    session.Run(Ort::RunOptions(), binding);
  }

  // Check the output value allocated based on the device binding.
  {
    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    ASSERT_EQ(output_values.size(), 1U);
    const Ort::Value& Y_value = output_values[0];
    ASSERT_TRUE(Y_value.IsTensor());
    Ort::TensorTypeAndShapeInfo type_info = Y_value.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_info.GetElementType());
    auto count = type_info.GetElementCount();
    ASSERT_EQ(expected_y.size(), count);
    const float* values = Y_value.GetTensorData<float>();
    ASSERT_TRUE(std::equal(values, values + count, std::begin(expected_y)));
  }

  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}

#if defined(USE_CUDA) || defined(USE_TENSORRT)
TEST(CApiTest, io_binding_cuda) {
  struct CudaMemoryDeleter {
    explicit CudaMemoryDeleter(const Ort::Allocator* alloc) {
      alloc_ = alloc;
    }
    void operator()(void* ptr) const {
      alloc_->Free(ptr);
    }

    const Ort::Allocator* alloc_;
  };

  Ort::SessionOptions session_options;
#ifdef USE_TENSORRT
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));
#else
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif
  Ort::Session session(*ort_env, MODEL_URI, session_options);

  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);

  Ort::Allocator cuda_allocator(session, info_cuda);
  auto allocator_info = cuda_allocator.GetInfo();
  ASSERT_TRUE(info_cuda == allocator_info);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(x_values.size() * sizeof(float)),
                                                             CudaMemoryDeleter(&cuda_allocator));
  ASSERT_NE(input_data.get(), nullptr);
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  const std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(expected_y.size() * sizeof(float)),
                                                              CudaMemoryDeleter(&cuda_allocator));
  ASSERT_NE(output_data.get(), nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data.get()),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  // Sychronize to make sure the copy on default stream is done since TensorRT isn't using default stream.
  binding.SynchronizeInputs();

  session.Run(Ort::RunOptions(), binding);

  binding.SynchronizeOutputs();

  // Check the values against the bound raw memory (needs copying from device to host first)
  std::array<float, 3 * 2> y_values_0;
  cudaMemcpy(y_values_0.data(), output_data.get(), sizeof(float) * y_values_0.size(), cudaMemcpyDeviceToHost);
  ASSERT_TRUE(std::equal(std::begin(y_values_0), std::end(y_values_0), std::begin(expected_y)));

  // Now compare values via GetOutputValues
  {
    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    ASSERT_EQ(output_values.size(), 1U);
    const Ort::Value& Y_value = output_values[0];
    ASSERT_TRUE(Y_value.IsTensor());
    Ort::TensorTypeAndShapeInfo type_info = Y_value.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_info.GetElementType());
    auto count = type_info.GetElementCount();
    ASSERT_EQ(expected_y.size(), count);
    const float* values = Y_value.GetTensorData<float>();

    std::array<float, 3 * 2> y_values_1;
    cudaMemcpy(y_values_1.data(), values, sizeof(float) * y_values_1.size(), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(std::equal(std::begin(y_values_1), std::end(y_values_1), std::begin(expected_y)));
  }

  {
    std::vector<std::string> output_names = binding.GetOutputNames();
    ASSERT_EQ(1U, output_names.size());
    ASSERT_EQ(output_names[0].compare("Y"), 0);
  }

  // Now replace binding of Y with an on device binding instead of pre-allocated memory.
  // This is when we can not allocate an OrtValue due to unknown dimensions
  {
    binding.BindOutput("Y", info_cuda);
    session.Run(Ort::RunOptions(), binding);
  }

  // Check the output value allocated based on the device binding.
  {
    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    ASSERT_EQ(output_values.size(), 1U);
    const Ort::Value& Y_value = output_values[0];
    ASSERT_TRUE(Y_value.IsTensor());
    Ort::TensorTypeAndShapeInfo type_info = Y_value.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_info.GetElementType());
    auto count = type_info.GetElementCount();
    ASSERT_EQ(expected_y.size(), count);
    const float* values = Y_value.GetTensorData<float>();

    std::array<float, 3 * 2> y_values_2;
    cudaMemcpy(y_values_2.data(), values, sizeof(float) * y_values_2.size(), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(std::equal(std::begin(y_values_2), std::end(y_values_2), std::begin(expected_y)));
  }

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}
#endif

#if defined(USE_CUDA)
TEST(CApiTest, cuda_graph) {
  const auto& api = Ort::GetApi();

  // Enable cuda graph in cuda provider option.
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  ASSERT_TRUE(api.CreateCUDAProviderOptions(&cuda_options) == nullptr);
  std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
  std::vector<const char*> keys{"enable_cuda_graph"};
  std::vector<const char*> values{"1"};
  ASSERT_TRUE(api.UpdateCUDAProviderOptions(
                  rel_cuda_options.get(), keys.data(), values.data(), 1) == nullptr);

  Ort::SessionOptions session_options;
  ASSERT_TRUE(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
                  static_cast<OrtSessionOptions*>(session_options),
                  rel_cuda_options.get()) == nullptr);

  // Create IoBinding for inputs and outputs.
  struct CudaMemoryDeleter {
    explicit CudaMemoryDeleter(const Ort::Allocator* alloc) {
      alloc_ = alloc;
    }
    void operator()(void* ptr) const {
      alloc_->Free(ptr);
    }

    const Ort::Allocator* alloc_;
  };

  Ort::Session session(*ort_env, MODEL_URI, session_options);
  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);

  Ort::Allocator cuda_allocator(session, info_cuda);
  auto allocator_info = cuda_allocator.GetInfo();
  ASSERT_TRUE(info_cuda == allocator_info);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(x_values.size() * sizeof(float)),
                                                             CudaMemoryDeleter(&cuda_allocator));
  ASSERT_NE(input_data.get(), nullptr);
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(expected_y.size() * sizeof(float)),
                                                              CudaMemoryDeleter(&cuda_allocator));
  ASSERT_NE(output_data.get(), nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data.get()),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  // One regular run for necessary memory allocation and graph capturing
  session.Run(Ort::RunOptions(), binding);

  // Check the values against the bound raw memory (needs copying from device to host first)
  std::array<float, 3 * 2> y_values;
  cudaMemcpy(y_values.data(), output_data.get(), sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);
  ASSERT_TRUE(std::equal(std::begin(y_values), std::end(y_values), std::begin(expected_y)));

  // Replay the captured CUDA graph
  session.Run(Ort::RunOptions(), binding);
  cudaMemcpy(y_values.data(), output_data.get(), sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);
  ASSERT_TRUE(std::equal(std::begin(y_values), std::end(y_values), std::begin(expected_y)));

  // Change the input and replay the CUDA graph again.
  x_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);
  session.Run(Ort::RunOptions(), binding);
  cudaMemcpy(y_values.data(), output_data.get(), sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);
  expected_y = {10.0f, 40.0f, 90.0f, 160.0f, 250.0f, 360.0f};
  ASSERT_TRUE(std::equal(std::begin(y_values), std::end(y_values), std::begin(expected_y)));

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}
#endif

TEST(CApiTest, create_tensor) {
  const char* s[] = {"abc", "kmp"};
  int64_t expected_len = 2;
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  Ort::Value tensor = Ort::Value::CreateTensor(default_allocator.get(), &expected_len, 1,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  Ort::ThrowOnError(Ort::GetApi().FillStringTensor(tensor, s, expected_len));
  auto shape_info = tensor.GetTensorTypeAndShapeInfo();

  int64_t len = shape_info.GetElementCount();
  ASSERT_EQ(len, expected_len);
  std::vector<int64_t> shape_array(len);

  size_t data_len = tensor.GetStringTensorDataLength();
  std::string result(data_len, '\0');
  std::vector<size_t> offsets(len);
  tensor.GetStringTensorContent((void*)result.data(), data_len, offsets.data(), offsets.size());
}

TEST(CApiTest, fill_string_tensor) {
  const char* s[] = {"abc", "kmp"};
  int64_t expected_len = 2;
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  Ort::Value tensor = Ort::Value::CreateTensor(default_allocator.get(), &expected_len, 1,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  for (int64_t i = 0; i < expected_len; i++) {
    tensor.FillStringTensorElement(s[i], i);
  }

  auto shape_info = tensor.GetTensorTypeAndShapeInfo();

  int64_t len = shape_info.GetElementCount();
  ASSERT_EQ(len, expected_len);
}

TEST(CApiTest, get_string_tensor_element) {
  const char* s[] = {"abc", "kmp"};
  int64_t expected_len = 2;
  int64_t element_index = 0;
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  Ort::Value tensor = Ort::Value::CreateTensor(default_allocator.get(), &expected_len, 1,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  tensor.FillStringTensor(s, expected_len);

  auto expected_string = s[element_index];
  size_t expected_string_len = strnlen(expected_string, onnxruntime::kMaxStrLen);

  std::string result(expected_string_len, '\0');
  tensor.GetStringTensorElement(expected_string_len, element_index, (void*)result.data());
  ASSERT_STREQ(result.c_str(), expected_string);

  auto string_len = tensor.GetStringTensorElementLength(element_index);
  ASSERT_EQ(expected_string_len, string_len);
}

TEST(CApiTest, create_tensor_with_data) {
  float values[] = {3.0f, 1.0f, 2.f, 0.f};
  constexpr size_t values_length = sizeof(values) / sizeof(values[0]);

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<int64_t> dims = {4};
  Ort::Value tensor = Ort::Value::CreateTensor<float>(info, values, values_length, dims.data(), dims.size());

  const float* new_pointer = tensor.GetTensorData<float>();
  ASSERT_EQ(new_pointer, values);

  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1u, tensor_info.GetDimensionsCount());
}

TEST(CApiTest, create_tensor_with_data_float16) {
  // Example with C++. However, what we are feeding underneath is really
  // a continuous buffer of uint16_t
  // Use 3rd party libraries such as Eigen to convert floats and doubles to float16 types.
  Ort::Float16_t values[] = {15360, 16384, 16896, 17408, 17664};  // 1.f, 2.f, 3.f, 4.f, 5.f
  constexpr size_t values_length = sizeof(values) / sizeof(values[0]);

  std::vector<int64_t> dims = {static_cast<int64_t>(values_length)};
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor<Ort::Float16_t>(info, values, values_length, dims.data(), dims.size());
  const auto* new_pointer = tensor.GetTensorData<Ort::Float16_t>();
  ASSERT_EQ(new_pointer, values);
  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1u, tensor_info.GetDimensionsCount());
  ASSERT_EQ(tensor_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  Ort::Float16_t value_at_1 = tensor.At<Ort::Float16_t>({1});
  ASSERT_EQ(values[1], value_at_1);
}

TEST(CApiTest, create_tensor_with_data_bfloat16) {
  // Example with C++. However, what we are feeding underneath is really
  // a continuous buffer of uint16_t
  // Conversion from float to bfloat16 is simple. Strip off half of the bytes from float.
  Ort::BFloat16_t values[] = {16256, 16384, 16448, 16512, 16544};  // 1.f, 2.f, 3.f, 4.f, 5.f
  constexpr size_t values_length = sizeof(values) / sizeof(values[0]);
  std::vector<int64_t> dims = {static_cast<int64_t>(values_length)};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor<Ort::BFloat16_t>(info, values, values_length, dims.data(), dims.size());
  const auto* new_pointer = tensor.GetTensorData<Ort::BFloat16_t>();
  ASSERT_EQ(new_pointer, values);
  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1u, tensor_info.GetDimensionsCount());
  ASSERT_EQ(tensor_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);

  Ort::BFloat16_t value_at_1 = tensor.At<Ort::BFloat16_t>({1});
  ASSERT_EQ(values[1], value_at_1);
}

TEST(CApiTest, access_tensor_data_elements) {
  /**
   * Create a 2x3 data blob that looks like:
   *
   *  0 1 2
   *  3 4 5
   */
  std::vector<int64_t> shape = {2, 3};
  int element_count = 6;  // 2*3
  std::vector<float> values(element_count);
  for (int i = 0; i < element_count; i++)
    values[i] = static_cast<float>(i);

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor<float>(info, values.data(), values.size(), shape.data(), shape.size());

  float expected_value = 0;
  for (int64_t row = 0; row < shape[0]; row++) {
    for (int64_t col = 0; col < shape[1]; col++) {
      ASSERT_EQ(expected_value++, tensor.At<float>({row, col}));
    }
  }
}

TEST(CApiTest, override_initializer) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto allocator = std::make_unique<MockedOrtAllocator>();
  // CreateTensor which is not owning this ptr
  bool Label_input[] = {true};
  std::vector<int64_t> dims = {1, 1};
  Ort::Value label_input_tensor = Ort::Value::CreateTensor<bool>(info, Label_input, 1U, dims.data(), dims.size());

  std::string f2_data{"f2_string"};
  // Place a string into Tensor OrtValue and assign to the
  Ort::Value f2_input_tensor = Ort::Value::CreateTensor(allocator.get(), dims.data(), dims.size(),
                                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
  const char* const input_char_string[] = {f2_data.c_str()};
  f2_input_tensor.FillStringTensor(input_char_string, 1U);

  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, OVERRIDABLE_INITIALIZER_MODEL_URI, session_options);

  // Get Overrideable initializers
  size_t init_count = session.GetOverridableInitializerCount();
  ASSERT_EQ(init_count, 1U);

  char* f1_init_name = session.GetOverridableInitializerName(0, allocator.get());
  ASSERT_TRUE(strcmp("F1", f1_init_name) == 0);
  allocator->Free(f1_init_name);

  Ort::TypeInfo init_type_info = session.GetOverridableInitializerTypeInfo(0);
  ASSERT_EQ(ONNX_TYPE_TENSOR, init_type_info.GetONNXType());

  // Let's override the initializer
  float f11_input_data[] = {2.0f};
  Ort::Value f11_input_tensor = Ort::Value::CreateTensor<float>(info, f11_input_data, 1U, dims.data(), dims.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(label_input_tensor));
  ort_inputs.push_back(std::move(f2_input_tensor));
  ort_inputs.push_back(std::move(f11_input_tensor));

  std::vector<const char*> input_names = {"Label", "F2", "F1"};
  const char* output_names[] = {"Label0", "F20", "F11"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                                    ort_inputs.data(), ort_inputs.size(),
                                                    output_names, countof(output_names));

  ASSERT_EQ(ort_outputs.size(), 3U);
  // Expecting the last output would be the overridden value of the initializer
  auto type_info = ort_outputs[2].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), dims);
  ASSERT_EQ(type_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_EQ(type_info.GetElementCount(), 1U);
  float* output_data = ort_outputs[2].GetTensorMutableData<float>();
  ASSERT_EQ(*output_data, f11_input_data[0]);
}

TEST(CApiTest, end_profiling) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto allocator = std::make_unique<MockedOrtAllocator>();

  // Create session with profiling enabled (profiling is automatically turned on)
  Ort::SessionOptions session_options_1;
#ifdef _WIN32
  session_options_1.EnableProfiling(L"profile_prefix");
#else
  session_options_1.EnableProfiling("profile_prefix");
#endif
  Ort::Session session_1(*ort_env, MODEL_WITH_CUSTOM_MODEL_METADATA, session_options_1);
  char* profile_file = session_1.EndProfiling(allocator.get());

  ASSERT_TRUE(std::string(profile_file).find("profile_prefix") != std::string::npos);
  allocator->Free(profile_file);
  // Create session with profiling disabled
  Ort::SessionOptions session_options_2;
#ifdef _WIN32
  session_options_2.DisableProfiling();
#else
  session_options_2.DisableProfiling();
#endif
  Ort::Session session_2(*ort_env, MODEL_WITH_CUSTOM_MODEL_METADATA, session_options_2);
  profile_file = session_2.EndProfiling(allocator.get());
  ASSERT_TRUE(std::string(profile_file) == std::string());
  allocator->Free(profile_file);
}

TEST(CApiTest, get_profiling_start_time) {
  // Test whether the C_API can access the profiler's start time
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::SessionOptions session_options;
#ifdef _WIN32
  session_options.EnableProfiling(L"profile_prefix");
#else
  session_options.EnableProfiling("profile_prefix");
#endif

  uint64_t before_start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::high_resolution_clock::now().time_since_epoch())
                                   .count();  // get current time
  Ort::Session session_1(*ort_env, MODEL_WITH_CUSTOM_MODEL_METADATA, session_options);
  uint64_t profiling_start_time = session_1.GetProfilingStartTimeNs();
  uint64_t after_start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::high_resolution_clock::now().time_since_epoch())
                                  .count();

  // the profiler's start time needs to be between before_time and after_time
  ASSERT_TRUE(before_start_time <= profiling_start_time && profiling_start_time <= after_start_time);
}

TEST(CApiTest, model_metadata) {
  auto allocator = std::make_unique<MockedOrtAllocator>();
  // The following all tap into the c++ APIs which internally wrap over C APIs

  // The following section tests a model containing all metadata supported via the APIs
  {
    Ort::SessionOptions session_options;
    Ort::Session session(*ort_env, MODEL_WITH_CUSTOM_MODEL_METADATA, session_options);

    // Fetch model metadata
    auto model_metadata = session.GetModelMetadata();

    char* producer_name = model_metadata.GetProducerName(allocator.get());
    ASSERT_TRUE(strcmp("Hari", producer_name) == 0);
    allocator.get()->Free(producer_name);

    char* graph_name = model_metadata.GetGraphName(allocator.get());
    ASSERT_TRUE(strcmp("matmul test", graph_name) == 0);
    allocator.get()->Free(graph_name);

    char* domain = model_metadata.GetDomain(allocator.get());
    ASSERT_TRUE(strcmp("", domain) == 0);
    allocator.get()->Free(domain);

    char* description = model_metadata.GetDescription(allocator.get());
    ASSERT_TRUE(strcmp("This is a test model with a valid ORT config Json", description) == 0);
    allocator.get()->Free(description);

    char* graph_description = model_metadata.GetGraphDescription(allocator.get());
    ASSERT_TRUE(strcmp("graph description", graph_description) == 0);
    allocator.get()->Free(graph_description);

    int64_t version = model_metadata.GetVersion();
    ASSERT_TRUE(version == 1);

    int64_t num_keys_in_custom_metadata_map;
    char** custom_metadata_map_keys = model_metadata.GetCustomMetadataMapKeys(allocator.get(),
                                                                              num_keys_in_custom_metadata_map);
    ASSERT_TRUE(num_keys_in_custom_metadata_map == 2);

    allocator.get()->Free(custom_metadata_map_keys[0]);
    allocator.get()->Free(custom_metadata_map_keys[1]);
    allocator.get()->Free(custom_metadata_map_keys);

    char* lookup_value_1 = model_metadata.LookupCustomMetadataMap("ort_config", allocator.get());
    ASSERT_TRUE(strcmp(lookup_value_1,
                       "{\"session_options\": {\"inter_op_num_threads\": 5, \"intra_op_num_threads\": 2, "
                       "\"graph_optimization_level\": 99, \"enable_profiling\": 1}}") == 0);
    allocator.get()->Free(lookup_value_1);

    char* lookup_value_2 = model_metadata.LookupCustomMetadataMap("dummy_key", allocator.get());
    ASSERT_TRUE(strcmp(lookup_value_2, "dummy_value") == 0);
    allocator.get()->Free(lookup_value_2);

    // key doesn't exist in custom metadata map
    char* lookup_value_3 = model_metadata.LookupCustomMetadataMap("key_doesnt_exist", allocator.get());
    ASSERT_TRUE(lookup_value_3 == nullptr);
  }

  // The following section tests a model with some missing metadata info
  // Adding this just to make sure the API implementation is able to handle empty/missing info
  {
    Ort::SessionOptions session_options;
    Ort::Session session(*ort_env, MODEL_URI, session_options);

    // Fetch model metadata
    auto model_metadata = session.GetModelMetadata();

    // Model description is empty
    char* description = model_metadata.GetDescription(allocator.get());
    ASSERT_TRUE(strcmp("", description) == 0);
    allocator.get()->Free(description);

    // Graph description is empty
    char* graph_description = model_metadata.GetGraphDescription(allocator.get());
    ASSERT_TRUE(strcmp("", graph_description) == 0);
    allocator.get()->Free(graph_description);

    // Model does not contain custom metadata map
    int64_t num_keys_in_custom_metadata_map;
    char** custom_metadata_map_keys = model_metadata.GetCustomMetadataMapKeys(allocator.get(),
                                                                              num_keys_in_custom_metadata_map);
    ASSERT_TRUE(num_keys_in_custom_metadata_map == 0);
    ASSERT_TRUE(custom_metadata_map_keys == nullptr);
  }
}

TEST(CApiTest, get_available_providers) {
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  int len = 0;
  char** providers;
  ASSERT_EQ(g_ort->GetAvailableProviders(&providers, &len), nullptr);
  ASSERT_GT(len, 0);
  ASSERT_STREQ(providers[len - 1], "CPUExecutionProvider");
  ASSERT_EQ(g_ort->ReleaseAvailableProviders(providers, len), nullptr);
}

TEST(CApiTest, get_available_providers_cpp) {
  std::vector<std::string> providers = Ort::GetAvailableProviders();
  ASSERT_FALSE(providers.empty());
  ASSERT_EQ(providers.back(), "CPUExecutionProvider");

#ifdef USE_CUDA
  // CUDA EP will exist in the list but its position may vary based on other EPs included in the build
  ASSERT_TRUE(std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end());
#endif
}
TEST(CApiTest, TestSharedAllocators) {
  OrtEnv* env_ptr = (OrtEnv*)(*ort_env);

  // prepare inputs
  std::vector<Input> inputs(1);
  Input& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto allocator_for_input_memory_allocation = std::make_unique<MockedOrtAllocator>();

  // prepare expected outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  // Create session options and configure it appropriately
  Ort::SessionOptions session_options;
  // Turn on sharing of the allocator between sessions
  session_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");

  const auto& api = Ort::GetApi();

  // CASE 1: We test creating and registering an ORT-internal allocator implementation instance
  // for sharing between sessions
  {
    OrtMemoryInfo* mem_info = nullptr;
    ASSERT_TRUE(api.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info) == nullptr);
    std::unique_ptr<OrtMemoryInfo, decltype(api.ReleaseMemoryInfo)> rel_info(mem_info, api.ReleaseMemoryInfo);

    OrtArenaCfg* arena_cfg = nullptr;
    ASSERT_TRUE(api.CreateArenaCfg(0, -1, -1, -1, &arena_cfg) == nullptr);
    std::unique_ptr<OrtArenaCfg, decltype(api.ReleaseArenaCfg)> rel_arena_cfg(arena_cfg, api.ReleaseArenaCfg);

    // This creates an ORT-internal allocator instance and registers it in the environment for sharing
    // NOTE: On x86 builds arenas are not supported and will default to using non-arena based allocator
    ASSERT_TRUE(api.CreateAndRegisterAllocator(env_ptr, mem_info, arena_cfg) == nullptr);

    // Test that duplicates are handled
    std::unique_ptr<OrtStatus, decltype(api.ReleaseStatus)> status_releaser(
        api.CreateAndRegisterAllocator(env_ptr, mem_info, arena_cfg),
        api.ReleaseStatus);
    ASSERT_FALSE(status_releaser.get() == nullptr);

    {
      // create session 1
      Ort::Session session1(*ort_env, MODEL_URI, session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session1,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);

      // create session 2
      Ort::Session session2(*ort_env, MODEL_URI, session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session2,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);
    }

    // Remove the registered shared allocator for part 2 of this test
    // where-in we will register a custom allocator for the same device.
    ASSERT_TRUE(api.UnregisterAllocator(env_ptr, mem_info) == nullptr);
  }

  // CASE 2: We test registering a custom allocator implementation
  // for sharing between sessions
  {
    // This creates a custom  allocator instance and registers it in the environment for sharing
    // NOTE: This is a very basic allocator implementation. For optimal performance, allocations
    // need to be aligned for certain devices/build configurations/math libraries.
    // See docs/C_API.md for details.
    MockedOrtAllocator custom_allocator;
    ASSERT_TRUE(api.RegisterAllocator(env_ptr, &custom_allocator) == nullptr);

    // Test that duplicates are handled
    std::unique_ptr<OrtStatus, decltype(api.ReleaseStatus)>
        status_releaser(
            api.RegisterAllocator(env_ptr, &custom_allocator),
            api.ReleaseStatus);
    ASSERT_FALSE(status_releaser.get() == nullptr);

    {
      // Keep this scoped to destroy the underlying sessions after use
      // This should trigger frees in our custom allocator

      // create session 1
      Ort::Session session1(*ort_env, MODEL_URI, session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session1,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);

      // create session 2
      Ort::Session session2(*ort_env, MODEL_URI, session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session2,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);
    }

    // Remove the registered shared allocator from the global environment
    // (common to all tests) to prevent its accidental usage elsewhere
    ASSERT_TRUE(api.UnregisterAllocator(env_ptr, custom_allocator.Info()) == nullptr);

    // Ensure that the registered custom allocator was indeed used for both sessions
    // We should have seen 2 allocations per session (one for the sole initializer
    // and one for the output). So, for two sessions, we should have seen 4 allocations.
    size_t num_allocations = custom_allocator.NumAllocations();
    ASSERT_TRUE(num_allocations == 4);

    // Ensure that there was no leak
    custom_allocator.LeakCheck();
  }
}

TEST(CApiTest, TestSharingOfInitializerAndItsPrepackedVersion) {
  // simple inference test
  // prepare inputs
  std::vector<Input> inputs(1);
  Input& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 1};
  std::vector<float> expected_values_y = {4.0f, 10.0f, 16.0f};

  Ort::SessionOptions session_options;
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // These values are different from the actual initializer values in the model
  float data[] = {2.0f, 1.0f};
  constexpr int data_len = sizeof(data) / sizeof(data[0]);
  const int64_t shape[] = {2, 1};
  constexpr size_t shape_len = sizeof(shape) / sizeof(shape[0]);
  Ort::Value val = Ort::Value::CreateTensor<float>(mem_info, data, data_len, shape, shape_len);
  session_options.AddInitializer("W", val);

  const auto& api = Ort::GetApi();

  OrtPrepackedWeightsContainer* prepacked_weights_container = nullptr;
  ASSERT_TRUE(api.CreatePrepackedWeightsContainer(&prepacked_weights_container) == nullptr);
  std::unique_ptr<OrtPrepackedWeightsContainer, decltype(api.ReleasePrepackedWeightsContainer)>
      rel_prepacked_weights_container(prepacked_weights_container, api.ReleasePrepackedWeightsContainer);

  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  // create session 1
  Ort::Session session1(*ort_env, MATMUL_MODEL_URI, session_options, prepacked_weights_container);
  RunSession<float>(default_allocator.get(),
                    session1,
                    inputs,
                    "Y",
                    expected_dims_y,
                    expected_values_y,
                    nullptr);

  // create session 2
  Ort::Session session2(*ort_env, MATMUL_MODEL_URI, session_options, prepacked_weights_container);
  RunSession<float>(default_allocator.get(),
                    session2,
                    inputs,
                    "Y",
                    expected_dims_y,
                    expected_values_y,
                    nullptr);
}

#ifndef ORT_NO_RTTI
TEST(CApiTest, TestIncorrectInputTypeToModel_Tensors) {
  // simple inference test
  // prepare inputs (incorrect type)
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  double data[] = {2., 1., 4., 3., 6., 5.};
  constexpr int data_len = sizeof(data) / sizeof(data[0]);
  const int64_t shape[] = {3, 2};
  constexpr size_t shape_len = sizeof(shape) / sizeof(shape[0]);
  Ort::Value val = Ort::Value::CreateTensor<double>(mem_info, data, data_len, shape, shape_len);

  std::vector<const char*> input_names{"X"};
  const char* output_names[] = {"Y"};
  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, MODEL_URI, session_options);
  bool exception_thrown = false;
  try {
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &val, 1, output_names, 1);
  } catch (const Ort::Exception& ex) {
    exception_thrown = true;
    const char* exception_string = ex.what();
    ASSERT_TRUE(strcmp(exception_string,
                       "Unexpected input data type. Actual: (tensor(double)) , expected: (tensor(float))") == 0);
  }

  ASSERT_TRUE(exception_thrown);
}
TEST(CApiTest, TestIncorrectInputTypeToModel_SequenceTensors) {
  // simple inference test
  // prepare inputs (incorrect type)
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  double data[] = {2., 1., 4., 3., 6., 5.};
  constexpr int data_len = sizeof(data) / sizeof(data[0]);
  const int64_t shape[] = {2, 3};
  constexpr size_t shape_len = sizeof(shape) / sizeof(shape[0]);
  Ort::Value val = Ort::Value::CreateTensor<double>(mem_info, data, data_len, shape, shape_len);

  std::vector<Ort::Value> seq;
  seq.push_back(std::move(val));

  Ort::Value seq_value = Ort::Value::CreateSequence(seq);

  std::vector<const char*> input_names{"X"};
  const char* output_names[] = {"Y"};
  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, SEQUENCE_MODEL_URI, session_options);
  bool exception_thrown = false;
  try {
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &seq_value, 1, output_names, 1);
  } catch (const Ort::Exception& ex) {
    exception_thrown = true;
    const char* exception_string = ex.what();
    ASSERT_TRUE(strcmp(exception_string,
                       "Unexpected input data type. Actual: (seq(double)) , expected: (seq(float))") == 0);
  }

  ASSERT_TRUE(exception_thrown);
}
#endif

TEST(CApiTest, AllocateInitializersFromNonArenaMemory) {
  Ort::SessionOptions session_options;

#ifdef USE_CUDA
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#else
  // arena is enabled but the sole initializer will still be allocated from non-arena memory
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 1));
#endif

  // disable using arena for the sole initializer in the model
  session_options.AddConfigEntry(kOrtSessionOptionsUseDeviceAllocatorForInitializers, "1");

  // This is mostly an usage example - if the logging level for the default logger is made INFO (by default it is at WARNING)
  // when the Ort::Env instance is instantiated, logs pertaining to initializer memory being allocated from non-arena memory
  // can be confirmed by seeing logs like "Reserving memory in BFCArena...".
  Ort::Session session(*ort_env, MODEL_URI, session_options);
}

#ifdef USE_CUDA

// Usage example showing how to use CreateArenaCfgV2() API to configure the default memory CUDA arena allocator
TEST(CApiTest, ConfigureCudaArenaAndDemonstrateMemoryArenaShrinkage) {
  const auto& api = Ort::GetApi();

  Ort::SessionOptions session_options;

  const char* keys[] = {"max_mem", "arena_extend_strategy", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk", "initial_growth_chunk_size_bytes"};
  const size_t values[] = {0 /*let ort pick default max memory*/, 0, 1024, 0, 256};

  OrtArenaCfg* arena_cfg = nullptr;
  ASSERT_TRUE(api.CreateArenaCfgV2(keys, values, 5, &arena_cfg) == nullptr);
  std::unique_ptr<OrtArenaCfg, decltype(api.ReleaseArenaCfg)> rel_arena_cfg(arena_cfg, api.ReleaseArenaCfg);

  OrtCUDAProviderOptions cuda_provider_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(nullptr);
  cuda_provider_options.default_memory_arena_cfg = arena_cfg;

  session_options.AppendExecutionProvider_CUDA(cuda_provider_options);

  Ort::Session session(*ort_env, MODEL_URI, session_options);

  // Use a run option like this while invoking Run() to trigger a memory arena shrinkage post Run()
  // This will shrink memory allocations left unused at the end of Run() and cap the arena growth
  // This does come with associated costs as there are costs to cudaFree() but the goodness it offers
  // is that the memory held by the arena (memory pool) is kept checked.
  Ort::RunOptions run_option;
  run_option.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "gpu:0");

  // To also trigger a cpu memory arena shrinkage along with the gpu arena shrinkage, use the following-
  // (Memory arena for the CPU should not have been disabled)
  //  run_option.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "cpu:0;gpu:0");
}
#endif

#ifdef USE_TENSORRT

// This test uses CreateTensorRTProviderOptions/UpdateTensorRTProviderOptions APIs to configure and create a TensorRT Execution Provider
TEST(CApiTest, TestConfigureTensorRTProviderOptions) {
  const auto& api = Ort::GetApi();
  OrtTensorRTProviderOptionsV2* trt_options;
  OrtAllocator* allocator;
  char* trt_options_str;
  ASSERT_TRUE(api.CreateTensorRTProviderOptions(&trt_options) == nullptr);
  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(trt_options, api.ReleaseTensorRTProviderOptions);

  const char* engine_cache_path = "./trt_engine_folder";

  std::vector<const char*> keys{"device_id", "trt_fp16_enable", "trt_int8_enable", "trt_engine_cache_enable", "trt_engine_cache_path"};

  std::vector<const char*> values{"0", "1", "0", "1", engine_cache_path};

  ASSERT_TRUE(api.UpdateTensorRTProviderOptions(rel_trt_options.get(), keys.data(), values.data(), 5) == nullptr);

  ASSERT_TRUE(api.GetAllocatorWithDefaultOptions(&allocator) == nullptr);
  ASSERT_TRUE(api.GetTensorRTProviderOptionsAsString(rel_trt_options.get(), allocator, &trt_options_str) == nullptr);
  std::string s(trt_options_str);
  ASSERT_TRUE(s.find(engine_cache_path) != std::string::npos);
  ASSERT_TRUE(api.AllocatorFree(allocator, (void*)trt_options_str) == nullptr);

  Ort::SessionOptions session_options;
  ASSERT_TRUE(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options), rel_trt_options.get()) == nullptr);

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
  std::basic_string<ORTCHAR_T> model_uri = MODEL_URI;

  // if session creation passes, model loads fine
  Ort::Session session(*ort_env, model_uri.c_str(), session_options);
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  //without preallocated output tensor
  RunSession(default_allocator.get(),
             session,
             inputs,
             "Y",
             expected_dims_y,
             expected_values_y,
             nullptr);

  struct stat buffer;
  ASSERT_TRUE(stat(engine_cache_path, &buffer) == 0);
}
#endif

#ifdef USE_CUDA

// This test uses CreateCUDAProviderOptions/UpdateCUDAProviderOptions APIs to configure and create a CUDA Execution Provider instance
TEST(CApiTest, TestConfigureCUDAProviderOptions) {
  const auto& api = Ort::GetApi();

  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  ASSERT_TRUE(api.CreateCUDAProviderOptions(&cuda_options) == nullptr);
  std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);

  std::vector<const char*> keys{
      "device_id", "gpu_mem_limit", "arena_extend_strategy",
      "cudnn_conv_algo_search", "do_copy_in_default_stream", "cudnn_conv_use_max_workspace"};

  std::vector<const char*> values{
      "0", "1024", "kSameAsRequested",
      "DEFAULT", "1", "1"};

  ASSERT_TRUE(api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 6) == nullptr);

  OrtAllocator* allocator;
  ASSERT_TRUE(api.GetAllocatorWithDefaultOptions(&allocator) == nullptr);

  char* cuda_options_str = nullptr;
  ASSERT_TRUE(api.GetCUDAProviderOptionsAsString(rel_cuda_options.get(), allocator, &cuda_options_str) == nullptr);
  std::string s(cuda_options_str, strnlen(cuda_options_str, 2048));
  ASSERT_TRUE(s.find("device_id=0") != std::string::npos);
  ASSERT_TRUE(s.find("gpu_mem_limit=1024") != std::string::npos);
  ASSERT_TRUE(s.find("arena_extend_strategy=kSameAsRequested") != std::string::npos);
  ASSERT_TRUE(s.find("cudnn_conv_algo_search=DEFAULT") != std::string::npos);
  ASSERT_TRUE(s.find("do_copy_in_default_stream=1") != std::string::npos);
  ASSERT_TRUE(s.find("cudnn_conv_use_max_workspace=1") != std::string::npos);

  ASSERT_TRUE(api.AllocatorFree(allocator, (void*)cuda_options_str) == nullptr);

  Ort::SessionOptions session_options;
  ASSERT_TRUE(api.SessionOptionsAppendExecutionProvider_CUDA_V2(static_cast<OrtSessionOptions*>(session_options), rel_cuda_options.get()) == nullptr);

  // if session creation passes, model loads fine
  std::basic_string<ORTCHAR_T> model_uri = MODEL_URI;
  Ort::Session session(*ort_env, model_uri.c_str(), session_options);
}

#endif

namespace TestPerSessionCustomThreadHooks {

std::vector<std::thread> threads;
int32_t custom_thread_creation_options = 5;
int32_t custom_creation_hook_called = 0;
int32_t custom_join_hook_called = 0;

OrtCustomThreadHandle CreateThreadCustomized(void* options, OrtThreadWorkerFn work_loop, void* param) {
  if (*((int32_t*)options) == 5) {
    custom_creation_hook_called += 1;
  }
  threads.push_back(std::thread(work_loop, param));
  return reinterpret_cast<OrtCustomThreadHandle>(threads.back().native_handle());
}

void JoinThreadCustomized(OrtCustomThreadHandle handle) {
  for (auto& t : threads) {
    if (reinterpret_cast<OrtCustomThreadHandle>(t.native_handle()) == handle) {
      custom_join_hook_called += 1;
      t.join();
    }
  }
}

TEST(CApiTest, TestPerSessionCustomThreadPoolHooks) {
  constexpr int32_t thread_count = 3;
  Ort::SessionOptions session_options;
  // test both intra and inter op thread pool
  session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  session_options.SetIntraOpNumThreads(thread_count);
  session_options.SetInterOpNumThreads(thread_count);
  session_options.SetCustomCreateThreadFn(CreateThreadCustomized);
  session_options.SetCustomThreadCreationOptions(&custom_thread_creation_options);
  session_options.SetCustomJoinThreadFn(JoinThreadCustomized);
  {
    Ort::Session session(*ort_env, MODEL_URI, session_options);
  }
  ASSERT_TRUE(custom_creation_hook_called == (thread_count - 1) << 1);
  ASSERT_TRUE(custom_join_hook_called == (thread_count - 1) << 1);
}

// Preventing resize tranformer issue:
// https://github.com/microsoft/onnxruntime/issues/9857
#ifndef REDUCED_OPS_BUILD
TEST(CApiTest, crop_and_resize) {
  std::vector<float> input_value_0;
  input_value_0.resize(2 * 36 * 36 * 3);
  for (ptrdiff_t i = 0; i < 36 * 36 * 3; ++i) {
    input_value_0[i] = 1.f;
    input_value_0[i + 36 * 36 * 3] = 2.f;
  }
  std::vector<int64_t> input_shape_0{2, 36, 36, 3};

  std::vector<int32_t> input_value_1{1, 0};
  std::vector<int64_t> input_shape_1{2};

  std::vector<const char*> input_names{"input:0", "input2:0"};
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(info, input_value_0.data(), input_value_0.size(), input_shape_0.data(), input_shape_0.size()));
  ort_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(info, input_value_1.data(), input_value_1.size(), input_shape_1.data(), input_shape_1.size()));

  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, RESIZE_AND_CROP_MODEL_URI, session_options);

  const char* output_names[] = {"output:0"};
  std::vector<int64_t> output_shape{2, 20, 20, 3};

  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, countof(output_names));
  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& output_0 = ort_outputs[0];
  ASSERT_TRUE(output_0.IsTensor());

  auto output_type_shape = output_0.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, output_type_shape.GetElementType());
  ASSERT_EQ(output_shape, output_type_shape.GetShape());
}
#endif

}  // namespace TestPerSessionCustomThreadHooks

#ifdef USE_CUDA
TEST(CApiTest, GitHubIssue10179) {
  // https://github.com/microsoft/onnxruntime/issues/10179
  // the issue was caused by a race condition in CUDAExecutionProvider::GetKernelRegistry()
  // if the test runs to completion, consider that run successful
  auto load_model_thread_fn = []() {
    try {
      const auto* model_path = MODEL_URI;
      Ort::SessionOptions session_options{};
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
      Ort::Session session{*ort_env, model_path, session_options};
    } catch (const std::exception& e) {
      std::cerr << "exception: " << e.what() << "\n";
      throw e;
    }
  };

  constexpr int num_threads = 4;
  constexpr int num_iterations = 10;

  for (int i = 0; i < num_iterations; ++i) {
    std::vector<std::thread> threads(num_threads);
    for (auto& thread : threads) {
      thread = std::thread{load_model_thread_fn};
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }
}

#endif

// Reduced Ops build doesn't support If (16) yet
#if !defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
TEST(CApiTest, TestCudaMemcpyToHostWithSequenceTensors) {
  const auto* model_path = SEQUENCE_MODEL_URI_2;
  Ort::SessionOptions session_options{};
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
  Ort::Session session{*ort_env, model_path, session_options};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names{"cond"};
  bool input_data[] = {false};
  std::vector<int64_t> input_dims{};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<bool>(info, input_data, 1U, input_dims.data(), 0));
  const char* output_names[] = {"sequence"};

  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                                    ort_inputs.data(), ort_inputs.size(),
                                                    output_names, countof(output_names));

  // There is no need to check the contents of the output, we are just checking to see if the
  // model runs without crashing
}

#endif
