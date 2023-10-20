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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/util/thread_utils.h"

#include "onnxruntime_config.h"
#include "providers.h"
#include "test_allocator.h"
#include "test_fixture.h"
#include "utils.h"
#include "custom_op_utils.h"
#include "core/common/gsl.h"

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

template <typename OutT, typename InT = float, typename InputT = Input>
void RunSession(OrtAllocator* allocator, Ort::Session& session_object,
                const std::vector<InputT>& inputs,
                const char* output_name,
                const std::vector<int64_t>& dims_y,
                const std::vector<OutT>& values_y,
                Ort::Value* output_tensor) {
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(
        Ort::Value::CreateTensor<InT>(allocator->Info(allocator), const_cast<InT*>(inputs[i].values.data()),
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
    if constexpr (std::is_same<OutT, float>::value || std::is_same<OutT, double>::value) {
      ASSERT_NEAR(values_y[i], f[i], 1e-3);
    } else {
      ASSERT_EQ(values_y[i], f[i]);
    }
  }
}

template <typename OutT, typename InT = float, typename InputT = Input>
static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
                          const std::vector<InputT>& inputs,
                          const char* output_name,
                          const std::vector<int64_t>& expected_dims_y,
                          const std::vector<OutT>& expected_values_y,
                          int provider_type,
                          OrtCustomOpDomain* custom_op_domain_ptr,
                          const ORTCHAR_T* custom_op_library_filename,
                          bool test_session_creation_only = false,
                          void* cuda_compute_stream = nullptr,
                          Ort::SessionOptions* predefined_session_options = nullptr) {
  Ort::SessionOptions default_session_options;
  Ort::SessionOptions& session_options = predefined_session_options ? *predefined_session_options
                                                                    : default_session_options;

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
    OrtDnnlProviderOptions dnnl_options;
    dnnl_options.use_arena = 1;
    dnnl_options.threadpool_args = nullptr;
    session_options.AppendExecutionProvider_Dnnl(dnnl_options);
#else
    return;
#endif
  } else if (provider_type == 3) {
#ifdef USE_ROCM
    OrtROCMProviderOptions rocm_options;
    session_options.AppendExecutionProvider_ROCM(rocm_options);
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
    session_options.RegisterCustomOpsLibrary(custom_op_library_filename);
  }

  // if session creation passes, model loads fine
  Ort::Session session(env, model_uri.c_str(), session_options);

  // caller wants to test running the model (not just loading the model)
  if (!test_session_creation_only) {
    // Now run
    auto default_allocator = std::make_unique<MockedOrtAllocator>();

    // without preallocated output tensor
    RunSession<OutT, InT, InputT>(default_allocator.get(),
                                  session,
                                  inputs,
                                  output_name,
                                  expected_dims_y,
                                  expected_values_y,
                                  nullptr);
    // with preallocated output tensor
    Ort::Value value_y = Ort::Value::CreateTensor<OutT>(default_allocator.get(),
                                                        expected_dims_y.data(), expected_dims_y.size());

    // test it twice
    for (int i = 0; i != 2; ++i)
      RunSession<OutT, InT, InputT>(default_allocator.get(),
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
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_ATTR_TESTER_URI = TSTR("testdata/custom_op_library/attr_tester.onnx");
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_TEST_MODEL_URI = TSTR("testdata/custom_op_library/custom_op_test.onnx");
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_COPY_TENSOR_ARRAY_2 = TSTR("testdata/custom_op_library/copy_2_inputs_2_outputs.onnx");
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_COPY_TENSOR_ARRAY_3 = TSTR("testdata/custom_op_library/copy_3_inputs_3_outputs.onnx");
#if !defined(DISABLE_FLOAT8_TYPES)
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_TEST_MODEL_FLOAT8_URI = TSTR("testdata/custom_op_library/custom_op_test_float8.onnx");
#endif
#if defined(USE_OPENVINO) && (!defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS))
static constexpr PATH_TYPE CUSTOM_OP_OPENVINO_WRAPPER_LIB_TEST_MODEL_URI = TSTR(
    "testdata/custom_op_openvino_wrapper_library/custom_op_mnist_ov_wrapper.onnx");
#endif
static constexpr PATH_TYPE OVERRIDABLE_INITIALIZER_MODEL_URI = TSTR("testdata/overridable_initializer.onnx");
static constexpr PATH_TYPE NAMED_AND_ANON_DIM_PARAM_URI = TSTR("testdata/capi_symbolic_dims.onnx");
static constexpr PATH_TYPE MODEL_WITH_CUSTOM_MODEL_METADATA = TSTR("testdata/model_with_valid_ort_config_json.onnx");
static constexpr PATH_TYPE VARIED_INPUT_CUSTOM_OP_MODEL_URI = TSTR("testdata/VariedInputCustomOp.onnx");
static constexpr PATH_TYPE VARIED_INPUT_CUSTOM_OP_MODEL_URI_2 = TSTR("testdata/foo_3.onnx");
static constexpr PATH_TYPE OPTIONAL_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI = TSTR("testdata/foo_bar_1.onnx");
static constexpr PATH_TYPE OPTIONAL_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI_2 = TSTR("testdata/foo_bar_2.onnx");
static constexpr PATH_TYPE VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI = TSTR("testdata/custom_op_variadic_io.onnx");
static constexpr PATH_TYPE VARIADIC_UNDEF_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI = TSTR(
    "testdata/custom_op_variadic_undef_io.onnx");
static constexpr PATH_TYPE CUSTOM_OP_MODEL_WITH_ATTRIBUTES_URI = TSTR("testdata/foo_bar_3.onnx");
static constexpr PATH_TYPE CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL = TSTR("testdata/custom_op_single_schema_multi_kernel.onnx");
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

static constexpr PATH_TYPE SIMPLIFIED_SSD_MODEL_URI = TSTR("testdata/multi_stream_models/simplified_ssd.onnx");

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
  auto dims = in0_ttsi.GetShape();
  if (!dims.empty()) dim_value = dims[0];
  in0_ttsi.GetSymbolicDimensions(&dim_param, 1);
  ASSERT_EQ(dim_value, -1) << "symbolic dimension should be -1";
  ASSERT_EQ(strcmp(dim_param, "n"), 0) << "Expected 'n'. Got: " << dim_param;

  auto out0 = session.GetOutputTypeInfo(0);
  auto out0_ttsi = out0.GetTensorTypeAndShapeInfo();
  auto num_output_dims = out0_ttsi.GetDimensionsCount();
  ASSERT_EQ(num_output_dims, 1u);

  dim_value = 0;
  dims = out0_ttsi.GetShape();
  if (!dims.empty()) dim_value = dims[0];

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
  ASSERT_TRUE(std::equal(values.cbegin(), values.cend(), val_span.begin(), val_span.end()));

  auto indices_ts = sparse_output.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_COO_INDICES);
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
  ASSERT_EQ(indices_shape, indices_ts.GetShape());

  size_t num_indices = 0;
  const int64_t* indices = sparse_output.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_COO_INDICES, num_indices);
  ASSERT_EQ(num_indices, static_cast<size_t>(indices_shape[0]));
  auto ind_span = gsl::make_span(indices, num_indices);
  ASSERT_TRUE(std::equal(coo_indices.cbegin(), coo_indices.cend(), ind_span.begin(), ind_span.end()));
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
  ASSERT_TRUE(std::equal(Y_result.cbegin(), Y_result.cend(), result_span.begin(), result_span.end()));
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
  MyCustomOp custom_op{onnxruntime::kCudaExecutionProvider};
#else
  MyCustomOp custom_op{onnxruntime::kCpuExecutionProvider};
#endif

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

#ifdef USE_CUDA
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 1,
                       custom_op_domain, nullptr, false, compute_stream);
  cudaStreamDestroy(compute_stream);
#else
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       custom_op_domain, nullptr);
#endif
}

#ifdef USE_CUDA
TEST(CApiTest, custom_op_set_input_memory_type) {
  std::cout << "Running custom op inference" << std::endl;

  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  cudaStream_t compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  MyCustomOpSecondInputOnCpu custom_op{onnxruntime::kCudaExecutionProvider, compute_stream};

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  auto x_mem_type = custom_op.GetInputMemoryType(0);
  auto y_mem_type = custom_op.GetInputMemoryType(1);
  ASSERT_EQ(x_mem_type, OrtMemType::OrtMemTypeDefault);
  ASSERT_EQ(y_mem_type, OrtMemType::OrtMemTypeCPUInput);

  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 1,
                       custom_op_domain, nullptr, false, compute_stream);
  cudaStreamDestroy(compute_stream);
}
#endif

#if !defined(ORT_MINIMAL_BUILD)
TEST(CApiTest, StandaloneOpHandler) {
  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

#ifdef USE_CUDA
  StandaloneCustomOp standalone_op{onnxruntime::kCudaExecutionProvider};
#else
  StandaloneCustomOp standalone_op{onnxruntime::kCpuExecutionProvider};
#endif

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&standalone_op);

#ifdef USE_CUDA
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 1,
                       custom_op_domain, nullptr);
#else
  Ort::SessionOptions session_options;
  const std::basic_string<ORTCHAR_T> ort_file = ORT_TSTR("testdata/foo_1.onnx.test_output.ort");
  session_options.SetOptimizedModelFilePath(ort_file.c_str());

  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       custom_op_domain, nullptr, false, nullptr, &session_options);

  TestInference<float>(*ort_env, ort_file, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       custom_op_domain, nullptr);
#endif
}
#endif

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

// test custom op which accepts float and double as inputs
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
  MyCustomOpMultipleDynamicInputs custom_op{onnxruntime::kCudaExecutionProvider};
#else
  MyCustomOpMultipleDynamicInputs custom_op{onnxruntime::kCpuExecutionProvider};
#endif

  Ort::CustomOpDomain custom_op_domain("test");
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

TEST(CApiTest, variadic_input_output_custom_op) {
  // Create a custom op with 1 variadic input and 1 variadic output.
  // The model passes in 3 string inputs and expects 3 int64_t outputs.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  std::vector<Ort::Value> ort_inputs;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::vector<int64_t>> expected_dims;
  std::vector<std::vector<int64_t>> expected_lens;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // Create inputs.
  AddInputForCustomStringLengthsKernel("hello", allocator, ort_inputs, input_names, output_names, expected_dims,
                                       expected_lens);
  AddInputForCustomStringLengthsKernel("", allocator, ort_inputs, input_names, output_names, expected_dims,
                                       expected_lens);
  AddInputForCustomStringLengthsKernel("123", allocator, ort_inputs, input_names, output_names, expected_dims,
                                       expected_lens);

  // Create arrays of c-strings for input and output names.
  auto get_c_str = [](const std::string& str) { return str.c_str(); };
  std::vector<const char*> input_name_cstrs(input_names.size());
  std::transform(input_names.begin(), input_names.end(), input_name_cstrs.begin(), get_c_str);
  std::vector<const char*> output_name_cstrs(output_names.size());
  std::transform(output_names.begin(), output_names.end(), output_name_cstrs.begin(), get_c_str);

  Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_name_cstrs.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_name_cstrs.data(), output_name_cstrs.size());
  ASSERT_EQ(ort_outputs.size(), 3u);

  // Validate outputs.
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    auto type_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), expected_dims[i]);
    ASSERT_EQ(type_info.GetElementCount(), 1u);

    int64_t* lens_data = ort_outputs[i].GetTensorMutableData<int64_t>();
    ASSERT_EQ(lens_data[0], expected_lens[i][0]);
  }
}

TEST(CApiTest, mixed_variadic_input_output_custom_op) {
  // Create a custom op with 2 inputs (required, variadic) and 2 outputs (required, variadic).
  // The model passes in 3 string inputs and expects 3 int64_t outputs.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  std::vector<Ort::Value> ort_inputs;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::vector<int64_t>> expected_dims;
  std::vector<std::vector<int64_t>> expected_lens;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // Create inputs.
  AddInputForCustomStringLengthsKernel("mixed variadic", allocator, ort_inputs, input_names, output_names,
                                       expected_dims, expected_lens);
  AddInputForCustomStringLengthsKernel("", allocator, ort_inputs, input_names, output_names, expected_dims,
                                       expected_lens);
  AddInputForCustomStringLengthsKernel("abcd", allocator, ort_inputs, input_names, output_names, expected_dims,
                                       expected_lens);

  // Create arrays of c-strings for input and output names.
  auto get_c_str = [](const std::string& str) { return str.c_str(); };
  std::vector<const char*> input_name_cstrs(input_names.size());
  std::transform(input_names.begin(), input_names.end(), input_name_cstrs.begin(), get_c_str);
  std::vector<const char*> output_name_cstrs(output_names.size());
  std::transform(output_names.begin(), output_names.end(), output_name_cstrs.begin(), get_c_str);

  Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_name_cstrs.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_name_cstrs.data(), output_name_cstrs.size());
  ASSERT_EQ(ort_outputs.size(), 3u);

  // Validate outputs.
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    auto type_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), expected_dims[i]);
    ASSERT_EQ(type_info.GetElementCount(), 1u);

    int64_t* lens_data = ort_outputs[i].GetTensorMutableData<int64_t>();
    ASSERT_EQ(lens_data[0], expected_lens[i][0]);
  }
}

TEST(CApiTest, variadic_undef_input_output_custom_op) {
  // Create a custom op with 1 variadic input and 1 variadic output.
  // Both the input and output are of undefined element type and allowed to differ in type (hetergeneous).
  // The model passes in inputs (string, int64_t, and float) which are then echoed in
  // reversed order (float, int64_t, string).
  TemplatedCustomOp<MyCustomEchoReversedArgsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      false,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      false);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  std::vector<Ort::Value> ort_inputs;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::ConstMemoryInfo mem_info = allocator.GetInfo();
  std::vector<int64_t> input_dims = {1};

  // Set string input.
  std::string str_input("hello_ort");
  Ort::Value& str_input_val = ort_inputs.emplace_back(
      Ort::Value::CreateTensor(allocator, input_dims.data(), input_dims.size(),
                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING));
  str_input_val.FillStringTensorElement(str_input.c_str(), 0);

  // Set int64_t input.
  std::array<int64_t, 1> int_inps = {23};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, int_inps.data(), int_inps.size(),
                                                            input_dims.data(), input_dims.size()));

  // Set float input.
  std::array<float, 1> float_inps = {10.0f};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem_info, float_inps.data(), float_inps.size(),
                                                          input_dims.data(), input_dims.size()));

  constexpr std::array<const char*, 3> input_names = {"input_0", "input_1", "input_2"};
  constexpr std::array<const char*, 3> output_names = {"output_0", "output_1", "output_2"};

  Ort::Session session(*ort_env, VARIADIC_UNDEF_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_names.data(), output_names.size());
  ASSERT_EQ(ort_outputs.size(), 3u);

  // Validate outputs.

  // First output should be a float.
  {
    auto& ort_output = ort_outputs[0];
    auto type_info = ort_output.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), input_dims);
    ASSERT_EQ(type_info.GetElementCount(), 1u);
    ASSERT_EQ(type_info.GetElementType(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    const float* out_ptr = ort_output.GetTensorData<float>();
    ASSERT_EQ(out_ptr[0], float_inps[0]);
  }

  // Second output should be a int64_t.
  {
    auto& ort_output = ort_outputs[1];
    auto type_info = ort_output.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), input_dims);
    ASSERT_EQ(type_info.GetElementCount(), 1u);
    ASSERT_EQ(type_info.GetElementType(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    const int64_t* out_ptr = ort_output.GetTensorData<int64_t>();
    ASSERT_EQ(out_ptr[0], int_inps[0]);
  }

  // Last output should be a string.
  {
    auto& ort_output = ort_outputs[2];
    auto type_info = ort_output.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), input_dims);
    ASSERT_EQ(type_info.GetElementCount(), 1u);
    ASSERT_EQ(type_info.GetElementType(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

    const size_t str_len = ort_output.GetStringTensorElementLength(0);
    ASSERT_EQ(str_len, str_input.size());

    std::string str;
    str.resize(str_len);

    ort_output.GetStringTensorElement(str_len, 0, str.data());
    ASSERT_EQ(str, str_input);
  }
}

TEST(CApiTest, invalid_variadic_input_not_last_custom_op) {
  // Create an invalid custom op with 2 inputs. The first input is variadic and the last is not.
  // Expect an error because only the last input may be marked as variadic.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED},
      1,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  try {
    Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_THAT(excpt.what(), testing::HasSubstr("Only the last input to a custom op may be marked variadic."));
  }
}

TEST(CApiTest, invalid_variadic_output_not_last_custom_op) {
  // Create an invalid custom op with 2 outputs. The first output is variadic and the last is not.
  // Expect an error because only the last output may be marked as variadic.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED},
      1,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  try {
    Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_THAT(excpt.what(), testing::HasSubstr("Only the last output to a custom op may be marked variadic."));
  }
}

TEST(CApiTest, invalid_variadic_input_min_arity_custom_op) {
  // Create a custom op with a variadic input with a minimum arity of 4.
  // Expect an error because the model passes in less than 4 inputs to the op.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      4,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  try {
    Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_THAT(excpt.what(), testing::HasSubstr("Error Node (VariadicNode0) has input size 3 not in range [min=4"));
  }
}

TEST(CApiTest, invalid_variadic_output_min_arity_custom_op) {
  // Create a custom op with a variadic output with a minimum arity of 4.
  // Expect an error because the model instantiates the op with less than 4 outputs.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      4,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  try {
    Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_THAT(excpt.what(), testing::HasSubstr("Error Node (VariadicNode0) has output size 3 not in range [min=4"));
  }
}

TEST(CApiTest, invalid_variadic_input_homogeneity_custom_op) {
  // Create a custom op with a homogeneous variadic input. The model has heterogeneous inputs,
  // so we expect an error.
  TemplatedCustomOp<MyCustomEchoReversedArgsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true,  // Input homogeneity requirement will cause error!
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      false);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  try {
    Ort::Session session(*ort_env, VARIADIC_UNDEF_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_THAT(excpt.what(), testing::HasSubstr("Type Error: Type parameter (Input0) of Optype (VariadicNode) bound "
                                                 "to different types"));
  }
}

TEST(CApiTest, optional_input_output_custom_op_handler) {
  MyCustomOpWithOptionalInput custom_op{onnxruntime::kCpuExecutionProvider};

  // `MyCustomOpFooBar` defines a custom op with atmost 3 inputs and the second input is optional.
  // In this test, we are going to try and run 2 models - one with the optional input and one without
  // the optional input.
  Ort::CustomOpDomain custom_op_domain("test");
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

  Ort::CustomOpDomain custom_op_domain("test");
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

  MyCustomOp custom_op_cpu{onnxruntime::kCpuExecutionProvider};
  // We are going to test session creation only - hence it is not a problem to use the default stream as the compute stream for the custom op
  MyCustomOp custom_op_cuda{onnxruntime::kCudaExecutionProvider};
  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op_cpu);
  custom_op_domain.Add(&custom_op_cuda);

  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y,
                       expected_values_y, 1, custom_op_domain, nullptr, true);
}
#endif

#if (!defined(ORT_MINIMAL_BUILD)) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
TEST(CApiTest, test_custom_op_get_const_input) {
  const auto* model_path = TSTR("testdata/test_kernel_info_get_const_input.onnx");

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;

  // input 0 (float type)
  input_names.emplace_back("input1");
  std::vector<float> input_0_data = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<int64_t> input_0_dims = {1, 4};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_0_data.data()),
                                      input_0_data.size(), input_0_dims.data(), input_0_dims.size()));
  const char* output_name = "output";

  const ORTCHAR_T* lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_get_const_input_test_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_get_const_input_test_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_get_const_input_test_library.so");
#endif

  Ort::SessionOptions session_opts;

  session_opts.RegisterCustomOpsLibrary(lib_name);

  Ort::Session session(*ort_env, model_path, session_opts);
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
              &output_name, 1);
}
#endif

#if defined(USE_OPENVINO) && (!defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS))
TEST(CApiTest, test_custom_op_openvino_wrapper_library) {
  // Tests a custom operator that wraps an OpenVINO MNIST model (.xml and .bin files serialized into node attributes).
  // The custom op extracts the serialized .xml/.bin bytes and creates an in-memory OpenVINO model
  // during kernel creation. The custom op is passed an image of a hand-drawn "1" as an input during computation, which
  // is then inferenced using OpenVINO C++ APIs.
  std::vector<Input> inputs(1);
  inputs[0].name = "Input3";
  inputs[0].dims = {1, 1, 28, 28};

  // Float image with the digit "1".
  inputs[0].values = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.75f, 1.0f, 0.75f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.85f, 0.99f, 0.85f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 1.0f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 1.0f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.99f, 1.0f, 0.99f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.94f, 0.99f, 0.94f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.75f, 0.75f, 0.75f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // prepare expected outputs
  std::vector<int64_t> expected_output_dims = {1, 10};

  // Digit 1 (index 1) has the highest probability (before applying softmax)
  std::vector<float> expected_vals = {-5.34957457f, 13.1904755f, -4.79670954f, -3.59232116f, 2.31260920f,
                                      -4.27866220f, -4.31867933f, 0.587718308f, -2.33952785f, -3.88515306f};

  const ORTCHAR_T* lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_openvino_wrapper_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_openvino_wrapper_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_openvino_wrapper_library.so");
#endif

  // Run with custom op session configurations.
  {
    Ort::SessionOptions session_opts;
    Ort::CustomOpConfigs custom_op_configs;

    custom_op_configs.AddConfig("OpenVINO_Wrapper", "device_type", "CPU");
    session_opts.RegisterCustomOpsLibrary(lib_name, custom_op_configs);

    Ort::Session session(*ort_env, CUSTOM_OP_OPENVINO_WRAPPER_LIB_TEST_MODEL_URI, session_opts);
    auto default_allocator = std::make_unique<MockedOrtAllocator>();

    RunSession(default_allocator.get(), session,
               inputs,
               "Plus214_Output_0",
               expected_output_dims,
               expected_vals,
               nullptr);
  }

  // Run without specifying any custom op session configurations.
  // Expect custom op to use "CPU" as OpenVINO's default backend.
  {
    Ort::SessionOptions session_opts;
    session_opts.RegisterCustomOpsLibrary(lib_name);

    Ort::Session session(*ort_env, CUSTOM_OP_OPENVINO_WRAPPER_LIB_TEST_MODEL_URI, session_opts);
    auto default_allocator = std::make_unique<MockedOrtAllocator>();

    RunSession(default_allocator.get(), session,
               inputs,
               "Plus214_Output_0",
               expected_output_dims,
               expected_vals,
               nullptr);
  }
}
#endif  // defined(USE_OPENVINO) && (!defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS))

// It has memory leak. The OrtCustomOpDomain created in custom_op_library.cc:RegisterCustomOps function was not freed
#if defined(__ANDROID__)
TEST(CApiTest, DISABLED_test_custom_op_library) {
// To accomodate a reduced op build pipeline
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

  onnxruntime::PathString lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_library.so");
#endif

#ifdef USE_CUDA
  TestInference<int32_t>(*ort_env, CUSTOM_OP_LIBRARY_TEST_MODEL_URI, inputs, "output", expected_dims_y,
                         expected_values_y, 1, nullptr, lib_name.c_str());
#elif USE_ROCM
  TestInference<int32_t>(*ort_env, CUSTOM_OP_LIBRARY_TEST_MODEL_URI, inputs, "output", expected_dims_y,
                         expected_values_y, 3, nullptr, lib_name.c_str());
#else
  TestInference<int32_t>(*ort_env, CUSTOM_OP_LIBRARY_TEST_MODEL_URI, inputs, "output", expected_dims_y,
                         expected_values_y, 0, nullptr, lib_name.c_str());
#endif
}

#if defined(__ANDROID__)
TEST(CApiTest, DISABLED_test_custom_op_shape_infer_attr) {
// To accomodate a reduced op build pipeline
#elif defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
TEST(CApiTest, DISABLED_test_custom_op_shape_infer_attr) {
#else
TEST(CApiTest, test_custom_op_shape_infer_attr) {
#endif
  std::vector<Input> inputs(1);
  inputs[0].name = "input_0";
  inputs[0].dims = {5};
  inputs[0].values = {1.f, 2.f, 3.f, 4.f, 5.f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {5};
  std::vector<float> expected_values_y = {6.f, 12.f, 18.f, 24.f, 30.f};

  onnxruntime::PathString lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_library.so");
#endif

  TestInference<float>(*ort_env, CUSTOM_OP_LIBRARY_ATTR_TESTER_URI, inputs, "output_0", expected_dims_y,
                       expected_values_y, 0, nullptr, lib_name.c_str());
}

// It has memory leak. The OrtCustomOpDomain created in custom_op_library.cc:RegisterCustomOps function was not freed
#if defined(__ANDROID__)
TEST(CApiTest, test_custom_op_library_copy_variadic) {
// To accomodate a reduced op build pipeline
#elif defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
TEST(CApiTest, test_custom_op_library_copy_variadic) {
#else
TEST(CApiTest, test_custom_op_library_copy_variadic) {
#endif
  std::cout << "Running inference using custom op shared library" << std::endl;

  std::vector<Input> inputs(2);
  inputs[0].name = "input_0";
  inputs[0].dims = {15};
  inputs[0].values = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                      6.6f, 7.7f, 8.8f, 9.9f, 10.0f,
                      11.1f, 12.2f, 13.3f, 14.4f, 15.5f};
  inputs[1].name = "input_1";
  inputs[1].dims = {15};
  inputs[1].values = {15.5f, 14.4f, 13.3f, 12.2f, 11.1f,
                      10.0f, 9.9f, 8.8f, 7.7f, 6.6f,
                      5.5f, 4.4f, 3.3f, 2.2f, 1.1f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {15};
  std::vector<float> expected_values_y = inputs[1].values;

  onnxruntime::PathString lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_library.so");
#endif

  TestInference<float>(*ort_env, CUSTOM_OP_LIBRARY_COPY_TENSOR_ARRAY_2,
                       inputs, "output_1", expected_dims_y,
                       expected_values_y, 0, nullptr, lib_name.c_str());

  inputs.push_back({});
  inputs[2].name = "input_2";
  inputs[2].dims = {15};
  inputs[2].values = {6.6f, 7.7f, 8.8f, 9.9f, 10.0f,
                      1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                      11.1f, 12.2f, 13.3f, 14.4f, 15.5f};

  expected_values_y = inputs[2].values;
  TestInference<float>(*ort_env, CUSTOM_OP_LIBRARY_COPY_TENSOR_ARRAY_3,
                       inputs, "output_2", expected_dims_y,
                       expected_values_y, 0, nullptr, lib_name.c_str());
}

#if !defined(DISABLE_FLOAT8_TYPES)

struct InputF8 {
  const char* name = nullptr;
  std::vector<int64_t> dims;
  std::vector<Ort::Float8E4M3FN_t> values;
};

// See test test_custom_op_library_float8.
#if defined(__ANDROID__)
TEST(CApiTest, DISABLED_test_custom_op_library_float8) {
#elif defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
TEST(CApiTest, DISABLED_test_custom_op_library_float8) {
#else
TEST(CApiTest, test_custom_op_library_float8) {
#endif
  std::cout << "Running inference using custom op shared library" << std::endl;

  std::vector<InputF8> inputs(2);
  inputs[0].name = "X";
  inputs[0].dims = {2};
  inputs[0].values = {0, 1};
  inputs[1].name = "Y";
  inputs[1].dims = {2};
  inputs[1].values = {3, 4};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {2};
  std::vector<Ort::Float8E4M3FN_t> expected_values_y = {0, 1};

  onnxruntime::PathString lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_library.so");
#endif

  TestInference<Ort::Float8E4M3FN_t, Ort::Float8E4M3FN_t, InputF8>(*ort_env, CUSTOM_OP_LIBRARY_TEST_MODEL_FLOAT8_URI, inputs,
                                                                   "Z", expected_dims_y,
                                                                   expected_values_y, 0, nullptr, lib_name.c_str());
}

#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#if defined(__ANDROID__)
// Disable on android because custom op libraries are not copied to the emulator.
TEST(CApiTest, DISABLED_test_custom_op_library_registration_error) {
#else
TEST(CApiTest, test_custom_op_library_registration_error) {
#endif  // defined(__ANDROID__)
  // Loads a custom op library with a RegisterCustomOps function that returns an error status.
  // This test tries to register the library with the session options and expects an error.
  const ORTCHAR_T* lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_invalid_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_invalid_library.dylib");
#else
lib_name = ORT_TSTR("./libcustom_op_invalid_library.so");
#endif

  Ort::SessionOptions session_options;

  try {
    session_options.RegisterCustomOpsLibrary(lib_name);
    FAIL();
  } catch (const Ort::Exception& exception) {
    ASSERT_THAT(exception.what(), testing::HasSubstr("Failure from custom op library's RegisterCustomOps()"));
  }
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

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
    // only test model loading, exception expected
    TestInference<float>(*ort_env, model_uri, inputs, "Y", expected_dims_y, expected_values_y, 0,
                         nullptr, nullptr, true);
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
  auto input_data = cuda_allocator.GetAllocation(x_values.size() * sizeof(float));
  ASSERT_NE(input_data.get(), nullptr);
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  const std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  auto output_data = cuda_allocator.GetAllocation(expected_y.size() * sizeof(float));
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

#if defined(USE_CUDA) || defined(USE_TENSORRT)
TEST(CApiTest, basic_cuda_graph) {
  const auto& api = Ort::GetApi();
  Ort::SessionOptions session_options;

#if defined(USE_TENSORRT)
  // Enable cuda graph in TRT provider option.
  OrtTensorRTProviderOptionsV2* trt_options;
  ASSERT_TRUE(api.CreateTensorRTProviderOptions(&trt_options) == nullptr);
  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)>
      rel_trt_options(trt_options, api.ReleaseTensorRTProviderOptions);
  std::vector<const char*> keys{"trt_cuda_graph_enable"};
  std::vector<const char*> values{"1"};
  ASSERT_TRUE(api.UpdateTensorRTProviderOptions(rel_trt_options.get(), keys.data(), values.data(), keys.size()) == nullptr);

  ASSERT_TRUE(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
                  static_cast<OrtSessionOptions*>(session_options),
                  rel_trt_options.get()) == nullptr);
#else
  // Enable cuda graph in cuda provider option.
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  ASSERT_TRUE(api.CreateCUDAProviderOptions(&cuda_options) == nullptr);
  std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)>
      rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
  std::vector<const char*> keys{"enable_cuda_graph"};
  std::vector<const char*> values{"1"};
  ASSERT_TRUE(api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 1) == nullptr);

  ASSERT_TRUE(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
                  static_cast<OrtSessionOptions*>(session_options),
                  rel_cuda_options.get()) == nullptr);
#endif

  Ort::Session session(*ort_env, MODEL_URI, session_options);
  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);

  Ort::Allocator cuda_allocator(session, info_cuda);
  auto allocator_info = cuda_allocator.GetInfo();
  ASSERT_TRUE(info_cuda == allocator_info);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = cuda_allocator.GetAllocation(x_values.size() * sizeof(float));

  ASSERT_NE(input_data.get(), nullptr);
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  auto output_data = cuda_allocator.GetAllocation(expected_y.size() * sizeof(float));

  ASSERT_NE(output_data.get(), nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data.get()),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  // Create IoBinding for inputs and outputs.
  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  // One regular run for necessary memory allocation and graph capturing
  session.Run(Ort::RunOptions(), binding);

  // Check the values against the bound raw memory (needs copying from device to host first)
  std::array<float, 3 * 2> y_values;
  cudaMemcpy(y_values.data(), output_data.get(), sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);
  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Replay the captured CUDA graph
  session.Run(Ort::RunOptions(), binding);
  cudaMemcpy(y_values.data(), output_data.get(), sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);
  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Change the input and replay the CUDA graph again.
  x_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);
  binding.SynchronizeInputs();

  session.Run(Ort::RunOptions(), binding);
  cudaMemcpy(y_values.data(), output_data.get(), sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);
  expected_y = {10.0f, 40.0f, 90.0f, 160.0f, 250.0f, 360.0f};
  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}

#ifndef REDUCED_OPS_BUILD
// The following test uses some ops not supported in the reduced ops build
TEST(CApiTest, cuda_graph_with_shape_nodes) {
  const auto& api = Ort::GetApi();

  // Enable cuda graph in cuda provider option.
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  ASSERT_TRUE(api.CreateCUDAProviderOptions(&cuda_options) == nullptr);
  std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)>
      rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
  std::vector<const char*> keys{"enable_cuda_graph"};
  std::vector<const char*> values{"1"};
  ASSERT_TRUE(api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 1) == nullptr);

  Ort::SessionOptions session_options;
  ASSERT_TRUE(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
                  static_cast<OrtSessionOptions*>(session_options),
                  rel_cuda_options.get()) == nullptr);

  // Successful loading of the ONNX model with shape nodes with cuda graph feature enabled
  Ort::Session session(*ort_env, TSTR("testdata/cuda_graph_with_shape_nodes.onnx"), session_options);
}

#endif

#endif

TEST(CApiTest, create_tensor) {
  const char* s[] = {"abc", "kmp"};
  constexpr int64_t expected_len = 2;
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  Ort::Value tensor = Ort::Value::CreateTensor(default_allocator.get(), &expected_len, 1,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  Ort::ThrowOnError(Ort::GetApi().FillStringTensor(tensor, s, expected_len));
  auto shape_info = tensor.GetTensorTypeAndShapeInfo();

  const auto len = shape_info.GetElementCount();
  ASSERT_EQ(len, static_cast<size_t>(expected_len));
  std::vector<int64_t> shape_array(len);

  size_t data_len = tensor.GetStringTensorDataLength();
  std::string result(data_len, '\0');
  std::vector<size_t> offsets(len);
  tensor.GetStringTensorContent((void*)result.data(), data_len, offsets.data(), offsets.size());
}

TEST(CApiTest, fill_string_tensor) {
  const char* s[] = {"abc", "kmp"};
  constexpr int64_t expected_len = 2;
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  Ort::Value tensor = Ort::Value::CreateTensor(default_allocator.get(), &expected_len, 1,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  for (size_t i = 0; i < expected_len; i++) {
    tensor.FillStringTensorElement(s[i], i);
  }

  auto shape_info = tensor.GetTensorTypeAndShapeInfo();

  const auto len = shape_info.GetElementCount();
  ASSERT_EQ(len, static_cast<size_t>(expected_len));
}

TEST(CApiTest, fill_string_tensor_directly) {
  constexpr std::string_view s[] = {"abc", "kmp"};
  constexpr int64_t expected_len = 2;

  MockedOrtAllocator default_allocator;
  Ort::Value tensor = Ort::Value::CreateTensor(&default_allocator, &expected_len, 1U,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  for (size_t i = 0; i < expected_len; i++) {
    auto* buffer = tensor.GetResizedStringTensorElementBuffer(i, s[i].size());
    memcpy(buffer, s[i].data(), s[i].size());
  }

  auto shape_info = tensor.GetTensorTypeAndShapeInfo();
  const auto len = shape_info.GetElementCount();
  ASSERT_EQ(len, static_cast<size_t>(expected_len));

  for (size_t i = 0; i < expected_len; i++) {
    auto element = tensor.GetStringTensorElement(i);
    ASSERT_EQ(s[i], element);
  }
}

TEST(CApiTest, get_string_tensor_element) {
  const char* s[] = {"abc", "kmp"};
  constexpr int64_t expected_len = 2;
  constexpr int64_t element_index = 0;
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
  constexpr float values[] = {1.f, 2.f, 3.f, 4.f, 5.f};
  constexpr size_t values_length = std::size(values);
  constexpr uint16_t expected_values[values_length] = {15360, 16384, 16896, 17408, 17664};

  std::vector<Ort::Float16_t> fp16_values;
  fp16_values.reserve(values_length);
  std::transform(std::begin(values), std::end(values), std::back_inserter(fp16_values),
                 [](float fl) { return Ort::Float16_t(fl); });

  for (size_t i = 0; i < values_length; ++i) {
    ASSERT_EQ(expected_values[i], fp16_values[i].val);
  }

  constexpr int64_t dims = static_cast<int64_t>(values_length);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor<Ort::Float16_t>(info, fp16_values.data(), values_length, &dims, 1u);
  const auto* new_pointer = tensor.GetTensorData<Ort::Float16_t>();

  ASSERT_EQ(new_pointer, fp16_values.data());
  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  const auto element_count = tensor_info.GetElementCount();
  ASSERT_EQ(values_length, element_count);
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1u, tensor_info.GetDimensionsCount());
  ASSERT_EQ(tensor_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  const Ort::Float16_t& value_at_1 = tensor.At<Ort::Float16_t>({1});
  ASSERT_EQ(expected_values[1], value_at_1.val);

  std::vector<float> output_values;
  output_values.reserve(values_length);
  const auto data_span = gsl::make_span(tensor.GetTensorData<Ort::Float16_t>(), element_count);
  std::transform(data_span.begin(), data_span.end(), std::back_inserter(output_values),
                 [](const Ort::Float16_t& fp16) { return static_cast<float>(fp16); });

  for (size_t i = 0; i < values_length; ++i) {
    ASSERT_NEAR(values[i], output_values[i], 1e-3);
  }
}

TEST(CApiTest, create_tensor_with_data_bfloat16) {
  // Example with C++. However, what we are feeding underneath is really
  // a continuous buffer of uint16_t
  // Conversion from float to bfloat16 is simple. Strip off half of the bytes from float.
  constexpr float values[] = {1.f, 2.f, 3.f, 4.f, 5.f};
  constexpr size_t values_length = std::size(values);
  constexpr uint16_t expected_values[] = {16256, 16384, 16448, 16512, 16544};  // 1.f, 2.f, 3.f, 4.f, 5.f

  constexpr int64_t dims = static_cast<int64_t>(values_length);

  std::vector<Ort::BFloat16_t> b16_values;
  b16_values.reserve(values_length);
  std::transform(std::begin(values), std::end(values), std::back_inserter(b16_values),
                 [](float fl) { return Ort::BFloat16_t(fl); });

  for (size_t i = 0; i < values_length; ++i) {
    ASSERT_EQ(expected_values[i], b16_values[i].val);
  }

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor<Ort::BFloat16_t>(info, b16_values.data(), values_length, &dims, 1u);
  const auto* new_pointer = tensor.GetTensorData<Ort::BFloat16_t>();
  ASSERT_EQ(new_pointer, b16_values.data());
  auto type_info = tensor.GetTypeInfo();
  const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  const auto element_count = tensor_info.GetElementCount();
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1u, tensor_info.GetDimensionsCount());
  ASSERT_EQ(tensor_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);

  const Ort::BFloat16_t& value_at_1 = tensor.At<Ort::BFloat16_t>({1});
  ASSERT_EQ(expected_values[1], value_at_1.val);

  std::vector<float> output_values;
  output_values.reserve(values_length);
  const auto data_span = gsl::make_span(tensor.GetTensorData<Ort::BFloat16_t>(), element_count);
  std::transform(data_span.begin(), data_span.end(), std::back_inserter(output_values),
                 [](const Ort::BFloat16_t& b16) { return static_cast<float>(b16); });

  for (size_t i = 0; i < values_length; ++i) {
    ASSERT_NEAR(values[i], output_values[i], 1e-3);
  }
}

#if !defined(DISABLE_FLOAT8_TYPES)

TEST(CApiTest, create_tensor_with_data_float8) {
  Ort::Float8E4M3FN_t values[] = {0, 1, 2, 3, 4};
  constexpr size_t values_length = sizeof(values) / sizeof(values[0]);
  std::vector<int64_t> dims = {static_cast<int64_t>(values_length)};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor<Ort::Float8E4M3FN_t>(info, values, values_length, dims.data(), dims.size());
  const auto* new_pointer = tensor.GetTensorData<Ort::Float8E4M3FN_t>();
  ASSERT_EQ(new_pointer, values);
  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1u, tensor_info.GetDimensionsCount());
  ASSERT_EQ(tensor_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN);

  Ort::Float8E4M3FN_t value_at_1 = tensor.At<Ort::Float8E4M3FN_t>({1});
  ASSERT_EQ(values[1], value_at_1);
}

#endif

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

  {
    auto f1_init_name = session.GetOverridableInitializerNameAllocated(0, allocator.get());
    ASSERT_TRUE(strcmp("F1", f1_init_name.get()) == 0);
  }

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
  {
    auto profile_file = session_1.EndProfilingAllocated(allocator.get());
    ASSERT_TRUE(std::string(profile_file.get()).find("profile_prefix") != std::string::npos);
  }
  // Create session with profiling disabled
  Ort::SessionOptions session_options_2;
#ifdef _WIN32
  session_options_2.DisableProfiling();
#else
  session_options_2.DisableProfiling();
#endif
  Ort::Session session_2(*ort_env, MODEL_WITH_CUSTOM_MODEL_METADATA, session_options_2);
  {
    auto profile_file = session_2.EndProfilingAllocated(allocator.get());
    ASSERT_TRUE(std::string(profile_file.get()) == std::string());
  }
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

    {
      auto producer_name = model_metadata.GetProducerNameAllocated(allocator.get());
      ASSERT_TRUE(strcmp("Hari", producer_name.get()) == 0);
    }

    {
      auto graph_name = model_metadata.GetGraphNameAllocated(allocator.get());
      ASSERT_TRUE(strcmp("matmul test", graph_name.get()) == 0);
    }

    {
      auto domain = model_metadata.GetDomainAllocated(allocator.get());
      ASSERT_TRUE(strcmp("", domain.get()) == 0);
    }

    {
      auto description = model_metadata.GetDescriptionAllocated(allocator.get());
      ASSERT_TRUE(strcmp("This is a test model with a valid ORT config Json", description.get()) == 0);
    }

    {
      auto graph_description = model_metadata.GetGraphDescriptionAllocated(allocator.get());
      ASSERT_TRUE(strcmp("graph description", graph_description.get()) == 0);
    }

    int64_t version = model_metadata.GetVersion();
    ASSERT_TRUE(version == 1);

    {
      auto custom_metadata_map_keys = model_metadata.GetCustomMetadataMapKeysAllocated(allocator.get());
      ASSERT_EQ(custom_metadata_map_keys.size(), 2U);
    }

    auto lookup_value_1 = model_metadata.LookupCustomMetadataMapAllocated("ort_config", allocator.get());
    ASSERT_TRUE(strcmp(lookup_value_1.get(),
                       "{\"session_options\": {\"inter_op_num_threads\": 5, \"intra_op_num_threads\": 2, "
                       "\"graph_optimization_level\": 99, \"enable_profiling\": 1}}") == 0);

    auto lookup_value_2 = model_metadata.LookupCustomMetadataMapAllocated("dummy_key", allocator.get());
    ASSERT_TRUE(strcmp(lookup_value_2.get(), "dummy_value") == 0);

    // key doesn't exist in custom metadata map
    auto lookup_value_3 = model_metadata.LookupCustomMetadataMapAllocated("key_doesnt_exist", allocator.get());
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
    {
      auto description = model_metadata.GetDescriptionAllocated(allocator.get());
      ASSERT_TRUE(strcmp("", description.get()) == 0);
    }

    // Graph description is empty
    {
      auto graph_description = model_metadata.GetGraphDescriptionAllocated(allocator.get());
      ASSERT_TRUE(strcmp("", graph_description.get()) == 0);
    }

    // Model does not contain custom metadata map
    auto custom_metadata_map_keys = model_metadata.GetCustomMetadataMapKeysAllocated(allocator.get());
    ASSERT_TRUE(custom_metadata_map_keys.empty());
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

TEST(CApiTest, get_version_string_cpp) {
  auto version_string = Ort::GetVersionString();
  ASSERT_FALSE(version_string.empty());
  ASSERT_EQ(version_string, std::string(ORT_VERSION));
}

TEST(CApiTest, get_build_info_string) {
  auto build_info_string = Ort::GetBuildInfoString();
  ASSERT_FALSE(build_info_string.empty());
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
#ifdef USE_CUDA
  {
    OrtMemoryInfo* cuda_meminfo = nullptr;
    ASSERT_TRUE(api.CreateMemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault, &cuda_meminfo) == nullptr);
    std::unique_ptr<OrtMemoryInfo, decltype(api.ReleaseMemoryInfo)> rel_info(cuda_meminfo, api.ReleaseMemoryInfo);

    OrtArenaCfg* arena_cfg = nullptr;
    ASSERT_TRUE(api.CreateArenaCfg(0, -1, -1, -1, &arena_cfg) == nullptr);
    std::unique_ptr<OrtArenaCfg, decltype(api.ReleaseArenaCfg)> rel_arena_cfg(arena_cfg, api.ReleaseArenaCfg);

    std::vector<const char*> keys, values;
    ASSERT_TRUE(api.CreateAndRegisterAllocatorV2(env_ptr, onnxruntime::kCudaExecutionProvider, cuda_meminfo, arena_cfg, keys.data(), values.data(), 0) == nullptr);

    // Test that duplicates are handled
    std::unique_ptr<OrtStatus, decltype(api.ReleaseStatus)> status_releaser(
        api.CreateAndRegisterAllocatorV2(env_ptr, onnxruntime::kCudaExecutionProvider, cuda_meminfo, arena_cfg, keys.data(), values.data(), 0),
        api.ReleaseStatus);
    ASSERT_FALSE(status_releaser.get() == nullptr);

    {
      // create session 1
      Ort::SessionOptions cuda_session_options;
      cuda_session_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");
      cuda_session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
      Ort::Session session1(*ort_env, MODEL_URI, cuda_session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session1,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);

      // create session 2
      Ort::Session session2(*ort_env, MODEL_URI, cuda_session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session2,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);
    }

    ASSERT_TRUE(api.UnregisterAllocator(env_ptr, cuda_meminfo) == nullptr);
  }
#endif
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

  // create session 1 (using model path)
  Ort::Session session1(*ort_env, MATMUL_MODEL_URI, session_options, prepacked_weights_container);
  RunSession<float>(default_allocator.get(),
                    session1,
                    inputs,
                    "Y",
                    expected_dims_y,
                    expected_values_y,
                    nullptr);

  // create session 2 (using model bytes)
  std::ifstream model_file_stream(MATMUL_MODEL_URI, std::ios::in | std::ios::binary);
  ASSERT_TRUE(model_file_stream.good());

  model_file_stream.seekg(0, std::ios::end);
  const auto size = onnxruntime::narrow<size_t>(model_file_stream.tellg());
  model_file_stream.seekg(0, std::ios::beg);
  std::vector<char> file_contents(size, 0);
  model_file_stream.read(&file_contents[0], size);
  model_file_stream.close();

  Ort::Session session2(*ort_env, file_contents.data(), size, session_options, prepacked_weights_container);
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

  const char* keys[] = {"max_mem", "arena_extend_strategy", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk", "initial_growth_chunk_size_bytes", "max_power_of_two_extend_bytes"};
  const size_t values[] = {0 /*let ort pick default max memory*/, 0, 1024, 0, 256, 1L << 24};

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
TEST(CApiTest, TestExternalCUDAStreamWithIOBinding) {
  const auto& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  OrtTensorRTProviderOptionsV2* trt_options;
  ASSERT_TRUE(api.CreateTensorRTProviderOptions(&trt_options) == nullptr);
  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)>
      rel_trt_options(trt_options, api.ReleaseTensorRTProviderOptions);

  // updating provider option with user provided compute stream
  cudaStream_t compute_stream = nullptr;
  void* user_compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  ASSERT_TRUE(api.UpdateTensorRTProviderOptionsWithValue(rel_trt_options.get(), "user_compute_stream", compute_stream) == nullptr);
  ASSERT_TRUE(api.GetTensorRTProviderOptionsByName(rel_trt_options.get(), "user_compute_stream", &user_compute_stream) == nullptr);
  ASSERT_TRUE(user_compute_stream == (void*)compute_stream);

  ASSERT_TRUE(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
                  static_cast<OrtSessionOptions*>(session_options),
                  rel_trt_options.get()) == nullptr);

  Ort::Session session(*ort_env, MODEL_URI, session_options);
  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);

  Ort::Allocator cuda_allocator(session, info_cuda);
  auto allocator_info = cuda_allocator.GetInfo();
  ASSERT_TRUE(info_cuda == allocator_info);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = cuda_allocator.GetAllocation(x_values.size() * sizeof(float));

  ASSERT_NE(input_data.get(), nullptr);
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  auto output_data = cuda_allocator.GetAllocation(expected_y.size() * sizeof(float));

  ASSERT_NE(output_data.get(), nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data.get()),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  // Create IoBinding for inputs and outputs.
  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  session.Run(Ort::RunOptions(), binding);

  // Check the values against the bound raw memory (needs copying from device to host first)
  std::array<float, 3 * 2> y_values;
  cudaMemcpy(y_values.data(), output_data.get(), sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);

  std::cout << "output: " << std::endl;
  for (auto y : y_values) {
    std::cout << y << std::endl;
  }
  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}

TEST(CApiTest, TestExternalCUDAStreamWithIOBinding2) {
  const auto& api = Ort::GetApi();
  Ort::SessionOptions session_options;

  OrtTensorRTProviderOptionsV2* trt_options;
  ASSERT_TRUE(api.CreateTensorRTProviderOptions(&trt_options) == nullptr);
  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)>
      rel_trt_options(trt_options, api.ReleaseTensorRTProviderOptions);

  // updating provider option with user provided compute stream
  cudaStream_t compute_stream = nullptr;
  void* user_compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  ASSERT_TRUE(api.UpdateTensorRTProviderOptionsWithValue(rel_trt_options.get(), "user_compute_stream", compute_stream) == nullptr);
  ASSERT_TRUE(api.GetTensorRTProviderOptionsByName(rel_trt_options.get(), "user_compute_stream", &user_compute_stream) == nullptr);
  ASSERT_TRUE(user_compute_stream == (void*)compute_stream);

  ASSERT_TRUE(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
                  static_cast<OrtSessionOptions*>(session_options),
                  rel_trt_options.get()) == nullptr);

  Ort::Session session(*ort_env, MODEL_URI, session_options);
  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  /*
   * Use cudaMallocHost() (pinned memory allocation) to create input/output tensors
   */
  float* input_data;
  cudaMallocHost(&input_data, 3 * 2 * sizeof(float));
  ASSERT_NE(input_data, nullptr);
  cudaMemcpy(input_data, x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

  std::cout << "pinned memory allocation" << std::endl;
  std::cout << "input tesnor:" << std::endl;
  for (int i = 0; i < 6; i++) {
    std::cout << input_data[i] << std::endl;
  }

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  float* output_data;
  cudaMallocHost(&output_data, 3 * 2 * sizeof(float));
  ASSERT_NE(output_data, nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  // Create IoBinding for inputs and outputs.
  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  /*
   * Use cudaMalloc() (pageable memory allocation first and then implicit pinned memory allocation) to create input/output tensors
   */
  float* input_data_2;
  cudaMalloc(&input_data_2, 3 * 2 * sizeof(float));
  ASSERT_NE(input_data_2, nullptr);
  cudaMemcpy(input_data_2, x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x_2 = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data_2), x_values.size(),
                                                x_shape.data(), x_shape.size());

  float* output_data_2;
  cudaMallocHost(&output_data_2, 3 * 2 * sizeof(float));
  ASSERT_NE(output_data_2, nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y_2 = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data_2),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  // Create IoBinding for inputs and outputs.
  Ort::IoBinding binding_2(session);
  binding_2.BindInput("X", bound_x_2);
  binding_2.BindOutput("Y", bound_y_2);

  // Run with first iobindings
  session.Run(Ort::RunOptions(), binding);

  // Check the values against the bound raw memory (needs copying from device to host first)
  std::array<float, 3 * 2> y_values;
  cudaMemcpy(y_values.data(), output_data, sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);

  std::cout << "pinned memory allocation" << std::endl;
  std::cout << "output: " << std::endl;
  for (auto y : y_values) {
    std::cout << y << std::endl;
  }
  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Run with second iobindings
  session.Run(Ort::RunOptions(), binding_2);

  // Check the values against the bound raw memory (needs copying from device to host first)
  cudaMemcpy(y_values.data(), output_data_2, sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);

  std::cout << "pageable memory allocation" << std::endl;
  std::cout << "output: " << std::endl;
  for (auto y : y_values) {
    std::cout << y << std::endl;
  }
  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
  binding_2.ClearBoundInputs();
  binding_2.ClearBoundOutputs();

  cudaFree(input_data);
  cudaFree(output_data);
  cudaFree(input_data_2);
  cudaFree(output_data_2);
  cudaStreamDestroy(compute_stream);
}

class CApiTensorRTTest : public testing::Test, public ::testing::WithParamInterface<std::string> {};

// This test uses CreateTensorRTProviderOptions/UpdateTensorRTProviderOptions APIs to configure and create a TensorRT Execution Provider
TEST_P(CApiTensorRTTest, TestConfigureTensorRTProviderOptions) {
  std::string param = GetParam();
  size_t pos = param.find("=");
  std::string option_name = param.substr(0, pos);
  std::string option_value = param.substr(pos + 1);
  ASSERT_NE(pos, std::string::npos);

  const auto& api = Ort::GetApi();
  OrtTensorRTProviderOptionsV2* trt_options;
  OrtAllocator* allocator;
  char* trt_options_str;
  ASSERT_TRUE(api.CreateTensorRTProviderOptions(&trt_options) == nullptr);
  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(trt_options, api.ReleaseTensorRTProviderOptions);

  const char* engine_cache_path = "./trt_engine_folder";

  std::vector<const char*> keys{"device_id", "has_user_compute_stream", "trt_fp16_enable", "trt_int8_enable", "trt_engine_cache_enable",
                                "trt_engine_cache_path", option_name.c_str()};

  std::vector<const char*> values{"0", "0", "1", "0", "1",
                                  engine_cache_path, option_value.c_str()};

  ASSERT_TRUE(api.UpdateTensorRTProviderOptions(rel_trt_options.get(), keys.data(), values.data(), keys.size()) == nullptr);

  ASSERT_TRUE(api.GetAllocatorWithDefaultOptions(&allocator) == nullptr);
  ASSERT_TRUE(api.GetTensorRTProviderOptionsAsString(rel_trt_options.get(), allocator, &trt_options_str) == nullptr);
  std::string s(trt_options_str);
  ASSERT_TRUE(s.find(engine_cache_path) != std::string::npos);
  ASSERT_TRUE(s.find(param.c_str()) != std::string::npos);
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

  // without preallocated output tensor
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

/*
 * The TensorrtExecutionProviderOptionsTest can be used to test TRT options
 */
INSTANTIATE_TEST_SUITE_P(CApiTensorRTTest, CApiTensorRTTest,
                         ::testing::Values("trt_build_heuristics_enable=1", "trt_sparsity_enable=1", "trt_builder_optimization_level=0", "trt_tactic_sources=-CUDNN,+CUBLAS", "trt_auxiliary_streams=2"));
#endif

#ifdef USE_CUDA

// This test uses CreateCUDAProviderOptions/UpdateCUDAProviderOptions/UpdateCUDAProviderOptionsWithValue APIs to configure and create a CUDA Execution Provider instance
TEST(CApiTest, TestConfigureCUDAProviderOptions) {
  const auto& api = Ort::GetApi();

  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  ASSERT_TRUE(api.CreateCUDAProviderOptions(&cuda_options) == nullptr);
  std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);

  // Only test updating OrtCUDAProviderOptionsV2 instance with user provided compute stream not running the inference
  cudaStream_t compute_stream = nullptr;
  void* user_compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  ASSERT_TRUE(api.UpdateCUDAProviderOptionsWithValue(rel_cuda_options.get(), "user_compute_stream", compute_stream) == nullptr);
  ASSERT_TRUE(api.GetCUDAProviderOptionsByName(rel_cuda_options.get(), "user_compute_stream", &user_compute_stream) == nullptr);
  ASSERT_TRUE(user_compute_stream == (void*)compute_stream);
  cudaStreamDestroy(compute_stream);

  std::vector<const char*> keys{
      "device_id", "has_user_compute_stream", "gpu_mem_limit", "arena_extend_strategy",
      "cudnn_conv_algo_search", "do_copy_in_default_stream", "cudnn_conv_use_max_workspace", "cudnn_conv1d_pad_to_nc1d"};

  std::vector<const char*> values{
      "0", "0", "1024", "kSameAsRequested",
      "DEFAULT", "1", "1"};

  ASSERT_TRUE(api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 6) == nullptr);

  OrtAllocator* allocator;
  ASSERT_TRUE(api.GetAllocatorWithDefaultOptions(&allocator) == nullptr);

  char* cuda_options_str = nullptr;
  ASSERT_TRUE(api.GetCUDAProviderOptionsAsString(rel_cuda_options.get(), allocator, &cuda_options_str) == nullptr);
  std::string s;
  if (cuda_options_str != nullptr) {
    s = std::string(cuda_options_str, strnlen(cuda_options_str, 2048));
  }
  ASSERT_TRUE(s.find("device_id=0") != std::string::npos);
  ASSERT_TRUE(s.find("gpu_mem_limit=1024") != std::string::npos);
  ASSERT_TRUE(s.find("arena_extend_strategy=kSameAsRequested") != std::string::npos);
  ASSERT_TRUE(s.find("cudnn_conv_algo_search=DEFAULT") != std::string::npos);
  ASSERT_TRUE(s.find("do_copy_in_default_stream=1") != std::string::npos);
  ASSERT_TRUE(s.find("cudnn_conv_use_max_workspace=1") != std::string::npos);
  ASSERT_TRUE(s.find("cudnn_conv1d_pad_to_nc1d") != std::string::npos);

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

// Preventing resize transformer issue:
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

// Reduced Ops build doesn't support OptionalHasElement (16) yet
#if !defined(REDUCED_OPS_BUILD) && !defined(DISABLE_OPTIONAL_TYPE)
TEST(CApiTest, GH_11717) {
  const auto* model_path = TSTR("testdata/gh_issue_11717.onnx");

  Ort::SessionOptions session_options{};
  // Just check if the model loads fine without a segmentation fault
  // in the default CPU EP
  EXPECT_NO_THROW(Ort::Session session(*ort_env, model_path, session_options));
}
#endif

#ifndef REDUCED_OPS_BUILD
TEST(CApiTest, TestMultiStreamInferenceSimpleSSD) {
  Ort::SessionOptions session_options{};
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  session_options.AddConfigEntry("session.node_partition_config_file",
                                 "./testdata/multi_stream_models/simplified_ssd_cpu.csv");
  Ort::Session session{*ort_env, SIMPLIFIED_SSD_MODEL_URI, session_options};
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  const char* input_names[] = {"graph_in"};
  std::unique_ptr<float[]> input_data = std::make_unique<float[]>(3 * 3 * 300 * 300);
  for (int i = 0; i < 3 * 3 * 300 * 300; ++i) {
    input_data[i] = 1.f;
  }
  int64_t input_dims[] = {3, 3, 300, 300};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(info, input_data.get(), 3 * 3 * 300 * 300, input_dims, 4U));
  const char* output_names[] = {"graph_out"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                                                    ort_inputs.data(), ort_inputs.size(),
                                                    output_names, countof(output_names));
  ASSERT_TRUE(ort_outputs.size() == 1);
  ASSERT_TRUE(ort_outputs[0].IsTensor());
  const auto& type_shape_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  std::vector<int64_t> output_dims = type_shape_info.GetShape();
  std::vector<int64_t> expected_output_dims = {3, 256, 150, 150};
  ASSERT_TRUE(output_dims == expected_output_dims);
}
#endif

#if !defined(REDUCED_OPS_BUILD) && !defined(DISABLE_OPTIONAL_TYPE)

TEST(LiteCustomOpTest, CustomFunc) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
#if defined(_WIN32)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("custom_op_library.dll"));
#elif defined(__APPLE__)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.dylib"));
#else
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("./libcustom_op_library.so"));
#endif

  Ort::Session session{*ort_env, TSTR("testdata/fuse_select_filter.onnx"), session_options};

  const char* input_names[] = {"vector_1", "vector_2", "alpha", "indices"};
  const char* output_names[] = {"vector_filtered"};

  float vector_1_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
  int64_t vector_1_dim[] = {10};

  float vector_2_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  int64_t vector_2_dim[] = {6};

  int32_t alpha_value[] = {2};
  int64_t alpha_dim[] = {1};

  int32_t indices_value[] = {0, 1, 2, 3, 4, 5};
  int64_t indices_dim[] = {6};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[] = {
      Ort::Value::CreateTensor<float>(memory_info, vector_1_value, 10, vector_1_dim, 1),
      Ort::Value::CreateTensor<float>(memory_info, vector_2_value, 6, vector_2_dim, 1),
      Ort::Value::CreateTensor<int32_t>(memory_info, alpha_value, 1, alpha_dim, 1),
      Ort::Value::CreateTensor<int32_t>(memory_info, indices_value, 6, indices_dim, 1)};

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 4, output_names, 1);
  const auto& vector_filterred = output_tensors.at(0);
  auto type_shape_info = vector_filterred.GetTensorTypeAndShapeInfo();
  const float* floats_output = static_cast<const float*>(vector_filterred.GetTensorRawData());
  ASSERT_TRUE(floats_output[0] == 8);
  ASSERT_TRUE(floats_output[1] == 16);
}

struct Merge {
  Merge(const OrtApi* ort_api, const OrtKernelInfo* info) {
    int64_t reverse;
    ORT_ENFORCE(ort_api->KernelInfoGetAttribute_int64(info, "reverse", &reverse) == nullptr);
    reverse_ = reverse != 0;
  }
  Ort::Status Compute(const Ort::Custom::Tensor<std::string_view>& strings_in,
                      std::string_view string_in,
                      Ort::Custom::Tensor<std::string>* strings_out) {
    if (strings_in.NumberOfElement() == 0) {
      return Ort::Status("the 1st input must have more than one string!", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    std::vector<std::string> string_pool;
    for (const auto& s : strings_in.Data()) {
      string_pool.emplace_back(s.data(), s.size());
    }
    string_pool.emplace_back(string_in.data(), string_in.size());
    if (reverse_) {
      for (auto& str : string_pool) {
        std::reverse(str.begin(), str.end());
      }
      std::reverse(string_pool.begin(), string_pool.end());
    }
    strings_out->SetStringOutput(string_pool, {static_cast<int64_t>(string_pool.size())});
    return Ort::Status(nullptr);
  }
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    auto input_count = ctx.GetInputCount();
    if (input_count != 2) {
      return Ort::Status("input count should be 2", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    Ort::ShapeInferContext::Shape shape_1 = {{-1}};
    ctx.SetOutputShape(0, shape_1);
    return Ort::Status(nullptr);
  }
  bool reverse_ = false;
};

TEST(LiteCustomOpTest, CustomStruct) {
  const auto& ortApi = Ort::GetApi();

  Ort::CustomOpDomain v2_domain{"v2"};
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> mrg_op_ptr{Ort::Custom::CreateLiteCustomOp<Merge>("Merge", "CPUExecutionProvider")};
  v2_domain.Add(mrg_op_ptr.get());

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.Add(v2_domain);
  session_options.SetLogSeverityLevel(0);

  Ort::Session session{*ort_env, TSTR("testdata/merge.onnx"), session_options};

  const char* input_names[] = {"str_in_1", "str_in_2"};
  const char* output_names[] = {"str_out"};

  OrtAllocator* allocator = nullptr;
  ASSERT_TRUE(!ortApi.GetAllocatorWithDefaultOptions(&allocator));
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  int64_t str_1_dims[] = {2};
  int64_t str_2_dims[] = {1};

  Ort::Value input_tensors[] = {Ort::Value::CreateTensor(allocator, str_1_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
                                Ort::Value::CreateTensor(allocator, str_2_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)};

  const char* str_1_raw[] = {"abc", "de"};
  const char* str_2_raw[] = {"fg"};

  input_tensors[0].FillStringTensor(str_1_raw, 2);
  input_tensors[1].FillStringTensor(str_2_raw, 1);

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 2, output_names, 1);
  const auto& str_out_tensor = output_tensors.at(0);
  auto num_chars = str_out_tensor.GetStringTensorDataLength();
  std::vector<char> chars(num_chars + 1, '\0');
  std::vector<size_t> offsets(3);
  str_out_tensor.GetStringTensorContent(static_cast<void*>(chars.data()), num_chars, offsets.data(), offsets.size());
  ASSERT_TRUE(strncmp(chars.data(), "gfedcba", 7) == 0);
}

TEST(LiteCustomOpTest, MissingOptional) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
#if defined(_WIN32)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("custom_op_library.dll"));
#elif defined(__APPLE__)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.dylib"));
#else
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("./libcustom_op_library.so"));
#endif

  Ort::Session session(*ort_env, TSTR("testdata/optional_2.onnx"), session_options);

  const char* input_names[] = {"float_in_1", "float_in_2"};
  const char* output_names[] = {"float_out_1"};

  float vector_1_value[] = {0.f, 1.f, 2.f};
  int64_t vector_1_dim[] = {3};

  float vector_2_value[] = {4.f, 5.f, 6.f, 7.f};
  int64_t vector_2_dim[] = {4};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[] = {
      Ort::Value::CreateTensor<float>(memory_info, vector_1_value, gsl::narrow_cast<size_t>(vector_1_dim[0]),
                                      vector_1_dim, 1),
      Ort::Value::CreateTensor<float>(memory_info, vector_2_value, gsl::narrow_cast<size_t>(vector_2_dim[0]),
                                      vector_2_dim, 1)};

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 2, output_names, 1);
  ASSERT_TRUE(output_tensors.size() == 1);
}

TEST(LiteCustomOpTest, HasOptional) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
#if defined(_WIN32)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("custom_op_library.dll"));
#elif defined(__APPLE__)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.dylib"));
#else
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("./libcustom_op_library.so"));
#endif

  Ort::Session session(*ort_env, TSTR("testdata/optional_3.onnx"), session_options);

  const char* input_names[] = {"float_in_1", "float_in_2", "float_in_3"};
  const char* output_names[] = {"float_out_1", "float_out_2"};

  float vector_1_value[] = {0.f, 1.f, 2.f};
  int64_t vector_1_dim[] = {3};

  float vector_2_value[] = {4.f, 5.f, 6.f, 7.f};
  int64_t vector_2_dim[] = {4};

  float vector_3_value[] = {8.f, 9.f};
  int64_t vector_3_dim[] = {2};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[] = {
      Ort::Value::CreateTensor<float>(memory_info, vector_1_value, gsl::narrow_cast<size_t>(vector_1_dim[0]),
                                      vector_1_dim, 1),
      Ort::Value::CreateTensor<float>(memory_info, vector_2_value, gsl::narrow_cast<size_t>(vector_2_dim[0]),
                                      vector_2_dim, 1),
      Ort::Value::CreateTensor<float>(memory_info, vector_3_value, gsl::narrow_cast<size_t>(vector_3_dim[0]),
                                      vector_3_dim, 1),
  };

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 3, output_names, 2);
  ASSERT_TRUE(output_tensors.size() == 2);
}

TEST(MultiKernelSingleSchemaTest, valid) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
#if defined(_WIN32)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("custom_op_library.dll"));
#elif defined(__APPLE__)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.dylib"));
#else
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("./libcustom_op_library.so"));
#endif

  Ort::Session session(*ort_env, CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL, session_options);

  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y", "Z"};
  float x_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
  int64_t x_dim[] = {10};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[1] = {
      Ort::Value::CreateTensor<float>(memory_info, x_value, 10, x_dim, 1),
  };

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 1, output_names, 2);
  ASSERT_TRUE(*output_tensors[1].GetTensorData<int32_t>() == 72);
}

// expect input count mismatch exception
TEST(MultiKernelSingleSchemaTest, InputCountMismatch) {
  Ort::CustomOpDomain v2_domain("v2");
  MulTopOpFloat mul_top_f32;
  MulTopOpDouble mul_top_double;

  v2_domain.Add(&mul_top_f32);
  v2_domain.Add(&mul_top_double);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
  session_options.Add(v2_domain);

  EXPECT_THROW(Ort::Session session(*ort_env, CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL, session_options), std::exception);
}

// expect output count mismatch exception
TEST(MultiKernelSingleSchemaTest, OutputMismatch) {
  Ort::CustomOpDomain v2_domain("v2");
  MulTopOpFloat mul_top_f32;
  MulTopOpInt16 mul_top_int64;

  v2_domain.Add(&mul_top_f32);
  v2_domain.Add(&mul_top_int64);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
  session_options.Add(v2_domain);

  EXPECT_THROW(Ort::Session session(*ort_env, CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL, session_options), std::exception);
}

// expect characteristic mismatch exception
TEST(MultiKernelSingleSchemaTest, CharacterMismatch) {
  Ort::CustomOpDomain v2_domain("v2");
  MulTopOpFloat mul_top_f32;
  MulTopOpFloat16 mul_top_f16;

  v2_domain.Add(&mul_top_f32);
  v2_domain.Add(&mul_top_f16);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
  session_options.Add(v2_domain);

  EXPECT_THROW(Ort::Session session(*ort_env, CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL, session_options), std::exception);
}

TEST(MultiKernelSingleSchemaTest, DuplicateKernel) {
  Ort::CustomOpDomain v2_domain("v2");
  MulTopOpFloat mul_top_f32_1;
  MulTopOpFloat mul_top_f32_2;
  MulTopOpInt32 mul_top_i32;

  v2_domain.Add(&mul_top_f32_1);
  v2_domain.Add(&mul_top_f32_2);
  v2_domain.Add(&mul_top_i32);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
  session_options.Add(v2_domain);

  EXPECT_NO_THROW(Ort::Session session(*ort_env, CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL, session_options));
}

#endif

static std::thread::id caller_tid = std::this_thread::get_id();
static std::atomic_bool atomic_wait{false};

void CallbackSucceed(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status_ptr) {
  auto callee_tid = std::this_thread::get_id();
  EXPECT_NE(*(reinterpret_cast<std::thread::id*>(user_data)), callee_tid);
  Ort::Status status(status_ptr);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(num_outputs, 1UL);
  Ort::Value output_value(outputs[0]);
  EXPECT_EQ(output_value.At<float>({1, 0}), 9.f);
  output_value.release();
  atomic_wait.store(true);
}

TEST(CApiTest, RunAsync) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(2);
  Ort::Session session(*ort_env, MODEL_URI, session_options);

  const char* input_names[] = {"X"};
  float x_value[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t x_dim[] = {3, 2};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[1] = {
      Ort::Value::CreateTensor<float>(memory_info, x_value, 6, x_dim, 2),
  };

  const char* output_names[] = {"Y"};
  Ort::RunOptions run_options;
  Ort::Value output_values[1] = {Ort::Value{nullptr}};

  EXPECT_NO_THROW(session.RunAsync(run_options,
                                   input_names,
                                   input_tensors,
                                   1,
                                   output_names,
                                   output_values,
                                   1,
                                   CallbackSucceed,
                                   &caller_tid));

  std::chrono::duration<double, std::milli> dur{100};
  // timeout in about 10 secs
  for (int i = 0; i < 100 && !atomic_wait.load(); ++i) {
    std::this_thread::sleep_for(dur);
  }

  EXPECT_EQ(atomic_wait.load(), true);
}

void CallbackFail(void*, OrtValue**, size_t, OrtStatusPtr) {
  EXPECT_TRUE(false);  // the callback is not supposed to be invoked
}

TEST(CApiTest, RunAsyncFail) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);  // This will cause RunAsync fail
  Ort::Session session(*ort_env, MODEL_URI, session_options);

  const char* input_names[] = {"X"};
  float x_value[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t x_dim[] = {3, 2};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[1] = {
      Ort::Value::CreateTensor<float>(memory_info, x_value, 6, x_dim, 2),
  };
  Ort::Value output_values[1] = {Ort::Value{nullptr}};
  const char* output_names[] = {"Y"};

  Ort::RunOptions run_options;
  EXPECT_THROW(session.RunAsync(run_options, input_names, input_tensors, 1, output_names, output_values, 1, CallbackFail, nullptr), std::exception);
}
