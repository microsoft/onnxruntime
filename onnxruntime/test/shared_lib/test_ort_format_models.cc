// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// if we can't load an ORT format model we can't really test anything
#if defined(ENABLE_ORT_FORMAT_LOAD)

// custom ops are only supported in a minimal build if explicitly enabled
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

#include "core/common/common.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test_allocator.h"
#include "utils.h"
#include "custom_op_utils.h"

#include "gtest/gtest.h"

extern std::unique_ptr<Ort::Env> ort_env;

static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
                          const std::vector<Input>& inputs, const char* output_name,
                          const std::vector<int64_t>& expected_dims_y, const std::vector<float>& expected_values_y,
                          Ort::CustomOpDomain& custom_op_domain, void* cuda_compute_stream = nullptr) {
  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

#ifdef USE_CUDA
  auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(cuda_compute_stream);
  session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
  ORT_UNUSED_PARAMETER(cuda_compute_stream);
#endif
  Ort::Session session(env, model_uri.c_str(), session_options);

  MockedOrtAllocator allocator;
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;

  for (size_t i = 0; i < inputs.size(); i++) {
    // we put the data in a Tensor, and Tensor has a method to get a mutable pointer to the data.
    // we never call that for an input, but need to do the const_cast to make this potential explicit
    float* input_data = const_cast<float*>(inputs[i].values.data());

    auto input_tensor = Ort::Value::CreateTensor<float>(allocator.Info(), input_data, inputs[i].values.size(),
                                                        inputs[i].dims.data(), inputs[i].dims.size());

    input_names.push_back(inputs[i].name);
    ort_inputs.push_back(std::move(input_tensor));
  }

  std::vector<Ort::Value> ort_outputs;
  ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                            &output_name, 1);

  ASSERT_EQ(ort_outputs.size(), size_t(1));
  const auto& output_tensor = &ort_outputs[0];

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), expected_dims_y);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(expected_values_y.size(), total_len);

  const auto* f = output_tensor->GetTensorMutableData<float>();
  for (size_t i = 0; i < total_len; ++i) {
    ASSERT_EQ(expected_values_y[i], f[i]);
  }
}

#if !defined(ORT_MINIMAL_BUILD)
TEST(OrtFormatCustomOpTests, ConvertOnnxModelToOrt) {
  const std::basic_string<ORTCHAR_T> onnx_file = ORT_TSTR("testdata/foo_1.onnx");
  const std::basic_string<ORTCHAR_T> ort_file = ORT_TSTR("testdata/foo_1.onnx.test_output.ort");

#ifdef USE_CUDA
  // We need to launch our custom op in the same compute stream as the one we will be
  // passing to ORT via Session options to use for the entire session (i.e.) CUDA ORT kernels
  // will now use the same stream too
  cudaStream_t compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  MyCustomOp custom_op{onnxruntime::kCudaExecutionProvider, compute_stream};
#else
  MyCustomOp custom_op{onnxruntime::kCpuExecutionProvider, nullptr};
#endif
  Ort::CustomOpDomain custom_op_domain("");
  custom_op_domain.Add(&custom_op);

  // convert to ort by loading the onnx model
  {
    Ort::SessionOptions so;
    so.Add(custom_op_domain);
    so.SetLogId("CustomOp");
    so.SetOptimizedModelFilePath(ort_file.c_str());

#ifdef USE_CUDA
    OrtCUDAProviderOptions cuda_options{};
    so.AppendExecutionProvider_CUDA(cuda_options);
#endif

    Ort::Session session(*ort_env, onnx_file.c_str(), so);
  }

  // now load the ORT format model and execute it
  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // model adds 1, 2, 3, 4, 5, 6 to the input values
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

#ifdef USE_CUDA
  TestInference(*ort_env, ort_file, inputs, "Y", expected_dims_y, expected_values_y, custom_op_domain, compute_stream);
  cudaStreamDestroy(compute_stream);
#else
  TestInference(*ort_env, ort_file, inputs, "Y", expected_dims_y, expected_values_y, custom_op_domain, nullptr);
#endif
}
#endif  // if !defined(ORT_MINIMAL_BUILD)

// the saved ORT format model has the CPU EP assigned to the custom op node, so we only test if we're not using the
// CUDA EP for the test.
#ifndef USE_CUDA
TEST(OrtFormatCustomOpTests, LoadOrtModel) {
  const std::basic_string<ORTCHAR_T> ort_file = ORT_TSTR("testdata/foo_1.onnx.ort");

  MyCustomOp custom_op{onnxruntime::kCpuExecutionProvider, nullptr};
  Ort::CustomOpDomain custom_op_domain("");
  custom_op_domain.Add(&custom_op);

  //  load the ORT format model and execute it
  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  // model adds 1, 2, 3, 4, 5, 6 to the input values
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f};

  TestInference(*ort_env, ort_file, inputs, "Y", expected_dims_y, expected_values_y, custom_op_domain);
}
#endif

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

#endif  // #if defined(ENABLE_ORT_FORMAT_LOAD)
