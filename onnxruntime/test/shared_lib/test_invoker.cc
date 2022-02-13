// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include <gtest/gtest.h>
#include "core/providers/cpu/cpu_provider_factory.h"

OrtValue* CreateOrtValue(OrtMemoryInfo* mem_info, std::vector<int64_t>& dims, std::vector<float>& values) {
  OrtValue* value;
  Ort::ThrowOnError(Ort::GetApi().CreateTensorWithDataAsOrtValue(mem_info,
                                                                 values.data(),
                                                                 values.size() * sizeof(float),
                                                                 dims.data(),
                                                                 dims.size(),
                                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                                 &value));
  return value;
}

TEST(CApiTest, invoker) {
  OrtEnv* env;
  OrtSessionOptions* options;
  OrtInvoker* invoker;
  OrtMemoryInfo* mem_info;
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
  Ort::ThrowOnError(Ort::GetApi().CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  Ort::ThrowOnError(Ort::GetApi().CreateSessionOptions(&options));

  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(options, /*use_arena=*/0));
  Ort::ThrowOnError(Ort::GetApi().CreateInvoker(env, options, 0, &invoker));

  std::vector<int64_t> input_node1_dims{2, 2};
  std::vector<int64_t> input_node2_dims{2, 1};
  std::vector<int64_t> output_node_dims{2, 2};

  std::vector<float> input_node1_data{1, 2, 3, 4};
  std::vector<float> input_node2_data{5, 6};
  std::vector<float> output_node_data{0, 0, 0, 0};

  OrtValue* input1 = CreateOrtValue(mem_info, input_node1_dims, input_node1_data);
  OrtValue* input2 = CreateOrtValue(mem_info, input_node2_dims, input_node2_data);
  std::array<OrtValue*, 2> inputs{input1, input2};
  OrtValue* output = CreateOrtValue(mem_info, output_node_dims, output_node_data);

  Ort::ThrowOnError(Ort::GetApi().Invoker_Invoke(invoker, "Add", inputs.data(), 2, &output, 1, nullptr, "", 13));

  EXPECT_EQ(output_node_data[0], 6);
  EXPECT_EQ(output_node_data[1], 7);
  EXPECT_EQ(output_node_data[2], 9);
  EXPECT_EQ(output_node_data[3], 10);

  Ort::GetApi().ReleaseValue(input1);
  Ort::GetApi().ReleaseValue(input2);
  Ort::GetApi().ReleaseValue(output);
  Ort::GetApi().ReleaseInvoker(invoker);
  Ort::GetApi().ReleaseSessionOptions(options);
  Ort::GetApi().ReleaseEnv(env);
  Ort::GetApi().ReleaseMemoryInfo(mem_info);
}

TEST(CApiTest, invoker_attribute) {
  OrtEnv* env;
  OrtSessionOptions* options;
  OrtInvoker* invoker;
  OrtMemoryInfo* mem_info;
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
  Ort::ThrowOnError(Ort::GetApi().CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  Ort::ThrowOnError(Ort::GetApi().CreateSessionOptions(&options));

  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(options, /*use_arena=*/0));
  Ort::ThrowOnError(Ort::GetApi().CreateInvoker(env, options, 0, &invoker));

  std::vector<int64_t> input_node_dims = {2, 2, 2};
  std::vector<int64_t> output_node_dims = {2};

  std::vector<float> input_node_data = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> output_node_data = {0, 0};

  OrtValue* input = CreateOrtValue(mem_info, input_node_dims, input_node_data);
  OrtValue* output = CreateOrtValue(mem_info, output_node_dims, output_node_data);

  OrtNodeAttributes* attributes;
  Ort::ThrowOnError(Ort::GetApi().CreateNodeAttributes(&attributes));
  std::vector<int64_t> axes{0, 2};
  Ort::ThrowOnError(Ort::GetApi().NodeAttributes_SetArray_int64(attributes, "axes", axes.data(), axes.size()));
  Ort::ThrowOnError(Ort::GetApi().NodeAttributes_Set_int64(attributes, "keepdims", 0));
  Ort::ThrowOnError(Ort::GetApi().Invoker_Invoke(invoker, "ReduceMean", &input, 1, &output, 1, attributes, "", 13));

  EXPECT_EQ(output_node_data[0], 3.5);
  EXPECT_EQ(output_node_data[1], 5.5);

  Ort::GetApi().ReleaseValue(input);
  Ort::GetApi().ReleaseValue(output);
  Ort::GetApi().ReleaseNodeAttributes(attributes);
  Ort::GetApi().ReleaseInvoker(invoker);
  Ort::GetApi().ReleaseSessionOptions(options);
  Ort::GetApi().ReleaseEnv(env);
  Ort::GetApi().ReleaseMemoryInfo(mem_info);
}
