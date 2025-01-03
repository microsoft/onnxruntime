// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_cxx_api.h"

#include "gtest/gtest.h"

OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream = nullptr);

template <typename T = float>
struct Input {
  const char* name = nullptr;
  std::vector<int64_t> dims;
  std::vector<T> values;
};

template <typename ModelOutputT, typename ModelInputT = float, typename InputT = Input<float>>
void RunSession(OrtAllocator* allocator,
                Ort::Session& session_object,
                const std::vector<InputT>& inputs,
                const char* output_name,
                const std::vector<int64_t>& output_dims,
                const std::vector<ModelOutputT>& expected_output,
                Ort::Value* output_tensor) {
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(
        Ort::Value::CreateTensor(allocator->Info(allocator), const_cast<ModelInputT*>(inputs[i].values.data()),
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
  ASSERT_EQ(type_info.GetShape(), output_dims);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(expected_output.size(), total_len);

  auto* actual = output_tensor->GetTensorMutableData<ModelOutputT>();
  for (size_t i = 0; i != total_len; ++i) {
    if constexpr (std::is_same<ModelOutputT, float>::value || std::is_same<ModelOutputT, double>::value) {
      EXPECT_NEAR(expected_output[i], actual[i], 1e-3) << "i=" << i;
    } else {
      EXPECT_EQ(expected_output[i], actual[i]) << "i=" << i;
    }
  }
}
