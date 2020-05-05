// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"
#include "test.h"
#include "RawApiHelpers.h"

namespace ml = Microsoft::AI::MachineLearning;

void RunOnDevice(ml::learning_model& model, ml::learning_model_device& device, bool copy_inputs)
{
    const wchar_t input_name[] = L"data_0";
    const wchar_t output_name[] = L"softmaxout_1";

    std::unique_ptr<ml::learning_model_session> session = nullptr;
    WINML_EXPECT_NO_THROW(session = std::make_unique<ml::learning_model_session>(model, device));

    std::unique_ptr<ml::learning_model_binding> binding = nullptr;
    WINML_EXPECT_NO_THROW(binding = std::make_unique<ml::learning_model_binding>(*session.get()));

    auto input_shape = std::vector<ml::tensor_shape_type>{ 1, 3, 224, 224 };
    auto input_data = std::vector<float>(1 * 3 * 224 * 224);
    auto output_shape = std::vector<ml::tensor_shape_type>{ 1, 1000, 1, 1 };

    if (copy_inputs)
    {
      WINML_EXPECT_HRESULT_SUCCEEDED(
        binding->bind<float>(
          input_name, _countof(input_name) - 1,
          input_shape.data(), input_shape.size(),
          input_data.data(), input_data.size()));
    }
    else
    {
      WINML_EXPECT_HRESULT_SUCCEEDED(
        binding->bind_as_reference<float>(
          input_name, _countof(input_name) - 1,
          input_shape.data(), input_shape.size(),
          input_data.data(), input_data.size()));
    }

    WINML_EXPECT_HRESULT_SUCCEEDED(
        binding->bind<float>(
            output_name, _countof(output_name) - 1,
            output_shape.data(), output_shape.size()));

    ml::learning_model_results results = session->evaluate(*binding.get());

    float* p_buffer = nullptr;
    size_t buffer_size = 0;
    WINML_EXPECT_HRESULT_SUCCEEDED(
        0 == results.get_output(
            output_name,
            _countof(output_name) - 1,
            reinterpret_cast<void**>(&p_buffer),
            &buffer_size)
    );
}
