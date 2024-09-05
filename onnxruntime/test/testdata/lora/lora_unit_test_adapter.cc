// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora/lora_format_version.h"
#include "lora/lora_format_utils.h"
#include "gtest/gtest.h"

#include <fstream>

namespace onnxruntime {
namespace test {

constexpr const int kAdapterVersion = 1;
constexpr const int kModelVersion = 1;
   

EST(LoraFormatTest, CreateAdapter) {
  // generate a random sequence of floats
  // shape = {8, 4}
  constexpr std::array<int64_t, 2> shape = {8, 4};
  std::vector<float> param_1(32);
  std::iota(param_1.begin(), param_1.end(), 0.0f);

  std::vector<float> param_2(32);
  std::iota(param_2.begin(), param_2.end(), 33.0f);

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<lora::Parameter>> params;
  params.reserve(2);
  flatbuffers::Offset<lora::Parameter> fbs_param_1, fbs_param_2;
  auto byte_span = ReinterpretAsSpan<uint8_t>(gsl::make_span(param_1));
  lora::utils::SaveLoraParameter(builder, "param_1", lora::TensorDataType::FLOAT, shape,
                                 byte_span, fbs_param_1);
  params.push_back(fbs_param_1);

  byte_span = ReinterpretAsSpan<uint8_t>(gsl::make_span(param_2));
  lora::utils::SaveLoraParameter(builder, "param_2", lora::TensorDataType::FLOAT, shape,
                                 byte_span, fbs_param_2);
  params.push_back(fbs_param_2);

  auto fbs_params = builder.CreateVector(params);
  auto fbs_adapter = lora::CreateAdapter(builder, lora::kLoraFormatVersion, 1, 1, fbs_params);
  builder.Finish(fbs_adapter, lora::AdapterIdentifier());

  constexpr const char* const file_name =
      "d:/dmitrism/downloads/generate-test-model/param_conversion/lora_unit_test_adapter.fb";
  std::ofstream file(file_name, std::ios::binary);
  ASSERT_TRUE(file.is_open());

  ASSERT_FALSE(file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize()).fail());
  ASSERT_FALSE(file.flush().fail());
  file.close();
}

}
}