// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora/lora_format_version.h"
#include "lora/lora_format_utils.h"
#include "gtest/gtest.h"

#include <fstream>

namespace onnxruntime {
namespace test {

TEST(LoraFormatTest, CreateAdapter) {
  // generate a random sequence of floats
  // shape = {8, 4}
  constexpr std::array<int64_t, 2> shape = {8, 4};
  std::vector<float> param_1(32);
  std::iota(param_1.begin(), param_1.end(), 0.0f);

  std::vector<float> param_2(32);
  std::iota(param_2.begin(), param_2.end(), 33.0f);

  flatbuffers::flatbufferbuilder builder;
  std::vector<flatbuffers::offset<lora::parameter>> params;
  params.reserve(2);
  flatbuffers::offset<lora::parameter> fbs_param_1, fbs_param_2;
  auto byte_span = reinterpretasspan<uint8_t>(gsl::make_span(param_1));
  lora::utils::saveloraparameter(builder, "param_1", lora::tensordatatype_float, shape,
                                 byte_span, fbs_param_1);
  params.push_back(fbs_param_1);

  byte_span = reinterpretasspan<uint8_t>(gsl::make_span(param_2));
  lora::utils::saveloraparameter(builder, "param_2", lora::tensordatatype_float, shape,
                                 byte_span, fbs_param_2);
  params.push_back(fbs_param_2);

  auto fbs_params = builder.createvector(params);
  auto fbs_adapter = lora::createadapter(builder, lora::kloraformatversion, 1, 1, fbs_params);
  builder.finish(fbs_adapter, lora::adapteridentifier());

  constexpr const char* const file_name =
      "d:/dmitrism/downloads/generate-test-model/param_conversion/lora_unit_test_adapter.fb";
  std::ofstream file(file_name, std::ios::binary);
  assert_true(file.is_open());

  assert_false(file.write(reinterpret_cast<const char*>(builder.getbufferpointer()), builder.getsize()).fail());
  assert_false(file.flush().fail());
  file.close();
}
}
}