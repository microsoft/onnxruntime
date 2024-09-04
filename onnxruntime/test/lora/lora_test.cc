// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora/lora_adapters.h"
#include "lora/lora_format_version.h"
#include "lora/lora_format_utils.h"
#include "gtest/gtest.h"

#include <fstream>

namespace onnxruntime {
namespace test {

// TEST(LoraFormatTest, CreateAdapter) {
//   // Generate a random sequence of floats
//   // shape = {8, 4}
//   constexpr std::array<int64_t, 2> shape = {8, 4};
//   std::vector<float> param_1(32);
//   std::iota(param_1.begin(), param_1.end(), 0.0f);
//
//   std::vector<float> param_2(32);
//   std::iota(param_2.begin(), param_2.end(), 33.0f);
//
//   flatbuffers::FlatBufferBuilder builder;
//   std::vector<flatbuffers::Offset<lora::Parameter>> params;
//   params.reserve(2);
//   flatbuffers::Offset<lora::Parameter> fbs_param_1, fbs_param_2;
//   auto byte_span = ReinterpretAsSpan<uint8_t>(gsl::make_span(param_1));
//   lora::utils::SaveLoraParameter(builder, "param_1", lora::TensorDataType_FLOAT, shape,
//                                  byte_span, fbs_param_1);
//   params.push_back(fbs_param_1);
//
//   byte_span = ReinterpretAsSpan<uint8_t>(gsl::make_span(param_2));
//   lora::utils::SaveLoraParameter(builder, "param_2", lora::TensorDataType_FLOAT, shape,
//                                  byte_span, fbs_param_2);
//   params.push_back(fbs_param_2);
//
//   auto fbs_params = builder.CreateVector(params);
//   auto fbs_adapter = lora::CreateAdapter(builder, lora::kLoraFormatVersion, 1, 1, fbs_params);
//   builder.Finish(fbs_adapter, lora::AdapterIdentifier());
//
//   constexpr const char* const file_name =
//       "D:/dmitrism/Downloads/generate-test-model/param_conversion/lora_unit_test_adapter.fb";
//   std::ofstream file(file_name, std::ios::binary);
//   ASSERT_TRUE(file.is_open());
//
//   ASSERT_FALSE(file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize()).fail());
//   ASSERT_FALSE(file.flush().fail());
//   file.close();
// }

TEST(LoraAdapterTest, Load) {
  // XXX: put this into test directory
  const std::filesystem::path file_path = "testdata/lora/lora_unit_test_adapter.fb";

  auto verify_load = [](const lora::LoraAdapter& adapter) {
    const auto param_num = adapter.GetParamNum();
    ASSERT_GE(param_num, 0U);

    std::vector<const char*> names;
    std::vector<OrtValue> ort_values;
    names.reserve(param_num);
    ort_values.reserve(param_num);

    adapter.OutputAdaptersParameters(std::back_inserter(names), std::back_inserter(ort_values));
    ASSERT_EQ(param_num, names.size());
    ASSERT_EQ(param_num, ort_values.size());

    for (size_t i = 0; i < param_num; ++i) {
      const auto& name = names[i];
      const auto& ort_value = ort_values[i];
      ASSERT_TRUE(name != nullptr);
      ASSERT_TRUE(ort_value.IsTensor());

      const auto& tensor = ort_value.Get<Tensor>();
      ASSERT_EQ(tensor.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

      const auto shape = tensor.Shape().GetDims();
      ASSERT_EQ(2, shape.size());
      ASSERT_EQ(8, shape[0]);
      ASSERT_EQ(4, shape[1]);

      // Read all the elements to make sure they are accessible
      const auto data = tensor.DataAsSpan<float>();
      for (size_t j = 0, lim = data.size(); j < lim; ++j) {
        ASSERT_EQ(static_cast<float>(j), data[j]);
      }
    }
  };

  {
    lora::LoraAdapter lora_adapter;
    lora_adapter.Load(file_path);
    verify_load(lora_adapter);
  }

  {
    lora::LoraAdapter lora_adapter;
    lora_adapter.MemoryMap(file_path);
    verify_load(lora_adapter);
  }
}

}  // namespace test
}  // namespace onnxruntime
