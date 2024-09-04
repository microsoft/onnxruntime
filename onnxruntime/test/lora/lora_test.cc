// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lora/lora_adapters.h"
#include "lora/lora_format_version.h"
#include "lora/lora_format_utils.h"
#include "gtest/gtest.h"

#include <fstream>

namespace onnxruntime {
namespace test {

TEST(LoraAdapterTest, Load) {
  // See file creation code at testdata/lora/lora_unit_test_adapter.cc
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
