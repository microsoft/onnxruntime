// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers_fwd.h"
#include "core/framework/data_types_internal.h"
#include "lora/lora_adapters.h"
#include "lora/lora_format_version.h"
#include "lora/lora_format_utils.h"
#include "gtest/gtest.h"

#include <cmath>

namespace onnxruntime {
namespace test {

namespace {

constexpr const int kAdapterVersion = 1;
constexpr const int kModelVersion = 1;

template <class T>
struct ReadAndValidateData {
  void operator()(const Tensor& parameter) const {
    auto data = parameter.DataAsSpan<T>();
    for (size_t i = 0, size = data.size(); i < size; ++i) {
      ASSERT_EQ(static_cast<T>(i), data[i]);
    }
  }
};

template <>
struct ReadAndValidateData<float> {
  void operator()(const Tensor& parameter) const {
    auto data = parameter.DataAsSpan<float>();
    for (size_t i = 0, size = data.size(); i < size; ++i) {
      ASSERT_FALSE(std::isnan(data[i]));
      ASSERT_TRUE(std::isfinite(data[i]));
      ASSERT_EQ(static_cast<float>(i), data[i]);
    }
  }
};

template <>
struct ReadAndValidateData<double> {
  void operator()(const Tensor& parameter) const {
    auto data = parameter.DataAsSpan<double>();
    for (size_t i = 0, size = data.size(); i < size; ++i) {
      ASSERT_FALSE(std::isnan(data[i]));
      ASSERT_TRUE(std::isfinite(data[i]));
      ASSERT_EQ(static_cast<double>(i), data[i]);
    }
  }
};


template<>
struct ReadAndValidateData<BFloat16> {
  void operator()(const Tensor& parameter) const {
    auto data = parameter.DataAsSpan<BFloat16>();
    for (size_t i = 0, size = data.size(); i < size; ++i) {
      ASSERT_FALSE(data[i].IsNaN());
      ASSERT_FALSE(data[i].IsInfinity());
      ASSERT_EQ(static_cast<float>(i), data[i].ToFloat());
    }
  }
};

template <>
struct ReadAndValidateData<MLFloat16> {
  void operator()(const Tensor& parameter) const {
    auto data = parameter.DataAsSpan<MLFloat16>();
    for (size_t i = 0, size = data.size(); i < size; ++i) {
      ASSERT_FALSE(data[i].IsNaN());
      ASSERT_FALSE(data[i].IsInfinity());
      ASSERT_EQ(static_cast<float>(i), data[i].ToFloat());
    }
  }
};

auto verify_load = [](const lora::LoraAdapter& adapter) {
  ASSERT_EQ(kAdapterVersion, adapter.AdapterVersion());
  ASSERT_EQ(kModelVersion, adapter.ModelVersion());

  const auto param_num = adapter.GetParamNum();
  ASSERT_GE(param_num, 0U);

  InlinedVector<const char*> names;
  InlinedVector<OrtValue> ort_values;
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
    ASSERT_NE(tensor.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

    const auto shape = tensor.Shape().GetDims();
    ASSERT_EQ(2, shape.size());
    ASSERT_EQ(8, shape[0]);
    ASSERT_EQ(4, shape[1]);

    // Read all the elements to make sure they are accessible
    // only on CPU
    const auto& mem_info = tensor.Location();
    if (mem_info.device.Type() == OrtDevice::CPU) {
      utils::MLTypeCallDispatcher<float, double, int8_t, uint8_t,
                                  int16_t, uint16_t, int32_t, uint32_t,
                                  int64_t, uint64_t,
                                  BFloat16, MLFloat16>
          disp(tensor.GetElementType());
      disp.Invoke<ReadAndValidateData>(tensor);
    }
  }
};

}  // namespace

TEST(LoraAdapterTest, Load) {
  // See file creation code at testdata/lora/lora_unit_test_adapter.cc
  // This is float
  const std::filesystem::path file_path = "testdata/lora/lora_unit_test_adapter.fb";

  {
    // Test memory load
    lora::LoraAdapter lora_adapter;
    lora_adapter.Load(file_path);
    verify_load(lora_adapter);
  }

  {
    // Test memory map
    lora::LoraAdapter lora_adapter;
    lora_adapter.MemoryMap(file_path);
    verify_load(lora_adapter);
  }
}

}  // namespace test
}  // namespace onnxruntime
