// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers_fwd.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/to_tensor_proto_element_type.h"

#include "lora/lora_adapters.h"
#include "lora/adapter_format_version.h"
#include "lora/adapter_format_utils.h"
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
    for (size_t i = static_cast<size_t>(data[0]), size = data.size(); i < size; ++i) {
      ASSERT_EQ(static_cast<T>(i), data[i]);
    }
  }
};

template <>
struct ReadAndValidateData<float> {
  void operator()(const Tensor& parameter) const {
    auto data = parameter.DataAsSpan<float>();
    for (size_t i = static_cast<size_t>(data[0]), size = data.size(); i < size; ++i) {
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
    for (size_t i = static_cast<size_t>(data[0]), size = data.size(); i < size; ++i) {
      ASSERT_FALSE(std::isnan(data[i]));
      ASSERT_TRUE(std::isfinite(data[i]));
      ASSERT_EQ(static_cast<double>(i), data[i]);
    }
  }
};

template <>
struct ReadAndValidateData<BFloat16> {
  void operator()(const Tensor& parameter) const {
    auto data = parameter.DataAsSpan<BFloat16>();
    for (size_t i = static_cast<size_t>(data[0].ToFloat()), size = data.size(); i < size; ++i) {
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
    for (size_t i = static_cast<size_t>(data[0].ToFloat()), size = data.size(); i < size; ++i) {
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
  ASSERT_EQ(param_num, 2U);

  InlinedVector<const char*> names;
  InlinedVector<const OrtValue*> ort_values;
  names.reserve(param_num);
  ort_values.reserve(param_num);

  adapter.OutputAdapterParameters(std::back_inserter(names), std::back_inserter(ort_values));
  ASSERT_EQ(param_num, names.size());
  ASSERT_EQ(param_num, ort_values.size());

  for (size_t i = 0; i < param_num; ++i) {
    const auto& name = names[i];
    const auto* ort_value = ort_values[i];
    ASSERT_TRUE(name != nullptr);
    ASSERT_TRUE(ort_value->IsTensor());

    const auto& tensor = ort_value->Get<Tensor>();
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

constexpr const std::array<int64_t, 2> param_shape = {8, 4};

template <class T>
struct CreateParam {
  InlinedVector<T> operator()() const {
    InlinedVector<T> param(32);
    std::iota(param.begin(), param.end(), T{0});
    return param;
  }
};

template <class T>
struct GenerateTestParameters {
  std::vector<uint8_t> operator()() const {
    constexpr const auto data_type = utils::ToTensorProtoElementType<T>();

    InlinedVector<T> param_1(32);
    InlinedVector<T> param_2(32);
    if constexpr (std::is_same<T, MLFloat16>::value || std::is_same<T, BFloat16>::value) {
      for (float f = 0.f; f < 32; ++f) {
        param_1[static_cast<size_t>(f)] = static_cast<T>(f);
        param_2[static_cast<size_t>(f)] = static_cast<T>(f + 32);
      }
    } else {
      std::iota(param_1.begin(), param_1.end(), T{0});
      std::iota(param_2.begin(), param_2.end(), T{32});
    }

    adapters::utils::AdapterFormatBuilder adapter_builder;
    adapter_builder.AddParameter("param_1", static_cast<adapters::TensorDataType>(data_type),
                                 param_shape, ReinterpretAsSpan<const uint8_t>(gsl::make_span(param_1)));
    adapter_builder.AddParameter("param_2", static_cast<adapters::TensorDataType>(data_type),
                                 param_shape, ReinterpretAsSpan<const uint8_t>(gsl::make_span(param_2)));

    return adapter_builder.Finish(kAdapterVersion, kModelVersion);
  }
};

template <class T>
struct TestDataType {
  void operator()() const {
    const auto test_params = GenerateTestParameters<T>()();
    lora::LoraAdapter lora_adapter;
    lora_adapter.Load(std::move(test_params));
    verify_load(lora_adapter);
  }
};

}  // namespace

TEST(LoraAdapterTest, Load) {
  // Test different data types
  const auto data_types = gsl::make_span(adapters::EnumValuesTensorDataType());
  for (size_t i = 1, size = data_types.size(); i < size; ++i) {
    if (i == 8 || i == 9 || i == 14 || i == 15 || (i > 16 && i < 21))
      continue;

    utils::MLTypeCallDispatcher<float, double, int8_t, uint8_t,
                                int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t,
                                BFloat16, MLFloat16>
        disp(static_cast<int32_t>(data_types[i]));
    disp.Invoke<TestDataType>();
  }
}

}  // namespace test
}  // namespace onnxruntime
