// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers_fwd.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/to_tensor_proto_element_type.h"

#include "test/util/include/default_providers.h"

#include "core/session/lora_adapters.h"
#include "lora/adapter_format_version.h"
#include "lora/adapter_format_utils.h"

#include "gtest/gtest.h"
#include <cmath>

#include "test/util/include/asserts.h"

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
    ASSERT_EQ(2U, shape.size());
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
    const auto dt = data_types[i];

    using namespace adapters;
    if (dt == TensorDataType::STRING ||
        dt == TensorDataType::BOOL ||
        dt == TensorDataType::COMPLEX64 ||
        dt == TensorDataType::COMPLEX128 ||
        static_cast<int>(dt) >= static_cast<int>(TensorDataType::BFLOAT16))
      continue;

    onnxruntime::utils::MLTypeCallDispatcher<float, double, int8_t, uint8_t,
                                             int16_t, uint16_t, int32_t, uint32_t,
                                             int64_t, uint64_t,
                                             BFloat16, MLFloat16>
        disp(static_cast<int32_t>(data_types[i]));
    disp.Invoke<TestDataType>();
  }
}

TEST(LoraAdapterTest, CreateOrtValueOverLoraParameter_ValidParam) {
  // Build a valid adapter with a single float parameter, then call
  // CreateOrtValueOverLoraParameter on the deserialized Parameter.
  constexpr std::array<int64_t, 2> shape = {8, 4};
  InlinedVector<float> data(32);
  std::iota(data.begin(), data.end(), 0.f);

  adapters::utils::AdapterFormatBuilder adapter_builder;
  adapter_builder.AddParameter("valid_param", adapters::TensorDataType::FLOAT,
                               shape, ReinterpretAsSpan<const uint8_t>(gsl::make_span(data)));

  auto buffer = adapter_builder.Finish(kAdapterVersion, kModelVersion);

  const auto* adapter = adapters::utils::ValidateAndGetAdapterFromBytes(buffer);
  ASSERT_NE(adapter, nullptr);
  ASSERT_NE(adapter->parameters(), nullptr);
  ASSERT_EQ(adapter->parameters()->size(), 1u);

  const auto* param = adapter->parameters()->Get(0);
  auto [name, ort_value] = adapters::utils::CreateOrtValueOverLoraParameter(*param);

  ASSERT_EQ(name, "valid_param");
  ASSERT_TRUE(ort_value.IsTensor());

  const auto& tensor = ort_value.Get<Tensor>();
  ASSERT_EQ(tensor.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto dims = tensor.Shape().GetDims();
  ASSERT_EQ(dims.size(), 2u);
  ASSERT_EQ(dims[0], 8);
  ASSERT_EQ(dims[1], 4);

  auto result_span = tensor.DataAsSpan<float>();
  ASSERT_EQ(result_span.size(), 32u);
  for (size_t i = 0; i < result_span.size(); ++i) {
    ASSERT_EQ(static_cast<float>(i), result_span[i]);
  }
}

#ifndef ORT_NO_EXCEPTIONS

namespace {
// Helper that wraps a single Parameter offset into a finished Adapter flatbuffer
// and returns a pointer to the deserialized Parameter.
// The FlatBufferBuilder must outlive the returned pointer.
const adapters::Parameter* BuildAdapterAndGetParam(flatbuffers::FlatBufferBuilder& fbb,
                                                   flatbuffers::Offset<adapters::Parameter> param_offset) {
  auto params_offset = fbb.CreateVector(&param_offset, 1);
  auto adapter_offset = adapters::CreateAdapter(
      fbb, adapters::kAdapterFormatVersion, kAdapterVersion, kModelVersion, params_offset);
  adapters::FinishAdapterBuffer(fbb, adapter_offset);

  const auto* adapter = adapters::GetAdapter(fbb.GetBufferPointer());
  return adapter->parameters()->Get(0);
}
}  // namespace

TEST(LoraAdapterTest, CreateOrtValueOverLoraParameter_RawDataSizeMismatch) {
  // Craft a flatbuffer Parameter where raw_data has fewer bytes than
  // shape (8 x 4) * sizeof(float) = 128 bytes.
  // We supply only 64 bytes (half the expected amount) so the validation
  // inside CreateOrtValueOverLoraParameter must throw.
  flatbuffers::FlatBufferBuilder fbb;

  auto name_offset = fbb.CreateString("bad_param");
  std::vector<int64_t> dims = {8, 4};
  auto dims_offset = fbb.CreateVector(dims);

  // 8 * 4 floats = 32 elements = 128 bytes expected.
  // Provide only 64 bytes (16 floats worth) to trigger the mismatch.
  std::vector<uint8_t> short_data(64, 0);
  fbb.ForceVectorAlignment(short_data.size(), sizeof(uint8_t), 8);
  auto data_offset = fbb.CreateVector(short_data);

  auto param_offset = adapters::CreateParameter(
      fbb, name_offset, dims_offset, adapters::TensorDataType::FLOAT, data_offset);

  // Wrap the single parameter inside an Adapter so the buffer is valid flatbuffers.
  auto params_offset = fbb.CreateVector(&param_offset, 1);
  auto adapter_offset = adapters::CreateAdapter(
      fbb, adapters::kAdapterFormatVersion, kAdapterVersion, kModelVersion, params_offset);
  adapters::FinishAdapterBuffer(fbb, adapter_offset);

  auto* buf = fbb.GetBufferPointer();

  // Retrieve the Parameter from the Adapter
  const auto* adapter = adapters::GetAdapter(buf);
  ASSERT_NE(adapter, nullptr);
  ASSERT_NE(adapter->parameters(), nullptr);
  ASSERT_EQ(adapter->parameters()->size(), 1u);

  const auto* param = adapter->parameters()->Get(0);
  ASSERT_NE(param, nullptr);

  // The raw_data is 64 bytes but shape says 8x4 floats = 128 bytes.
  // CreateOrtValueOverLoraParameter must throw.
  ASSERT_THROW(adapters::utils::CreateOrtValueOverLoraParameter(*param), OnnxRuntimeException);
}

TEST(LoraAdapterTest, CreateOrtValueOverLoraParameter_ExcessRawData) {
  // Craft a flatbuffer Parameter where raw_data has MORE bytes than expected.
  // Shape (2, 2) with float => 4 elements => 16 bytes expected, but we supply 32.
  flatbuffers::FlatBufferBuilder fbb;

  auto name_offset = fbb.CreateString("excess_param");
  std::vector<int64_t> dims = {2, 2};
  auto dims_offset = fbb.CreateVector(dims);

  // 2 * 2 floats = 4 elements = 16 bytes expected.  Supply 32.
  std::vector<uint8_t> excess_data(32, 0);
  fbb.ForceVectorAlignment(excess_data.size(), sizeof(uint8_t), 8);
  auto data_offset = fbb.CreateVector(excess_data);

  auto param_offset = adapters::CreateParameter(
      fbb, name_offset, dims_offset, adapters::TensorDataType::FLOAT, data_offset);

  auto params_offset = fbb.CreateVector(&param_offset, 1);
  auto adapter_offset = adapters::CreateAdapter(
      fbb, adapters::kAdapterFormatVersion, kAdapterVersion, kModelVersion, params_offset);
  adapters::FinishAdapterBuffer(fbb, adapter_offset);

  const auto* adapter = adapters::GetAdapter(fbb.GetBufferPointer());
  ASSERT_NE(adapter, nullptr);

  const auto* param = adapter->parameters()->Get(0);
  ASSERT_NE(param, nullptr);

  // Excess data should also trigger the mismatch throw.
  ASSERT_THROW(adapters::utils::CreateOrtValueOverLoraParameter(*param), OnnxRuntimeException);
}

TEST(LoraAdapterTest, CreateOrtValueOverLoraParameter_MissingName) {
  // Parameter with null name should throw gracefully.
  flatbuffers::FlatBufferBuilder fbb;

  std::vector<int64_t> dims = {2, 2};
  std::vector<uint8_t> raw_data(16, 0);  // 2*2 floats = 16 bytes

  // name is nullptr, all other fields are valid
  auto param_offset = adapters::CreateParameterDirect(
      fbb, /*name=*/nullptr, &dims, adapters::TensorDataType::FLOAT, &raw_data);

  const auto* param = BuildAdapterAndGetParam(fbb, param_offset);
  ASSERT_NE(param, nullptr);
  ASSERT_EQ(param->name(), nullptr);

  ASSERT_THROW(adapters::utils::CreateOrtValueOverLoraParameter(*param), OnnxRuntimeException);
}

TEST(LoraAdapterTest, CreateOrtValueOverLoraParameter_MissingDims) {
  // Parameter with null dims should throw gracefully.
  flatbuffers::FlatBufferBuilder fbb;

  std::vector<uint8_t> raw_data(16, 0);

  // dims is nullptr
  auto param_offset = adapters::CreateParameterDirect(
      fbb, "no_dims_param", /*dims=*/nullptr, adapters::TensorDataType::FLOAT, &raw_data);

  const auto* param = BuildAdapterAndGetParam(fbb, param_offset);
  ASSERT_NE(param, nullptr);
  ASSERT_EQ(param->dims(), nullptr);

  ASSERT_THROW(adapters::utils::CreateOrtValueOverLoraParameter(*param), OnnxRuntimeException);
}

TEST(LoraAdapterTest, CreateOrtValueOverLoraParameter_EmptyDims) {
  // Parameter with an empty dims vector should throw gracefully.
  flatbuffers::FlatBufferBuilder fbb;

  std::vector<int64_t> empty_dims;
  std::vector<uint8_t> raw_data(16, 0);

  auto param_offset = adapters::CreateParameterDirect(
      fbb, "empty_dims_param", &empty_dims, adapters::TensorDataType::FLOAT, &raw_data);

  const auto* param = BuildAdapterAndGetParam(fbb, param_offset);
  ASSERT_NE(param, nullptr);
  ASSERT_EQ(param->dims()->size(), 0u);

  ASSERT_THROW(adapters::utils::CreateOrtValueOverLoraParameter(*param), OnnxRuntimeException);
}

TEST(LoraAdapterTest, CreateOrtValueOverLoraParameter_MissingRawData) {
  // Parameter with null raw_data should throw gracefully.
  flatbuffers::FlatBufferBuilder fbb;

  std::vector<int64_t> dims = {2, 2};

  // raw_data is nullptr
  auto param_offset = adapters::CreateParameterDirect(
      fbb, "no_data_param", &dims, adapters::TensorDataType::FLOAT, /*raw_data=*/nullptr);

  const auto* param = BuildAdapterAndGetParam(fbb, param_offset);
  ASSERT_NE(param, nullptr);
  ASSERT_EQ(param->raw_data(), nullptr);

  ASSERT_THROW(adapters::utils::CreateOrtValueOverLoraParameter(*param), OnnxRuntimeException);
}

TEST(LoraAdapterTest, CreateOrtValueOverLoraParameter_UndefinedDataType) {
  // Parameter with UNDEFINED data_type should throw gracefully.
  flatbuffers::FlatBufferBuilder fbb;

  std::vector<int64_t> dims = {2, 2};
  std::vector<uint8_t> raw_data(16, 0);

  // data_type defaults to UNDEFINED when not set
  auto param_offset = adapters::CreateParameterDirect(
      fbb, "undef_type_param", &dims, adapters::TensorDataType::UNDEFINED, &raw_data);

  const auto* param = BuildAdapterAndGetParam(fbb, param_offset);
  ASSERT_NE(param, nullptr);
  ASSERT_EQ(param->data_type(), adapters::TensorDataType::UNDEFINED);

  ASSERT_THROW(adapters::utils::CreateOrtValueOverLoraParameter(*param), OnnxRuntimeException);
}

#endif  // ORT_NO_EXCEPTIONS

#ifdef USE_CUDA
TEST(LoraAdapterTest, VerifyDeviceCopy) {
  auto cpu_ep = DefaultCpuExecutionProvider();
  auto cpu_allocator = cpu_ep->CreatePreferredAllocators()[0];
  auto cuda_ep = DefaultCudaExecutionProvider();
  auto cuda_allocator = cuda_ep->CreatePreferredAllocators()[0];

  auto gpu_transfer = cuda_ep->GetDataTransfer();

  auto test_params = GenerateTestParameters<float>()();
  lora::LoraAdapter adapter(std::move(cuda_allocator));
  adapter.Load(std::move(test_params));

  auto [begin, end] = adapter.GetParamIterators();
  for (; begin != end; ++begin) {
    const auto& [_, param] = *begin;
    const auto& tensor_device = param.GetDeviceOrMapped().Get<Tensor>();
    const auto& mem_info = tensor_device.Location();
    ASSERT_EQ(mem_info.device.Type(), OrtDevice::GPU);
    ASSERT_EQ(mem_info.device.Vendor(), OrtDevice::VendorIds::NVIDIA);

    const auto& tensor_cpu = param.GetMapped().Get<Tensor>();
    ASSERT_EQ(tensor_cpu.Shape().Size(), tensor_device.Shape().Size());

    Tensor copy(tensor_cpu.DataType(), tensor_cpu.Shape(), cpu_allocator);
    ASSERT_TRUE(gpu_transfer->CanCopy(tensor_device.Location().device,
                                      copy.Location().device));
    ASSERT_STATUS_OK(gpu_transfer->CopyTensor(tensor_device, copy));

    auto expected_span = tensor_cpu.DataAsSpan<float>();
    auto copy_span = copy.DataAsSpan<float>();

    ASSERT_EQ(expected_span, copy_span);
  }
}
#endif
}  // namespace test
}  // namespace onnxruntime
