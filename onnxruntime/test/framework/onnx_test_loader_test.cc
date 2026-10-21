// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstring>
#include <utility>

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/onnx/callback.h"
#include "test/onnx/mem_buffer.h"
#include "test/onnx/tensorprotoutils.h"

namespace onnxruntime {
namespace test {
namespace {

using ElementTypes = std::pair<int, ONNXTensorElementDataType>;

class OnnxTestLoaderTest : public testing::TestWithParam<ElementTypes> {
 protected:
  void CheckTensor(bool raw_data) {
    const auto [proto_type, api_type] = GetParam();
    EXPECT_EQ(CApiElementTypeFromProtoType(proto_type), api_type);

    constexpr int64_t element_count = 5;
    const size_t byte_count = api_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2 ||
                                      api_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2
                                  ? 2
                                  : 5;
    const std::array<uint8_t, element_count> data{0x12, 0x34, 0x56, 0x78, 0x7f};
    ONNX_NAMESPACE::TensorProto proto;
    proto.set_data_type(proto_type);
    proto.add_dims(element_count);
    if (raw_data) {
      proto.set_raw_data(data.data(), byte_count);
    } else {
      for (size_t i = 0; i < byte_count; ++i) {
        proto.add_int32_data(data[i]);
      }
    }

    std::array<uint8_t, element_count> buffer{};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    MemBuffer memory(buffer.data(), byte_count, *memory_info);
    Ort::Value value{nullptr};
    OrtCallback deleter;
    auto status = TensorProtoToMLValue(proto, memory, value, deleter);
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    EXPECT_EQ(deleter.f, nullptr);

    const auto tensor_info = value.GetTensorTypeAndShapeInfo();
    EXPECT_EQ(tensor_info.GetElementType(), api_type);
    EXPECT_EQ(tensor_info.GetElementCount(), static_cast<size_t>(element_count));
    ASSERT_EQ(tensor_info.GetDimensionsCount(), 1u);
    EXPECT_EQ(tensor_info.GetShape()[0], element_count);
    EXPECT_EQ(std::memcmp(value.GetTensorRawData(), data.data(), byte_count), 0);
  }
};

TEST_P(OnnxTestLoaderTest, RawDataPreservesElementTypeAndPayload) {
  CheckTensor(true);
}

TEST_P(OnnxTestLoaderTest, Int32DataPreservesElementTypeAndPayload) {
  CheckTensor(false);
}

constexpr ElementTypes kElementTypes[] = {
    {ONNX_NAMESPACE::TensorProto_DataType_UINT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8},
    {ONNX_NAMESPACE::TensorProto_DataType_INT2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT2, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2},
#if !defined(DISABLE_FLOAT8_TYPES)
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E8M0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0},
#endif
};

INSTANTIATE_TEST_SUITE_P(TensorProto, OnnxTestLoaderTest, testing::ValuesIn(kElementTypes));

}  // namespace
}  // namespace test
}  // namespace onnxruntime
