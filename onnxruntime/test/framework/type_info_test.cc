// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "model_builder_utils.h"

#include "core/framework/onnxruntime_optional_type_info.h"
#include "core/framework/onnxruntime_map_type_info.h"
#include "core/framework/onnxruntime_sequence_type_info.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/onnxruntime_typeinfo.h"

namespace onnxruntime {
namespace test {

namespace mb = modelbuilder;

TEST(TypeInfoTests, TensorProto) {
  mb::Type tensor_type = {1, 2, 3, 4};

  auto tensor_type_info = OrtTypeInfo::FromTypeProto(tensor_type.value);
  ASSERT_EQ(ONNX_TYPE_TENSOR, tensor_type_info->type);
  ASSERT_NE(nullptr, tensor_type_info->data);
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, tensor_type_info->data->type);
  ASSERT_TRUE(SpanEq(AsSpan<int64_t>({1, 2, 3, 4}), tensor_type_info->data->shape.GetDims()));
}

TEST(TypeInfoTests, SequenceWithTensorElement) {
  mb::Type tensor_type = {1, 2, 3, 4};
  auto sequence_proto = mb::Type::MakeSequence(tensor_type.value);
  auto seq_type_info = OrtTypeInfo::FromTypeProto(sequence_proto.value);

  ASSERT_EQ(ONNX_TYPE_SEQUENCE, seq_type_info->type);
  ASSERT_NE(nullptr, seq_type_info->sequence_type_info);
  const auto& tensor_type_info = *seq_type_info->sequence_type_info->sequence_key_type_;

  ASSERT_EQ(ONNX_TYPE_TENSOR, tensor_type_info.type);
  ASSERT_NE(nullptr, tensor_type_info.data);
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, tensor_type_info.data->type);
  ASSERT_TRUE(SpanEq(AsSpan<int64_t>({1, 2, 3, 4}), tensor_type_info.data->shape.GetDims()));
}

TEST(TypeInfoTests, OptionalWithTensorProto) {
  mb::Type tensor_type = {1, 2, 3, 4};
  auto optional_proto = mb::Type::MakeOptional(tensor_type.value);

  auto optional_type_info = OrtTypeInfo::FromTypeProto(optional_proto.value);

  ASSERT_EQ(ONNX_TYPE_OPTIONAL, optional_type_info->type);
  ASSERT_NE(nullptr, optional_type_info->optional_type_info);
  ASSERT_NE(nullptr, optional_type_info->optional_type_info->contained_type_);

  const auto& contained_type = *optional_type_info->optional_type_info->contained_type_;
  ASSERT_EQ(ONNX_TYPE_TENSOR, contained_type.type);
  ASSERT_NE(nullptr, contained_type.data);
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, contained_type.data->type);
  ASSERT_TRUE(SpanEq(AsSpan<int64_t>({1, 2, 3, 4}), contained_type.data->shape.GetDims()));
}

#if !defined(DISABLE_ML_OPS)
TEST(TypeInfoTests, MapWithTensorValue) {
  mb::Type value_type = {1, 2, 3, 4};
  auto map_proto = mb::Type::MakeMap(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, value_type.value);
  auto map_type_info = OrtTypeInfo::FromTypeProto(map_proto.value);

  ASSERT_EQ(ONNX_TYPE_MAP, map_type_info->type);
  ASSERT_NE(nullptr, map_type_info->map_type_info);
  const auto& map_info = *map_type_info->map_type_info;

  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, map_info.map_key_type_);
  ASSERT_NE(nullptr, map_info.map_value_type_);
  const auto& tensor_type_info = *map_info.map_value_type_;

  ASSERT_EQ(ONNX_TYPE_TENSOR, tensor_type_info.type);
  ASSERT_NE(nullptr, tensor_type_info.data);
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, tensor_type_info.data->type);
  ASSERT_TRUE(SpanEq(AsSpan<int64_t>({1, 2, 3, 4}), tensor_type_info.data->shape.GetDims()));
}
#endif

}  // namespace test
}  // namespace onnxruntime