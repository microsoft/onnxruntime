// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/graph/onnx_protobuf.h"

using ONNX_NAMESPACE::Utils::DataTypeUtils;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(OpUtilsTest, TestPTYPE) {
  DataType p1 = DataTypeUtils::ToType("tensor(int32)");
  DataType p2 = DataTypeUtils::ToType("tensor(int32)");
  DataType p3 = DataTypeUtils::ToType("tensor(int32)");
  EXPECT_EQ(p1, p2);
  EXPECT_EQ(p2, p3);
  EXPECT_EQ(p1, p3);
  DataType p4 = DataTypeUtils::ToType("seq(tensor(int32))");
  DataType p5 = DataTypeUtils::ToType("seq(tensor(int32))");
  DataType p6 = DataTypeUtils::ToType("seq(tensor(int32))");
  EXPECT_EQ(p4, p5);
  EXPECT_EQ(p5, p6);
  EXPECT_EQ(p4, p6);

  TypeProto t1 = DataTypeUtils::ToTypeProto(p1);
  EXPECT_TRUE(t1.has_tensor_type());
  EXPECT_TRUE(t1.tensor_type().has_elem_type());
  EXPECT_EQ(t1.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t2 = DataTypeUtils::ToTypeProto(p2);
  EXPECT_TRUE(t2.has_tensor_type());
  EXPECT_TRUE(t2.tensor_type().has_elem_type());
  EXPECT_EQ(t2.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t3 = DataTypeUtils::ToTypeProto(p3);
  EXPECT_TRUE(t3.has_tensor_type());
  EXPECT_TRUE(t3.tensor_type().has_elem_type());
  EXPECT_EQ(t3.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t4 = DataTypeUtils::ToTypeProto(p4);
  EXPECT_TRUE(t4.has_sequence_type());
  EXPECT_TRUE(t4.sequence_type().has_elem_type());
  EXPECT_TRUE(t4.sequence_type().elem_type().has_tensor_type());
  EXPECT_TRUE(t4.sequence_type().elem_type().tensor_type().has_elem_type());
  EXPECT_EQ(t4.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t5 = DataTypeUtils::ToTypeProto(p5);
  EXPECT_TRUE(t5.has_sequence_type());
  EXPECT_TRUE(t5.sequence_type().has_elem_type());
  EXPECT_TRUE(t5.sequence_type().elem_type().has_tensor_type());
  EXPECT_TRUE(t5.sequence_type().elem_type().tensor_type().has_elem_type());
  EXPECT_EQ(t5.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t6 = DataTypeUtils::ToTypeProto(p6);
  EXPECT_TRUE(t6.has_sequence_type());
  EXPECT_TRUE(t6.sequence_type().has_elem_type());
  EXPECT_TRUE(t6.sequence_type().elem_type().has_tensor_type());
  EXPECT_TRUE(t6.sequence_type().elem_type().tensor_type().has_elem_type());
  EXPECT_EQ(t6.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
}
}  // namespace test
}  // namespace onnxruntime
