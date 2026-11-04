// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>

#include "core/graph/onnx_protobuf.h"
#include "core/providers/openvino/ov_protobuf_utils.h"

#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

// Builds a FLOAT scalar whose value lives in raw_data, with an empty float_data
// field.
static TensorProto MakeRawDataFloatScalar(float value) {
  TensorProto tp;
  tp.set_data_type(TensorProto_DataType_FLOAT);
  tp.set_raw_data(&value, sizeof(float));
  return tp;
}

// Builds a FLOAT scalar whose value lives in the typed float_data field.
static TensorProto MakeFloatDataScalar(float value) {
  TensorProto tp;
  tp.set_data_type(TensorProto_DataType_FLOAT);
  tp.add_float_data(value);
  return tp;
}

TEST(OpenVINO_OvProtobufUtils, GetFromRawData) {
  TensorProto tp = MakeRawDataFloatScalar(4.0f);
  ASSERT_EQ(tp.float_data_size(), 0);  // value is only in raw_data

  EXPECT_FLOAT_EQ(openvino_ep::get_float_initializer_data(&tp), 4.0f);
}

TEST(OpenVINO_OvProtobufUtils, SetIntoRawData) {
  TensorProto tp = MakeRawDataFloatScalar(4.0f);
  ASSERT_EQ(tp.float_data_size(), 0);

  openvino_ep::set_float_initializer_data(&tp, 0.5f);

  // The write must land in raw_data (the field that actually holds the value),
  // and must be readable back through the getter.
  ASSERT_GE(tp.raw_data().size(), sizeof(float));
  float stored;
  std::memcpy(&stored, tp.raw_data().data(), sizeof(float));
  EXPECT_FLOAT_EQ(stored, 0.5f);
  EXPECT_FLOAT_EQ(openvino_ep::get_float_initializer_data(&tp), 0.5f);
}

TEST(OpenVINO_OvProtobufUtils, GetFromFloatData) {
  TensorProto tp = MakeFloatDataScalar(3.0f);
  EXPECT_FLOAT_EQ(openvino_ep::get_float_initializer_data(&tp), 3.0f);
}

TEST(OpenVINO_OvProtobufUtils, SetIntoFloatData) {
  TensorProto tp = MakeFloatDataScalar(3.0f);
  openvino_ep::set_float_initializer_data(&tp, 7.0f);
  EXPECT_FLOAT_EQ(tp.float_data(0), 7.0f);
  EXPECT_FLOAT_EQ(openvino_ep::get_float_initializer_data(&tp), 7.0f);
}

}  // namespace test
}  // namespace onnxruntime
