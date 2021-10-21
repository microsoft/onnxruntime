// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include <gtest/gtest.h>

extern std::unique_ptr<Ort::Env> ort_env;

static void TestModelInfo(const Ort::Session& session, bool is_input, const std::vector<int64_t>& dims) {
  size_t input_count;
  if (is_input) {
    input_count = session.GetInputCount();
  } else {
    input_count = session.GetOutputCount();
  }
  ASSERT_EQ(1u, input_count);
  Ort::TypeInfo input_type_info = is_input ? session.GetInputTypeInfo(0) : session.GetOutputTypeInfo(0);
  ASSERT_NE(nullptr, input_type_info);

  ONNXType otype = input_type_info.GetONNXType();
  ASSERT_EQ(ONNX_TYPE_TENSOR, otype);

  auto p = input_type_info.GetTensorTypeAndShapeInfo();
  ASSERT_NE(nullptr, p);

  ONNXTensorElementDataType ele_type = p.GetElementType();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ele_type);
  ASSERT_EQ(dims.size(), p.GetDimensionsCount());
  std::vector<int64_t> real_dims = p.GetShape();
  ASSERT_EQ(real_dims, dims);
}

TEST(CApiTest, input_output_type_info) {
  constexpr const ORTCHAR_T* model_uri = ORT_TSTR("../models/opset8/test_squeezenet/model.onnx");
  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, model_uri, session_options);
  TestModelInfo(session, true, {1, 3, 224, 224});
  TestModelInfo(session, false, {1, 1000, 1, 1});
}
