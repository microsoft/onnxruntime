// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <google/protobuf/stubs/status.h>

#include "gtest/gtest.h"

#include "predict.pb.h"
#include "hosting/util.h"

namespace onnxruntime {
namespace hosting {
namespace test {
namespace protobufutil = google::protobuf::util;

TEST(PositiveTests, UtilTest) {
  std::string input_json = "";
  onnx::hosting::PredictRequest request;
  protobufutil::Status status = onnx::hosting::GetRequestFromJson(input_json, request);

  EXPECT_TRUE(true);
}
}
}
}