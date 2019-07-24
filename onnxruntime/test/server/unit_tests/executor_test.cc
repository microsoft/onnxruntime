// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "gtest/gtest.h"

#include "server/executor.h"
#include "server/http/json_handling.h"
#include "test/test_environment.h"

namespace onnxruntime {
namespace server {
namespace test {

TEST(ExecutorTests, TestMul_1) {
  const static auto model_file = "testdata/mul_1.pb";
  const static auto input_json = R"({"inputs":{"X":{"dims":[3,2],"dataType":1,"floatData":[1,2,3,4,5,6]}},"outputFilter":["Y"]})";
  const static auto expected = R"({"outputs":{"Y":{"dims":["3","2"],"dataType":1,"floatData":[1,4,9,16,25,36]}}})";

  onnxruntime::server::ServerEnvironment env(logging::Severity::kWARNING, logging::LoggingManager::InstanceType::Temporal, false);

  auto status = env.InitializeModel(model_file);
  EXPECT_TRUE(status.IsOK());

  status = env.GetSession()->Initialize();
  EXPECT_TRUE(status.IsOK());

  onnxruntime::server::Executor executor(&env, "RequestId");
  onnxruntime::server::PredictRequest request{};
  onnxruntime::server::PredictResponse response{};

  auto protostatus = onnxruntime::server::GetRequestFromJson(input_json, request);
  EXPECT_TRUE(protostatus.ok());

  auto prediction_res = executor.Predict("Name", "version", request, response);
  EXPECT_TRUE(prediction_res.ok());

  std::string body;
  protostatus = GenerateResponseInJson(response, body);
  EXPECT_EQ(expected, body);
}

}  // namespace test
}  // namespace server
}  // namespace onnxruntime
