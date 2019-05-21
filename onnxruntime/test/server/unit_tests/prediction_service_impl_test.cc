#include <iostream>

#include "gtest/gtest.h"

#include "server/executor.h"
#include "server/grpc/prediction_service_impl.h"
#include "test/test_environment.h"

namespace onnxruntime {
namespace server {
namespace grpc {
namespace test {

PredictRequest GetRequest() {
  PredictRequest req{};
  req.add_output_filter("Y");
  onnx::TensorProto proto{};
  proto.add_dims(3);
  proto.add_dims(2);
  proto.set_data_type(1);
  std::vector<float> floats = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::for_each(floats.begin(), floats.end(), [&](float& f) { proto.add_float_data(f); });
  (*req.mutable_inputs())["X"] = proto;
  return req;
}

std::shared_ptr<onnxruntime::server::ServerEnvironment> GetEnvironment() {
  const static auto model_file = "testdata/mul_1.pb";

  auto env = std::make_shared<onnxruntime::server::ServerEnvironment>(logging::Severity::kWARNING, logging::LoggingManager::InstanceType::Temporal, false);

  auto status = env->InitializeModel(model_file);
  EXPECT_TRUE(status.IsOK());

  status = env->GetSession()->Initialize();
  EXPECT_TRUE(status.IsOK());
  return env;
}

TEST(PredictionServiceImplTests, Test_OK) {
  auto env = GetEnvironment();
  PredictionServiceImpl test{env};
  auto request = GetRequest();
  PredictResponse resp{};
  ::grpc::ServerContext context{};
  auto status = test.Predict(&context, &request, &resp);
  EXPECT_TRUE(status.ok());
}

TEST(PredictionServiceImplTests, Test_fail_env_not_loaded) {
  auto env = std::make_shared<onnxruntime::server::ServerEnvironment>(logging::Severity::kWARNING, logging::LoggingManager::InstanceType::Temporal, false);
  PredictionServiceImpl test{env};
  auto request = GetRequest();
  PredictResponse resp{};
  ::grpc::ServerContext context{};
  auto status = test.Predict(&context, &request, &resp);
  EXPECT_FALSE(status.ok());
}

}  // namespace test
}  // namespace grpc
}  // namespace server
}  // namespace onnxruntime
