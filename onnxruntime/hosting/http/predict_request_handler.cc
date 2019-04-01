// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <google/protobuf/stubs/status.h>

#include "environment.h"
#include "http_server.h"
#include "json_handling.h"
#include "executor.h"
#include "util.h"

namespace onnxruntime {
namespace hosting {

namespace protobufutil = google::protobuf::util;

#define GenerateErrorResponse(logger, error_code, status, context)                   \
  {                                                                                  \
    auto http_error_code = (error_code);                                             \
    auto error_message = CreateJsonError(http_error_code, (status).error_message()); \
    LOGS((*logger), VERBOSE) << error_message;                                       \
    (context).response.result(http_error_code);                                      \
    (context).response.body() = error_message;                                       \
    (context).response.set(http::field::content_type, "application/json");           \
  }

// TODO: decide whether this should be a class
void Predict(const std::string& name,
             const std::string& version,
             const std::string& action,
             /* in, out */ HttpContext& context,
             std::shared_ptr<HostingEnvironment> env) {
  auto logger = env->GetLogger(context.uuid);
  LOGS(*logger, VERBOSE) << "Name: " << name << " Version: " << version << " Action: " << action;

  auto body = context.request.body();
  PredictRequest predictRequest{};
  auto status = GetRequestFromJson(body, predictRequest);
  if (!status.ok()) {
    GenerateErrorResponse(logger, GetHttpStatusCode((status)), status, context);
    return;
  }

  Executor executor(env);
  PredictResponse predictResponse{};
  status = executor.Predict(name, version, "request_id", predictRequest, predictResponse);
  if (!status.ok()) {
    GenerateErrorResponse(logger, GetHttpStatusCode((status)), status, context);
    return;
  }

  std::string response_body{};
  status = GenerateResponseInJson(predictResponse, response_body);
  if (!status.ok()) {
    GenerateErrorResponse(logger, http::status::internal_server_error, status, context);
    return;
  }

  context.response.body() = response_body;
  context.response.result(http::status::ok);
  context.response.set(http::field::content_type, "application/json");
};

}  // namespace hosting
}  // namespace onnxruntime
