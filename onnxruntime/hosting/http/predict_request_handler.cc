// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "environment.h"
#include "http_server.h"
#include "json_handling.h"

namespace onnxruntime {
namespace hosting {

// TODO: decide whether this should be a class
void Predict(const std::string& name,
             const std::string& version,
             const std::string& action,
             HttpContext& context,
             std::shared_ptr<HostingEnvironment> env) {
  PredictRequest predictRequest{};
  auto logger = env->GetLogger(context.uuid);

  LOGS(*logger, VERBOSE) << "Name: " << name << " Version: " << version  << " Action: " << action;

  auto body = context.request.body();
  auto status = GetRequestFromJson(body, predictRequest);

  if (!status.ok()) {
    context.response.result(400);
    context.response.body() = CreateJsonError(http::status::bad_request, status.error_message());
    return;
  }

  context.response.result(200);
  context.response.body() = body;
};

}  // namespace hosting
}  // namespace onnxruntime
