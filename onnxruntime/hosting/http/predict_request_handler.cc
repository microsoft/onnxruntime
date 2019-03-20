// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "environment.h"
#include "http_server.h"
#include "json_handling.h"

namespace onnxruntime {
namespace hosting {

namespace beast = boost::beast;
namespace http = beast::http;

void BadRequest(HttpContext& context, const std::string& error_message) {
  auto json_error = R"({"error_code": 400, "error_message": )" + error_message + " }";

  http::response<http::string_body> res{http::status::bad_request, context.request.version()};
  res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
  res.set(http::field::content_type, "application/json");
  res.keep_alive(context.request.keep_alive());
  res.body() = std::string(json_error);
  res.prepare_payload();
  context.response = res;
}

// TODO: decide whether this should be a class
void Predict(const std::string& name,
             const std::string& version,
             const std::string& action,
             HttpContext& context,
             HostingEnvironment& env) {
  PredictRequest predictRequest{};
  auto logger = env.GetLogger();

  LOGS(logger, VERBOSE) << "Name: " << name
                        << "Version: " << version
                        << "Action: " << action;

  auto body = context.request.body();
  auto status = GetRequestFromJson(body, predictRequest);

  if (!status.ok()) {
    return BadRequest(context, status.error_message());
  }

  http::response<http::string_body> res{std::piecewise_construct,
                                        std::make_tuple(body),
                                        std::make_tuple(http::status::ok, context.request.version())};
  res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
  res.set(http::field::content_type, "application/json");
  res.keep_alive(context.request.keep_alive());
  context.response = res;
};

}  // namespace hosting
}  // namespace onnxruntime
