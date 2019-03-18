// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "http_server.h"
#include "server_configuration.h"
#include "environment.h"

namespace beast = boost::beast;
namespace http = beast::http;

void test_request(const std::string& name, const std::string& version,
                  const std::string& action, onnxruntime::hosting::HttpContext& context) {
  std::stringstream ss;

  ss << "\tModel Name: " << name << std::endl;
  ss << "\tModel Version: " << version << std::endl;
  ss << "\tAction: " << action << std::endl;
  ss << "\tHTTP method: " << context.request.method() << std::endl;

  http::response<http::string_body>
      res{std::piecewise_construct, std::make_tuple(ss.str()), std::make_tuple(http::status::ok, context.request.version())};

  res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
  res.set(http::field::content_type, "plain/text");
  res.keep_alive(context.request.keep_alive());
  context.response = res;
}

int main(int argc, char* argv[]) {
  onnxruntime::hosting::ServerConfiguration config{};
  auto res = config.ParseInput(argc, argv);

  if (res == onnxruntime::hosting::Result::ExitSuccess) {
    exit(EXIT_SUCCESS);
  } else if (res == onnxruntime::hosting::Result::ExitFailure) {
    exit(EXIT_FAILURE);
  }

  onnxruntime::hosting::HostingEnvironment env;
  auto logger = env.GetLogger();

  // TODO: below code snippet just trying to show case how to use the "env".
  //       Will be moved to proper place.
  LOGS(logger, VERBOSE) << "Logging manager initialized.";
  LOGS(logger, VERBOSE) << "Model path: " << config.model_path;
  auto status = env.GetSession()->Load(config.model_path);
  LOGS(logger, VERBOSE) << "Load Model Status: " << status.Code() << " ---- Error: [" << status.ErrorMessage() << "]";

  auto const boost_address = boost::asio::ip::make_address(config.address);

  onnxruntime::hosting::App app{};
  app.Post(R"(/v1/models/([^/:]+)(?:/versions/(\d+))?:(classify|regress|predict))", test_request)
     .Bind(boost_address, config.http_port)
     .NumThreads(config.num_http_threads)
     .Run();

  return EXIT_SUCCESS;
}
