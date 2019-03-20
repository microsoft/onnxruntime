// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "environment.h"
#include "http_server.h"
#include "predict_request_handler.h"
#include "server_configuration.h"

namespace beast = boost::beast;
namespace http = beast::http;
namespace hosting = onnxruntime::hosting;

int main(int argc, char* argv[]) {
  hosting::ServerConfiguration config{};
  auto res = config.ParseInput(argc, argv);

  if (res == hosting::Result::ExitSuccess) {
    exit(EXIT_SUCCESS);
  } else if (res == hosting::Result::ExitFailure) {
    exit(EXIT_FAILURE);
  }

  hosting::HostingEnvironment env;
  auto logger = env.GetLogger();

  // TODO: below code snippet just trying to show case how to use the "env". Move later.
  LOGS(logger, VERBOSE) << "Logging manager initialized.";
  LOGS(logger, VERBOSE) << "Model path: " << config.model_path;
  auto status = env.GetSession()->Load(config.model_path);
  LOGS(logger, VERBOSE) << "Load Model Status: " << status.Code() << " ---- Error: [" << status.ErrorMessage() << "]";

  auto const boost_address = boost::asio::ip::make_address(config.address);

  hosting::App app{};
  app.Post(R"(/v1/models/([^/:]+)(?:/versions/(\d+))?:(classify|regress|predict))",
           [&env](const std::string& name, const std::string& version, const std::string& action, hosting::HttpContext& context) {
             hosting::Predict(name, version, action, context, env);
           });

  app.Bind(boost_address, config.http_port)
      .NumThreads(config.num_http_threads)
      .Run();

  return EXIT_SUCCESS;
}
