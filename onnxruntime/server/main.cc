// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "environment.h"
#include "http_server.h"
#include "predict_request_handler.h"
#include "server_configuration.h"
#include "grpc/grpc_app.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/syslog_sink.h>
#include <spdlog/fmt/ostr.h>

#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)
#define VAR_NAME_VALUE(var) #var "=" VALUE(var)

#define LOCAL_BUILD_VERSION "local_build"
#if !defined(SRV_VERSION)
#define SRV_VERSION LOCAL_BUILD_VERSION
#endif
#pragma message(VAR_NAME_VALUE(SRV_VERSION))

#define DEFAULT_COMMIT_ID "default"
#if !defined(LATEST_COMMIT_ID)
#define LATEST_COMMIT_ID DEFAULT_COMMIT_ID
#endif
#pragma message(VAR_NAME_VALUE(LATEST_COMMIT_ID))

namespace beast = boost::beast;
namespace http = beast::http;
namespace server = onnxruntime::server;

int main(int argc, char* argv[]) {
  // Here we use std::cout print out the version and latest commit id,
  // to make sure in case even logger has problem, we still have the version information and commit id.
  std::string version = SRV_VERSION;
  if (version.empty()) {
    version = LOCAL_BUILD_VERSION;
  }

  std::string commit_id = LATEST_COMMIT_ID;
  if (commit_id.empty()) {
    commit_id = DEFAULT_COMMIT_ID;
  }

  std::cout << "Version: " << version << std::endl;
  std::cout << "Commit ID: " << commit_id << std::endl;
  std::cout << std::endl;

  server::ServerConfiguration config{};
  auto res = config.ParseInput(argc, argv);

  if (res == server::Result::ExitSuccess) {
    exit(EXIT_SUCCESS);
  } else if (res == server::Result::ExitFailure) {
    exit(EXIT_FAILURE);
  }

  const auto env = std::make_shared<server::ServerEnvironment>(config.logging_level, spdlog::sinks_init_list{std::make_shared<spdlog::sinks::stdout_sink_mt>(), std::make_shared<spdlog::sinks::syslog_sink_mt>()});
  auto logger = env->GetAppLogger();
  logger->info("Model path: {}, ", config.model_path);
  logger->info("Model name: {}", config.model_name);
  logger->info("Model version: {}", config.model_version);

  try {
    env->InitializeModel(config.model_path, config.model_name, config.model_version);
    logger->debug("Initialize Model Successfully!");
  } catch (const Ort::Exception& ex) {
    logger->critical("Initialize Model Failed: {} ---- Error: [{}]", ex.GetOrtErrorCode(), ex.what());
    exit(EXIT_FAILURE);
  }

  //Setup GRPC Server
  auto const grpc_address = config.address;
  auto const grpc_port = config.grpc_port;

  server::GRPCApp grpc_app{env, grpc_address, grpc_port};

  logger->info("GRPC Listening at: {}:{}", grpc_address, grpc_port);

  //Setup HTTP Server
  auto const boost_address = boost::asio::ip::make_address(config.address);
  server::App app{};

  app.RegisterStartup(
      [&env](const auto& details) -> void {
        auto logger = env->GetAppLogger();
        logger->info("Listening at: http://{}:{}", details.address.to_string(), details.port);
      });

  app.RegisterError(
      [&env](auto& context) -> void {
        auto logger = env->GetLogger(context.request_id);
        logger->debug("Error code: {}", context.error_code);
        logger->debug("Error message: {}", context.error_message);

        context.response.result(context.error_code);
        context.response.insert("Content-Type", "application/json");
        context.response.insert(server::util::MS_REQUEST_ID_HEADER, context.request_id);
        if (!context.client_request_id.empty()) {
          context.response.insert(server::util::MS_CLIENT_REQUEST_ID_HEADER, (context).client_request_id);
        }
        context.response.body() = server::CreateJsonError(context.error_code, context.error_message);
      });

  app.RegisterPost(
      R"(/(?:v1/models/([^/:]+)(?:/versions/(\d+))?:(classify|regress|predict))|(?:score()()()))",
      [&env](const auto& name, const auto& version, const auto& action, auto& context) -> void {
        server::Predict(name, version, action, context, env);
      });

  app.RegisterPost(
    R"(/score()()())",
     [&env](const auto& name, const auto& version, const auto& action, auto& context) -> void {
        server::Predict(name, version, action, context, env);
      }
  );

  app.Bind(boost_address, config.http_port)
      .NumThreads(config.num_http_threads)
      .Run();

  grpc_app.Run();

  return EXIT_SUCCESS;
}
