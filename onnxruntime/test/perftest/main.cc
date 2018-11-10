// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/common/logging/sinks/clog_sink.h>
#include <core/common/logging/logging.h>
#include <core/framework/environment.h>
#include <core/platform/env.h>

#include "command_args_parser.h"
#include "performance_runner.h"

using namespace onnxruntime;

int main(int argc, char* args[]) {
  std::string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kWARNING, false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  std::unique_ptr<Environment> env;
  auto status = Environment::Create(env);
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "failed to create environment:%s", status.ErrorMessage().c_str());
    return -1;
  }

  ::onnxruntime::perftest::PerformanceTestConfig test_config;
  if (!::onnxruntime::perftest::CommandLineParser::ParseArguments(test_config, argc, args)) {
    ::onnxruntime::perftest::CommandLineParser::ShowUsage();
    return -1;
  }

  ::onnxruntime::perftest::PerformanceRunner perf_runner(test_config);
  status = perf_runner.Run();
  if (!status.IsOK()) {
    LOGF_DEFAULT(ERROR, "Run failed:%s", status.ErrorMessage().c_str());
    return -1;
  }

  perf_runner.SerializeResult();

  return 0;
}
