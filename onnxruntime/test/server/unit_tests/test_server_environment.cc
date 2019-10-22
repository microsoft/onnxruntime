#include "test_server_environment.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_sinks.h>

namespace onnxruntime {
namespace server {
namespace test {
static std::unique_ptr<onnxruntime::server::ServerEnvironment> s_env;

ServerEnvironment* ServerEnv() {
  ORT_ENFORCE(s_env != nullptr,
              "Need a TestServerEnvironment instance to provide the server environment.");

  return s_env.get();
}

TestServerEnvironment::TestServerEnvironment() {
  auto console = spdlog::stdout_logger_mt("console");
  spdlog::set_default_logger(console);
  spdlog::sink_ptr ptr = std::make_shared<spdlog::sinks::stdout_sink_st>();
  s_env = onnxruntime::make_unique<onnxruntime::server::ServerEnvironment>(ORT_LOGGING_LEVEL_WARNING, spdlog::sinks_init_list{ptr});
}
TestServerEnvironment::~TestServerEnvironment() {
  //destruct env to make sure the default logger is destoryed before the logger mutex.
  s_env = nullptr;
}

}  // namespace test
}  // namespace server
}  // namespace onnxruntime