#include "core/framework/session_options.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/cerr_sink.h"

namespace onnxruntime {
namespace python {

using namespace onnxruntime;
using namespace onnxruntime::logging;

static const SessionOptions& GetDefaultCPUSessionOptions() {
  static SessionOptions so;
  return so;
}

class SessionObjectInitializer {
 public:
  typedef const SessionOptions& Arg1;
  typedef logging::LoggingManager* Arg2;
  operator Arg1() {
    return GetDefaultCPUSessionOptions();
  }

  operator Arg2() {
    static std::string default_logger_id{"Default"};
    static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CErrSink{}},
                                                  Severity::kWARNING, false, LoggingManager::InstanceType::Default,
                                                  &default_logger_id};
    return &default_logging_manager;
  }

  static SessionObjectInitializer Get() {
    return SessionObjectInitializer();
  }
};
}
}

