#include "ep.h"
#include "ep_factory.h"
#include "ep_stream_support.h"

ExampleEp::ExampleEp(ExampleEpFactory& factory, const std::string& name,
                     const OrtSessionOptions& session_options, const OrtLogger& logger)
    : ApiPtrs(static_cast<const ApiPtrs&>(factory)),
      factory_{factory},
      name_{name},
      session_options_{session_options},
      logger_{logger} {
  // Initialize the execution provider's function table
  GetName = GetNameImpl;

  auto status = ort_api.Logger_LogMessage(&logger_,
                                          OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                          ("ExampleEp has been created with name " + name_).c_str(),
                                          ORT_FILE, __LINE__, __FUNCTION__);
  // ignore status for now
  (void)status;
}

/*static*/
const char* ExampleEp ::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const ExampleEp*>(this_ptr);
  return ep->name_.c_str();
}
