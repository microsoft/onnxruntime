// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/framework/allocator.h"
#include "core/framework/session_options.h"
#include "core/session/environment.h"
#include "core/platform/env.h"

namespace onnxruntime {
class InferenceSession;

namespace python {

using namespace onnxruntime;
using namespace onnxruntime::logging;

struct OrtStatus {
  OrtErrorCode code;
  char msg[1];  // a null-terminated string
};

struct CustomOpLibrary {
  CustomOpLibrary(const char* library_path, OrtSessionOptions& ort_so) {
    Env::Default().LoadDynamicLibrary(library_path, &library_handle_);

    if (!library_handle_)
      throw std::runtime_error("RegisterCustomOpsLibrary: Failed to load library");

    OrtStatus*(_stdcall * RegisterCustomOps)(OrtSessionOptions * options, const OrtApiBase* api);

    Env::Default().GetSymbolFromLibrary(library_handle_, "RegisterCustomOps", (void**)&RegisterCustomOps);

    if (!RegisterCustomOps)
      throw std::runtime_error("RegisterCustomOpsLibrary: Entry point RegisterCustomOps not found in library");

    auto* status = RegisterCustomOps(&ort_so, OrtGetApiBase());

    if (status) {
      // A non-nullptr indicates some error
      // Free status and throw
      Env::Default().UnloadDynamicLibrary(library_handle_);
      ::free(status);
      throw std::runtime_error("TODO");
    }

    // No status to free if it is a nullptr
  }
  ~CustomOpLibrary() {
    Env::Default().UnloadDynamicLibrary(library_handle_);
  }

  CustomOpLibrary(CustomOpLibrary&& other) = delete;

  CustomOpLibrary& operator=(CustomOpLibrary&& other) = delete;

  void* library_handle_ = nullptr;
};

struct PySessionOptions : public SessionOptions {
  // Have the life cycle of the OrtCustomOpDomain pointers managed by a smart pointer
  std::vector<std::shared_ptr<OrtCustomOpDomain>> custom_op_domains_;

  std::vector<std::shared_ptr<CustomOpLibrary>> custom_op_libraries_;
};

inline const PySessionOptions& GetDefaultCPUSessionOptions() {
  static PySessionOptions so;
  return so;
}

inline AllocatorPtr& GetAllocator() {
  static AllocatorPtr alloc = std::make_shared<TAllocator>();
  return alloc;
}

class SessionObjectInitializer {
 public:
  typedef const PySessionOptions& Arg1;
  // typedef logging::LoggingManager* Arg2;
  static const std::string default_logger_id;
  operator Arg1() {
    return GetDefaultCPUSessionOptions();
  }

  // operator Arg2() {
  //   static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CErrSink{}},
  //                                                 Severity::kWARNING, false, LoggingManager::InstanceType::Default,
  //                                                 &default_logger_id};
  //   return &default_logging_manager;
  // }

  static SessionObjectInitializer Get() {
    return SessionObjectInitializer();
  }
};

Environment& get_env();

void InitializeSession(InferenceSession* sess, const std::vector<std::string>& provider_types);

}  // namespace python
}  // namespace onnxruntime
