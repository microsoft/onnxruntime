// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/framework/allocator.h"
#include "core/framework/session_options.h"
#include "core/session/environment.h"

namespace onnxruntime {
class InferenceSession;

namespace python {

using namespace onnxruntime;
using namespace onnxruntime::logging;

struct CustomOpLibrary {
  CustomOpLibrary(const char* library_path, OrtSessionOptions& ort_so);

  ~CustomOpLibrary();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpLibrary);

 private:
  void UnloadLibrary();
  void* library_handle_ = nullptr;
};

// Thin wrapper over internal C++ SessionOptions to accommodate custom op library management for the Python user
struct PySessionOptions : public SessionOptions {
  // Hold CustomOpLibrary resources so as to tie it to the life cycle of the InferenceSession needing it.
  std::vector<std::shared_ptr<CustomOpLibrary>> custom_op_libraries_;

  // Hold raw `OrtCustomOpDomain` pointers - it is upto the shared library to release the OrtCustomOpDomains
  // that was created when the library is unloaded
  std::vector<OrtCustomOpDomain*> custom_op_domains_;
};

// Thin wrapper over internal C++ InferenceSession to accommodate custom op library management for the Python user
struct PyInferenceSession {
  // Hold CustomOpLibrary resources so as to tie it to the life cycle of the InferenceSession needing it.
  // NOTE: Declare this above InferenceSession so that this is destructed AFTER the InferenceSession instance -
  // this is so that the custom ops held by the InferenceSession get destroyed prior to the library getting unloaded
  // (if ref count of the shared_ptr reaches 0)
  std::vector<std::shared_ptr<CustomOpLibrary>> custom_op_libraries_;

  std::unique_ptr<InferenceSession> sess_;

  virtual InferenceSession* GetSessionHandle() const { return sess_.get(); }
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

Environment& GetEnv();

void InitializeSession(InferenceSession* sess, const std::vector<std::string>& provider_types);

}  // namespace python
}  // namespace onnxruntime
