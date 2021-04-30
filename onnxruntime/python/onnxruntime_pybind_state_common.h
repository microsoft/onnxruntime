// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/cerr_sink.h"
#include "core/framework/allocator.h"
#include "core/framework/session_options.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"

namespace onnxruntime {
namespace python {

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
struct CustomOpLibrary {
  CustomOpLibrary(const char* library_path, OrtSessionOptions& ort_so);

  ~CustomOpLibrary();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CustomOpLibrary);

 private:
  void UnloadLibrary();

  std::string library_path_;
  void* library_handle_ = nullptr;
};
#endif

// Thin wrapper over internal C++ SessionOptions to accommodate custom op library management for the Python user
struct PySessionOptions : public SessionOptions {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // `PySessionOptions` has a vector of shared_ptrs to CustomOpLibrary, because so that it can be re-used for all
  // `PyInferenceSession`s using the same `PySessionOptions` and that each `PyInferenceSession` need not construct
  // duplicate CustomOpLibrary instances.
  std::vector<std::shared_ptr<CustomOpLibrary>> custom_op_libraries_;

  // Hold raw `OrtCustomOpDomain` pointers - it is upto the shared library to release the OrtCustomOpDomains
  // that was created when the library is unloaded
  std::vector<OrtCustomOpDomain*> custom_op_domains_;
#endif
};

// Thin wrapper over internal C++ InferenceSession to accommodate custom op library management for the Python user
struct PyInferenceSession {
  PyInferenceSession(Environment& env, const PySessionOptions& so) {
    sess_ = std::make_unique<InferenceSession>(so, env);
  }

#if !defined(ORT_MINIMAL_BUILD)
  PyInferenceSession(Environment& env, const PySessionOptions& so, const std::string& arg, bool is_arg_file_name) {
    if (is_arg_file_name) {
      // Given arg is the file path. Invoke the corresponding ctor().
      sess_ = std::make_unique<InferenceSession>(so, env, arg);
    } else {
      // Given arg is the model content as bytes. Invoke the corresponding ctor().
      std::istringstream buffer(arg);
      sess_ = std::make_unique<InferenceSession>(so, env, buffer);
    }
  }
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  void AddCustomOpLibraries(const std::vector<std::shared_ptr<CustomOpLibrary>>& custom_op_libraries) {
    if (!custom_op_libraries.empty()) {
      custom_op_libraries_.reserve(custom_op_libraries.size());
      for (size_t i = 0; i < custom_op_libraries.size(); ++i) {
        custom_op_libraries_.push_back(custom_op_libraries[i]);
      }
    }
  }
#endif

  InferenceSession* GetSessionHandle() const { return sess_.get(); }

  virtual ~PyInferenceSession() {}

 protected:
  PyInferenceSession(std::unique_ptr<InferenceSession> sess) {
    sess_ = std::move(sess);
  }

 private:
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // Hold CustomOpLibrary resources so as to tie it to the life cycle of the InferenceSession needing it.
  // NOTE: Define this above `sess_` so that this is destructed AFTER the InferenceSession instance -
  // this is so that the custom ops held by the InferenceSession gets destroyed prior to the library getting unloaded
  // (if ref count of the shared_ptr reaches 0)
  std::vector<std::shared_ptr<CustomOpLibrary>> custom_op_libraries_;
#endif

  std::unique_ptr<InferenceSession> sess_;
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

// Initialize an InferenceSession.
// Any provider_options should have entries in matching order to provider_types.
void InitializeSession(InferenceSession* sess,
                       const std::vector<std::string>& provider_types = {},
                       const ProviderOptionsVector& provider_options = {},
                       const std::unordered_set<std::string>& disabled_optimizer_names = {});

// Checks if PyErrOccured, fetches status and throws.
void ThrowIfPyErrOccured();

}  // namespace python
}  // namespace onnxruntime
