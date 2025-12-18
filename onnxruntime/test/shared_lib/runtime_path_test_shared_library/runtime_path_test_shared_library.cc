// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "runtime_path_test_shared_library.h"

#include <exception>
#include <iostream>

#include "core/platform/env.h"
#include "core/common/logging/logging.h"
#include "core/platform/logging/make_platform_default_log_sink.h"

namespace {

std::basic_string<PATH_CHAR_T> InitializeAndGetRuntimePath() {
  using namespace onnxruntime;

  const bool default_filter_user_data = false;
  const std::string default_logger_id = "DefaultLogger";
  auto logging_manager = logging::LoggingManager(logging::MakePlatformDefaultLogSink(),
                                                 logging::Severity::kVERBOSE,
                                                 default_filter_user_data,
                                                 logging::LoggingManager::InstanceType::Default,
                                                 &default_logger_id);

  auto runtime_path = Env::Default().GetRuntimePath();

  return runtime_path;
}

}  // namespace

extern "C" const PATH_CHAR_T* OrtTestGetSharedLibraryRuntimePath(void) {
#if !defined(ORT_NO_EXCEPTIONS)
  try {
#endif

    static const auto runtime_path = InitializeAndGetRuntimePath();
    return runtime_path.c_str();

#if !defined(ORT_NO_EXCEPTIONS)
  } catch (const std::exception& e) {
    std::cerr << __FUNCTION__ << " - caught exception: " << e.what() << "\n";
    return nullptr;
  }
#endif
}
