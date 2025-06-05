// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "utils.h"

class ExampleEp : public OrtEp, public ApiPtrs {
 public:
  ExampleEp(ApiPtrs apis, const std::string& name, const OrtSessionOptions& session_options, const OrtLogger& logger)
      : ApiPtrs(apis), name_{name}, session_options_{session_options}, logger_{logger} {
    // Initialize the execution provider's function table
    GetName = GetNameImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

    auto status = ort_api.Logger_LogMessage(&logger_,
                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                            ("ExampleEp has been created with name " + name_).c_str(),
                                            ORT_FILE, __LINE__, __FUNCTION__);
    // ignore status for now
    (void)status;
  }

  ~ExampleEp() {
    // Clean up the execution provider
  }

 private:
  static const char* GetNameImpl(const OrtEp* this_ptr);
  static OrtStatus* CreateSyncStreamForDeviceImpl(OrtEp* this_ptr, /*const OrtSession* session,*/
                                                  const OrtMemoryDevice* memory_device,
                                                  OrtSyncStream** stream);

  std::string name_;
  const OrtSessionOptions& session_options_;
  const OrtLogger& logger_;
};
