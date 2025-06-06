// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "utils.h"

class ExampleEpFactory;

class ExampleEp : public OrtEp, public ApiPtrs {
 public:
  ExampleEp(ExampleEpFactory& factory, const std::string& name,
            const OrtSessionOptions& session_options, const OrtLogger& logger);

  ~ExampleEp() = default;

 private:
  static const char* GetNameImpl(const OrtEp* this_ptr);
  static OrtStatus* CreateSyncStreamForDeviceImpl(OrtEp* this_ptr, /*const OrtSession* session,*/
                                                  const OrtMemoryDevice* memory_device,
                                                  OrtSyncStream** stream);

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEp* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

  static void ORT_API_CALL ReleaseDataTransferImpl(OrtEp* /*this_ptr*/,
                                                   OrtDataTransferImpl* /*data_transfer*/) noexcept {
    // no-op. the factory owns the instance
  }

  ExampleEpFactory& factory_;
  std::string name_;
  const OrtSessionOptions& session_options_;
  const OrtLogger& logger_;
};
