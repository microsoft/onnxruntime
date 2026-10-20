// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/file_util.h"

namespace onnxruntime::test {

class ScopedWebGpuPluginRegistration {
 public:
  ScopedWebGpuPluginRegistration(Ort::Env& env, const char* registration_name)
      : env_(env), registration_name_(registration_name) {
    // Reuse the executable's registration so shared allocators and copies use the same factory.
    if (!GetEpDevices().empty()) {
      return;
    }
    const auto library_path = GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_webgpu"));
    env_.RegisterExecutionProviderLibrary(registration_name_.c_str(), library_path.c_str());
    owns_registration_ = true;
  }

  ~ScopedWebGpuPluginRegistration() {
    if (owns_registration_) {
      Ort::Status status{
          Ort::GetApi().UnregisterExecutionProviderLibrary(env_, registration_name_.c_str())};
      EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
    }
  }

  InlinedVector<Ort::ConstEpDevice> GetEpDevices() const {
    InlinedVector<Ort::ConstEpDevice> devices;
    for (const auto& candidate : env_.GetEpDevices()) {
      if (std::string_view{candidate.EpName()} == kWebGpuExecutionProvider) {
        devices.push_back(candidate);
      }
    }
    return devices;
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedWebGpuPluginRegistration);

  Ort::Env& env_;
  std::string registration_name_;
  bool owns_registration_{false};
};

}  // namespace onnxruntime::test
