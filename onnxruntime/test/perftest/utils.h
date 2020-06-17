// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <sstream>
#include "onnxruntime_c_api.h"

extern const OrtApi* c_api;

namespace onnxruntime {
namespace perftest {
inline void ThrowOnError(OrtStatus* status) {
  if (status) {
    std::ostringstream  oss;
    oss << c_api->GetErrorMessage(status) << ", error code" <<c_api->GetErrorCode(status);
    c_api->ReleaseStatus(status);
    throw std::runtime_error(oss.str());
  }
}

namespace utils {


size_t GetPeakWorkingSetSize();

class ICPUUsage {
 public:
  virtual ~ICPUUsage() = default;

  virtual short GetUsage() const = 0;

  virtual void Reset() = 0;
};

std::unique_ptr<ICPUUsage> CreateICPUUsage();

}  // namespace utils
}  // namespace perftest
}  // namespace onnxruntime
