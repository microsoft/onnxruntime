// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace onnxruntime {

class DeferredRunInstrumentationRecord {
 public:
  virtual ~DeferredRunInstrumentationRecord() = default;

  // Returns an empty string on success or an error message on failure.
  virtual std::string Emit() = 0;
};

}  // namespace onnxruntime
