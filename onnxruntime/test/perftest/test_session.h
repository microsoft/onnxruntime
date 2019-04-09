// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
namespace onnxruntime {
namespace perftest {
class TestSession {
 public:
  virtual std::chrono::duration<double> Run(const OrtValue* const* input) = 0;
  virtual ~TestSession() = default;
};
}  // namespace perftest
}  // namespace onnxruntime