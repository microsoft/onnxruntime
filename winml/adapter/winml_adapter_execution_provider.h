// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {
class IExecutionProvider;
}

struct OrtExecutionProvider {
 public:
  static OrtStatus* CreateProvider(const std::string provider_id, OrtExecutionProvider** out);

 private:
  OrtExecutionProvider(const std::string provider_id);
  OrtExecutionProvider(const OrtExecutionProvider& other) = delete;
  OrtExecutionProvider& operator=(const OrtExecutionProvider& other) = delete;

  const std::string provider_id_;
};
