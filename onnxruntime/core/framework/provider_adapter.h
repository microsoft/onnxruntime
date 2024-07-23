// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class ExecutionProviderAdapter : public IExecutionProvider {
  public:
  ExecutionProviderAdapter(OrtExecutionProvider* ep) : IExecutionProvider(ep->type), ep_impl_(ep) {}
  private:
  OrtExecutionProvider* ep_impl_;
};
}
